"""
效率测量：参数量 / MACs / 延迟 / 峰值显存
==========================================
回答审稿人 R1-4、R2-2、R2-3：
  - 不把 GPU 显存当成 NPU 内存来叙述
  - 每个配置在**独立子进程**里测显存，避免 caching allocator 的残留污染
  - MACs 与 FLOPs 分开报（FLOPs = 2 x MACs）

用法：
    python probe_eff.py --run runs/av_rf                  # 单配置
    python probe_eff.py --run runs/av_rf runs/en_rf ...   # 多配置汇总成表
    python probe_eff.py --one runs/av_rf --gpu 0          # 内部用：单配置单进程
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from config import config
from utils import apply_run_config

FRAME_MS = 20.0   # 2 s / 1000 帧（stride=16 时）


class MacCounter:
    """前向钩子统计乘加次数（MACs），不依赖第三方库"""

    def __init__(self):
        self.total = 0

    def attach(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                m.register_forward_hook(self._conv)
            elif isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                m.register_forward_hook(self._convt)
            elif isinstance(m, nn.Linear):
                m.register_forward_hook(self._linear)

    def _conv(self, mod, inp, out):
        k = 1
        for s in mod.kernel_size:
            k *= s
        self.total += out.numel() * (mod.in_channels // mod.groups) * k

    def _convt(self, mod, inp, out):
        k = 1
        for s in mod.kernel_size:
            k *= s
        self.total += out.numel() * (mod.out_channels // mod.groups) * k

    def _linear(self, mod, inp, out):
        self.total += out.numel() * mod.in_features


@torch.no_grad()
def measure(run_dir, gpu="0", iters=30, warmup=5):
    from models import build_model
    info = apply_run_config(run_dir)
    kind = info["kind"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = build_model(kind, config).to(device).eval()
    ck = Path(run_dir) / "best.pt"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])

    params = sum(p.numel() for p in model.parameters())
    mix = torch.randn(1, 1, config.clip_samples, device=device)
    vid = torch.randint(0, 255, (1, 1, config.clip_frames, config.mouth_size, config.mouth_size),
                        dtype=torch.uint8, device=device)
    enr = torch.randn(1, config.clip_samples, device=device)

    # ---- MACs（一次前向，fp32）----
    counter = MacCounter()
    counter.attach(model)
    from models import NEEDS_ENROLL, forward_batch
    forward_batch(model, kind, mix, vid, enr if kind in NEEDS_ENROLL else None)
    macs = counter.total

    # ---- 延迟（AMP，与训练/部署一致）----
    use_amp = device.type == "cuda"
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            forward_batch(model, kind, mix, vid, enr if kind in NEEDS_ENROLL else None)
    torch.cuda.synchronize() if device.type == "cuda" else None
    lat = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            forward_batch(model, kind, mix, vid, enr if kind in NEEDS_ENROLL else None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)

    lat_sorted = sorted(lat)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))]  # noqa: E731

    peak_alloc = peak_res = None
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
        peak_res = torch.cuda.max_memory_reserved() / 2 ** 20

    # 音频帧率 = 采样率 / 编码器 stride
    return {
        "run": Path(run_dir).name,
        "kind": kind,
        "params": params,
        "macs_g": round(macs / 1e9, 3),
        "flops_g": round(2 * macs / 1e9, 3),
        "enc_stride": config.enc_stride,
        "frame_rate_hz": round(config.sample_rate / config.enc_stride, 1),
        "lat_mean_ms": round(float(sum(lat) / len(lat)), 2),
        "lat_p50_ms": round(p(0.50), 2),
        "lat_p95_ms": round(p(0.95), 2),
        "peak_alloc_mb": round(peak_alloc, 2) if peak_alloc else None,
        "peak_reserved_mb": round(peak_res, 2) if peak_res else None,
        "device": str(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--one", default="")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    if a.one:                                   # 子进程模式：单配置，写临时 json
        r = measure(a.one, a.gpu)
        print(json.dumps(r, ensure_ascii=False))
        return

    rows = []
    for run_dir in a.run:
        # 每个配置起独立进程 —— 显存峰值必须干净
        out = subprocess.run(
            [sys.executable, __file__, "--one", run_dir, "--gpu", a.gpu],
            capture_output=True, text=True, check=False)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
        if not line:
            print(f"[跳过] {run_dir}: {out.stderr.strip()[-300:]}")
            continue
        r = json.loads(line[-1])
        rows.append(r)
        print(f"[{r['run']}] {r['params']/1e6:.2f}M  {r['macs_g']:.2f} GMAC  "
              f"{r['frame_rate_hz']:.0f} Hz  P50={r['lat_p50_ms']:.1f} ms  "
              f"peak={r['peak_alloc_mb']} MB")

    config.result_dir.mkdir(parents=True, exist_ok=True)
    (config.result_dir / "efficiency.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))

    head = ("| 配置 | 线索 | 参数量 | MACs (G) | FLOPs (G) | 帧率 (Hz) | "
            "延迟 P50 (ms) | 延迟 P95 (ms) | 峰值显存 (MB) |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['kind']} | {r['params']/1e6:.2f} M | {r['macs_g']:.2f} | "
                     f"{r['flops_g']:.2f} | {r['frame_rate_hz']:.0f} | {r['lat_p50_ms']:.1f} | "
                     f"{r['lat_p95_ms']:.1f} | {r['peak_alloc_mb']} |")
    (config.result_dir / "efficiency.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
