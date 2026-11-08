"""
分块（流式近似）推理评估（审稿人 R2-4：streaming / causal evaluation）
=====================================================================
模型是卷积式 TCN，本身不含因果掩码。这里用最朴素的分块协议量化"以有限
延迟运行"要付出多少代价：

    对每个块 [t, t+C):  输入 [t-R, t+C+R) 的音频与其对齐视频帧，
                        只保留中间 C 秒的输出，拼接成整句
    C = 块长（决定主要延迟），R = 双侧上下文（补偿卷积感受野的边界效应）

C = 整句长度时退化为整句推理（上界）。报告 SI-SNRi 随 C 的变化即"延迟—精度"曲线。

用法：
    python stream_eval.py --ckpt runs/av_rf/best.pt --gpu 0 --chunks 0.5,1.0,2.0
"""
import argparse
import json

import numpy as np
import torch

from config import config
from dataset import make_loader
from utils import load_run_model, si_snri

SR = 16000


def frames_for(a, b):
    """采样点区间 [a, b) 对应的视频帧区间（25 fps）"""
    fps = config.fps
    return int(round(a / SR * fps)), int(round(b / SR * fps))


@torch.no_grad()
def infer_blocks(model, mix, vid, chunk_s, ctx_s, use_extra=None):
    """按块推理并拼接输出；mix:[B,1,T] vid:[B,1,F,H,W]"""
    C = int(chunk_s * SR)
    R = int(ctx_s * SR)
    T = mix.shape[-1]
    est = torch.zeros_like(mix)
    for t0 in range(0, T, C):
        t1 = min(t0 + C, T)
        a0, a1 = max(0, t0 - R), min(T, t1 + R)
        f0, f1 = frames_for(a0, a1)
        seg_mix = mix[..., a0:a1]
        seg_vid = vid[:, :, f0:max(f1, f0 + 1)]
        if use_extra is not None:
            out, _ = model(seg_mix, seg_vid, use_extra)
        else:
            out, _ = model(seg_mix, seg_vid)
        out = out.float()
        lo, hi = t0 - a0, t1 - a0
        est[..., t0:t1] = out[..., lo:hi]
    return est


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/av_rf/best.pt")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--chunks", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--ctx", default="0.5")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    model, kind, _ = load_run_model(a.ckpt, device)
    loader = make_loader(config.cache_dir, a.split, config, deterministic=True,
                         limit=a.limit, shuffle=False, batch_size=4)

    chunks = [float(x) for x in a.chunks.split(",")]
    ctxs = [float(x) for x in a.ctx.split(",")]
    acc = {(c, r): [] for c in chunks for r in ctxs}
    full = []

    n_done = 0
    for batch in loader:
        mix = batch["mix"].to(device)
        tgt = batch["target"].to(device)
        vid = batch["video"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            est, _ = model(mix, vid)
        full.extend(si_snri(est.float(), tgt, mix).cpu().numpy().tolist())
        for c in chunks:
            for r in ctxs:
                if c >= config.clip_seconds:
                    acc[(c, r)].extend(si_snri(est.float(), tgt, mix).cpu().numpy().tolist())
                    continue
                e = infer_blocks(model, mix, vid, c, r)
                acc[(c, r)].extend(si_snri(e, tgt, mix).cpu().numpy().tolist())
        n_done += mix.size(0)
        print(f"  {n_done} 条", end="\r", flush=True)

    out = {"full_utterance": {"si_snri": round(float(np.mean(full)), 3),
                              "std": round(float(np.std(full)), 3), "n": len(full)},
           "blocks": {}}
    for (c, r), v in acc.items():
        out["blocks"][f"C={c}s,R={r}s"] = {
            "chunk_s": c, "ctx_s": r,
            "latency_s": round(c + r, 3),       # 块长 + 右侧上下文 = 算法延迟
            "si_snri": round(float(np.mean(v)), 3),
            "std": round(float(np.std(v)), 3),
            "n": len(v),
        }
    tag = a.ckpt.split("/")[-2] if "/" in a.ckpt else a.ckpt
    (config.result_dir / f"stream_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))

    print(f"\n[{tag}] 整句: {out['full_utterance']['si_snri']:+.2f} dB")
    for k, v in out["blocks"].items():
        print(f"  {k:16s} 延迟 {v['latency_s']:.2f}s  SI-SNRi {v['si_snri']:+.2f} dB")


if __name__ == "__main__":
    main()
