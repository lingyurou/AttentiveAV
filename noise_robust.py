"""
未见噪声条件下的零样本鲁棒性（审稿人 R1-2(3) / R2-4：单数据集、无真实环境）
==========================================================================
不重训，只在测试时把额外噪声加到混合信号上，检验"视觉线索"是否比
"注册语音线索"更抗噪。三类噪声：
    white   高斯白噪声
    babble  同批次其他测试语音叠加（模拟多人交谈背景）
    comb    白噪声 + 短梳状滤波（粗糙的混响代理，不代表任何真实房间）

用法：
    python noise_robust.py --ckpt runs/av_rf/best.pt --ckpt runs/en_rf/best.pt --gpu 0
"""
import argparse
import json

import numpy as np
import torch

from config import config
from dataset import make_loader
from utils import load_run_model, si_snri

SNRS = [10.0, 5.0, 0.0]


def add_noise(mix, kind, snr_db, batch_mix=None):
    if kind == "white":
        n = torch.randn_like(mix)
    elif kind == "babble":
        if batch_mix is None or batch_mix.size(0) < 2:
            n = torch.randn_like(mix)
        else:
            n = torch.roll(batch_mix, shifts=1, dims=0)
    else:  # comb：白噪声过梳状滤波，粗代理"混响 + 稳态噪声"
        n = torch.randn_like(mix)
        k = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35],
                         device=mix.device, dtype=mix.dtype).view(1, 1, -1)
        n = torch.nn.functional.conv1d(n, k, padding=6)
    p_m = mix.pow(2).mean(dim=-1, keepdim=True) + 1e-9
    p_n = n.pow(2).mean(dim=-1, keepdim=True) + 1e-9
    n = n * torch.sqrt(p_m / (p_n * 10 ** (snr_db / 10.0)))
    return mix + n


@torch.no_grad()
def run(ckpt, device, split="test", limit=400):
    model, kind, _ = load_run_model(ckpt, device)
    loader = make_loader(config.cache_dir, split, config, deterministic=True,
                         limit=limit, shuffle=False, batch_size=8,
                         use_enroll=kind in ("enroll", "voicefilter"))
    out = {}
    for nk in ("white", "babble", "comb"):
        for snr in SNRS:
            vals = []
            for batch in loader:
                mix = batch["mix"].to(device)
                tgt = batch["target"].to(device)
                vid = batch["video"].to(device)
                enr = batch["enroll"].to(device) if "enroll" in batch else None
                noisy = add_noise(mix, nk, snr, mix)
                with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    est, _ = model(noisy, vid, enr) if enr is not None else model(noisy, vid)
                vals.extend(si_snri(est.float(), tgt, noisy).cpu().numpy().tolist())
            out[f"{nk}@{snr:+.0f}dB"] = {"si_snri": round(float(np.mean(vals)), 3),
                                         "std": round(float(np.std(vals)), 3),
                                         "n": len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    allout = {}
    for ck in a.ckpt:
        tag = ck.split("/")[-2] if "/" in ck else ck
        allout[tag] = run(ck, device, a.split, a.limit)
        print(f"[{tag}] " + "  ".join(f"{k}={v['si_snri']:+.2f}" for k, v in allout[tag].items()))
    (config.result_dir / "noise_robust.json").write_text(
        json.dumps(allout, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
