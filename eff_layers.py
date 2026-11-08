"""
逐阶段算力分解：给 Table 1 提供实测 MACs，替代手算估计
=======================================================
    python eff_layers.py --ckpt runs/av_rf/best.pt
输出:
    results/eff_layers_<tag>.json

做法：对每个顶层子模块单独挂前向钩子各跑一遍，避免嵌套归属含糊。
CPU 上跑一次前向即可（batch 1），耗时几秒。
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from config import config
from models import prep_video
from utils import load_run_model

STAGES = ["encoder", "sepnet", "q_proj", "k_proj", "mask_head", "aux_head", "decoder", "visual"]


def count_macs(module, *args):
    total = 0

    def conv_hook(mod, inp, out):
        nonlocal total
        k = 1
        for s in mod.kernel_size:
            k *= s
        total += out.numel() * (mod.in_channels // mod.groups) * k

    def convt_hook(mod, inp, out):
        nonlocal total
        k = 1
        for s in mod.kernel_size:
            k *= s
        total += out.numel() * (mod.out_channels // mod.groups) * k

    def lin_hook(mod, inp, out):
        nonlocal total
        total += out.numel() * mod.in_features

    hs = []
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            hs.append(m.register_forward_hook(convt_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    try:
        with torch.no_grad():
            module(*args)
    finally:
        for h in hs:
            h.remove()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/av_rf/best.pt")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, kind, _ = load_run_model(args.ckpt, device)
    model.eval()

    n_samp = int(config.sample_rate * config.clip_seconds)
    n_frm = int(config.fps * config.clip_seconds)
    mix = torch.zeros(1, 1, n_samp)
    video = torch.zeros(1, 1, n_frm,
                        config.mouth_size, config.mouth_size, dtype=torch.uint8)

    # 整体前向一次，拿总 MACs
    macs_total = count_macs(model, mix, video)

    stages = {}
    # 视觉与音频两条支路互不依赖，各自前向即可
    vf = prep_video(video.clone())
    if hasattr(model, "visual"):
        stages["visual"] = count_macs(model.visual, vf)
    if hasattr(model, "encoder"):
        enc = model.encoder(mix)
        stages["encoder"] = count_macs(model.encoder, mix)
        stages["sepnet"] = count_macs(model.sepnet, enc)
        for nm in ("q_proj", "k_proj"):
            m = getattr(model, nm, None)
            if m is not None:
                stages[nm] = count_macs(m, enc)
        if getattr(model, "mask_head", None) is not None:
            # 掩膜头吃的是 [分离特征 ; 视觉特征] 的拼接，通道数翻倍
            stages["mask_head"] = count_macs(model.mask_head, torch.cat([enc, enc], dim=1))
        if getattr(model, "aux_head", None) is not None:
            with torch.no_grad():
                v0 = model.visual(vf)
            stages["aux_head"] = count_macs(model.aux_head, v0)
        stages["decoder"] = count_macs(model.decoder, enc)

    params = {n: sum(p.numel() for p in getattr(model, n).parameters())
              for n in STAGES if getattr(model, n, None) is not None}
    total_params = sum(p.numel() for p in model.parameters())

    tag = Path(args.ckpt).parent.name
    out = {
        "run": tag,
        "kind": kind,
        "params": total_params,
        "macs_total_g": round(macs_total / 1e9, 3),
        "stages_macs_g": {k: round(v / 1e9, 3) for k, v in stages.items()},
        "stages_params": params,
        "frame_rate_hz": round(config.sample_rate / 16.0, 1),
        "clip_seconds": config.clip_seconds,
    }
    p = config.result_dir / f"eff_layers_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"写入 {p}")
    print(f"  总参数 {total_params/1e6:.2f} M，总 MACs {out['macs_total_g']:.2f} G")
    for k, v in out["stages_macs_g"].items():
        print(f"  {k:10s} {v:6.3f} G  ({v/out['macs_total_g']:.0%})")


if __name__ == "__main__":
    main()
