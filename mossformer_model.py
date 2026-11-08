#!/usr/bin/env python3
"""MossFormer（ICASSP 2023，WSJ0-2mix 8k）直接推理，绕开 funasr 依赖。

模型实现来自 modelscope 包内置的
modelscope.models.audio.separation.mossformer.MossFormer，
权重来自魔搭官方发布 damo/speech_separation_mossformer_8k_pytorch（model.pb）。
"""
import sys
from pathlib import Path

import torch

MODEL_DIR = Path("pretrained/mossformer8k")


def build_model(device="cpu"):
    from modelscope.models.audio.separation.mossformer import (
        MossFormer, MossFormerMaskNet, MossFormerM)

    kw = dict(kernel_size=16, out_channels=512, in_channels=512, stride=8,
              bias=False, num_blocks=24, d_model=512, attn_dropout=0.1,
              group_size=256, query_key_dim=128, expansion_factor=4,
              causal=False, norm="ln", num_spks=2)
    model = MossFormer(str(MODEL_DIR), **kw)

    ck = torch.load(MODEL_DIR / "model.pb", map_location="cpu",
                    weights_only=False)["model"]
    sd = {}
    for k, v in ck.items():
        if k.startswith("enc."):
            k = "encoder." + k[len("enc."):]
        elif k.startswith("dec."):
            k = "decoder." + k[len("dec."):]
        else:
            if ".intra_mdl." in k:
                k = k.replace(".intra_mdl.", ".att_mdl.")
            if ".intra_norm." in k:
                k = k.replace(".intra_norm.", ".att_norm.")
            if ".att_mdl.norm." in k:
                k = k.replace(".att_mdl.norm.", ".att_mdl.norm.norm.")
        sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 只允许 buffer 类缺失；参数缺失/多余都要报出来
    bad_m = [m for m in missing if "pos_enc" not in m and "buf" not in m]
    if bad_m or unexpected:
        raise RuntimeError(f"load mismatch: missing={bad_m[:5]} "
                           f"unexpected={unexpected[:5]}")
    return model.to(device).eval()


@torch.no_grad()
def separate(model, mix8k, device="cpu"):
    """mix8k: Tensor [T] @ 8 kHz → list of two [T'] tensors"""
    est = model(mix8k.view(1, -1).to(device))      # [B, T, num_spks]
    return [est[0, :, i].float().cpu() for i in range(est.shape[-1])]


if __name__ == "__main__":
    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    m = build_model(dev)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params {n_params/1e6:.2f} M")
    # 用官方 examples 里的样例试听级验证
    ex = sorted((MODEL_DIR / "examples").glob("*.wav"))
    print("examples:", [e.name for e in ex][:4])
    import soundfile as sf
    x, fs = sf.read(ex[0], dtype="float32")
    print("fs", fs, "dur", len(x)/fs)
    outs = separate(m, torch.from_numpy(x), dev)
    print("est shapes:", [tuple(o.shape) for o in outs])
    print("OK")
