"""
视频线索对照实验（审稿人必问：模型真的在用嘴型吗？）
====================================================
    python controls.py --ckpt runs/av_s1/best.pt --gpu 1

三种输入条件下的同一权重：
    correct   真实视频（正常条件）
    zero      视频置零（等于纯音频，但 BN/分布不变）
    shuffled  换成同批次另一个说话人的视频（线索错误但分布正确）

如果 shuffled 与 zero 都显著劣于 correct，说明模型确实依赖视觉线索；
如果三者接近，说明视觉分支只是正则项，论文的核心主张就不成立。
"""
import argparse
import json

import numpy as np
import torch

from config import config
from dataset import make_loader
from models import build_model, forward_batch
from utils import load_run_model, si_snri


@torch.no_grad()
def run(ckpt_path, device, split="test", limit=0):
    model, kind, ckpt = load_run_model(ckpt_path, device)
    loader = make_loader(config.cache_dir, split, config, deterministic=True,
                         limit=limit, shuffle=False, batch_size=8)
    res = {"correct": [], "zero": [], "shuffled": []}
    for batch in loader:
        mix = batch["mix"].to(device)
        tgt = batch["target"].to(device)
        vid = batch["video"].to(device)
        bs = mix.size(0)
        roll = torch.roll(vid, shifts=1, dims=0)          # 换成同批另一条视频
        for name, v in (("correct", vid), ("zero", torch.zeros_like(vid)), ("shuffled", roll)):
            with torch.autocast("cuda", dtype=torch.float16):
                est, _ = forward_batch(model, kind, mix, v, None)
            res[name].extend(si_snri(est.float(), tgt, mix).cpu().numpy().tolist())
    out = {}
    for k, v in res.items():
        v = np.asarray(v)
        out[k] = {"n": int(v.size), "si_snri": round(float(v.mean()), 3),
                  "std": round(float(v.std()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    all_out = {}
    for ck in a.ckpt:
        tag = ck.split("/")[-2] if "/" in ck else ck
        out = run(ck, device, a.split, a.limit)
        all_out[tag] = out
        print(f"[{tag}] correct={out['correct']['si_snri']:+.2f} "
              f"zero={out['zero']['si_snri']:+.2f} "
              f"shuffled={out['shuffled']['si_snri']:+.2f} dB")
    (config.result_dir / "controls.json").write_text(
        json.dumps(all_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
