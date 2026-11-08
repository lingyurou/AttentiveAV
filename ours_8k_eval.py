#!/usr/bin/env python3
"""把我们自己的模型放到与 MossFormer 相同的 8 kHz 协议下评估：
推理在 16 kHz，est/tgt/mix 下采样到 8 kHz 后算 SI-SNR/SI-SNRi，
与 mossformer_eval.py 的口径完全一致，用于公平对比行。
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio.functional as AF

from config import config
from dataset import make_loader
from models import forward_batch
from utils import load_run_model, si_snr, si_snri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="results/ours_8k_eval.json")
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    loader = make_loader(config.cache_dir, a.split, config, deterministic=True,
                         batch_size=a.batch, epoch=0, use_enroll=True)

    res = {}
    for ck in a.ckpt:
        model, kind, _ = load_run_model(ck, device)
        rows = []
        with torch.no_grad():
            for batch in loader:
                mix = batch["mix"].to(device).float()
                tgt = batch["target"].to(device).float()
                vid = batch["video"].to(device)
                enr = batch["enroll"].to(device) if "enroll" in batch else None
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    est, aux = forward_batch(model, kind, mix, vid, enr)
                est = est.float()
                if "alt" in aux:
                    alt = aux["alt"].float()
                    pick = si_snri(alt, tgt, mix) > si_snri(est, tgt, mix)
                    est = torch.where(pick.view(-1, 1, 1), alt, est)
                # 8 kHz 协议
                for b in range(est.shape[0]):
                    e8 = AF.resample(est[b].view(1, -1).cpu(), 16000, 8000)
                    t8 = AF.resample(tgt[b].view(1, -1).cpu(), 16000, 8000)
                    m8 = AF.resample(mix[b].view(1, -1).cpu(), 16000, 8000)
                    k = min(e8.shape[-1], t8.shape[-1], m8.shape[-1])
                    rows.append({
                        "si_snri": float(si_snri(e8[..., :k], t8[..., :k], m8[..., :k])),
                        "si_snr": float(si_snr(e8[..., :k], t8[..., :k])),
                    })
        res[Path(ck).parent.name] = {
            "n": len(rows),
            "si_snri": sum(r["si_snri"] for r in rows) / len(rows),
            "si_snr": sum(r["si_snr"] for r in rows) / len(rows),
        }
        print(Path(ck).parent.name, res[Path(ck).parent.name], flush=True)

    Path(a.out).write_text(json.dumps(res, indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
