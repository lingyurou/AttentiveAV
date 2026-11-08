#!/usr/bin/env python3
"""已发表分离模型对比基线：MossFormer（ICASSP 2023，WSJ0-2mix 8k SOTA，魔搭官方权重）。

协议：
  - 测试混合与主表完全一致（同一份确定性 test 混合，16 kHz）。
  - MossFormer 原生 8 kHz：混合下采样到 8 kHz 分离，参考信号同步下采样，
    指标在 8 kHz 上计算（对所有被比模型一致的 8 kHz 协议）。
  - 分离出两路后做 oracle 选择（与本文 pit_rf 同一套"分离能力上界"口径），
    因为该模型没有任何目标线索——这 正 是 第 3 节 论 证 的 不 可 辨 识 性。
  - 零样本（不在 LRS2 上重训），报告为 published-model 行。

用法：
  systemd-run --user -p LimitAS=infinity ... python mossformer_eval.py --gpu 0 --limit 0
"""
import argparse
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

from config import config
from dataset import make_loader
from utils import si_snr, si_snri

MODEL_ID = "damo/speech_separation_mossformer_8k_pytorch"


def wav16(path, x8k):
    x = np.clip(x8k.numpy(), -1, 1)
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量 test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/mossformer_eval.json")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- 模型：直接加载魔搭官方权重（绕开 funasr 依赖）----
    from mossformer_model import build_model, separate

    model = build_model(device)

    loader = make_loader(config.cache_dir, a.split, config, deterministic=True,
                         batch_size=1, epoch=0)

    tmpdir = Path(tempfile.mkdtemp(prefix="mossf_"))
    rows, n = [], 0
    with torch.no_grad():
        for batch in loader:
            mix = batch["mix"][0].float().cpu()     # (1, T) 16 kHz
            tgt1 = batch["target"][0].float().cpu().view(-1)
            mix8 = AF.resample(mix, 16000, 8000).view(-1)
            tgt8 = AF.resample(tgt1, 16000, 8000)

            try:
                ests = separate(model, mix8, device)
            except Exception as e:              # 单条失败跳过并记录
                rows.append({"uid": n, "error": str(e)[:120]})
                n += 1
                continue
            k = min(len(tgt8), min(len(e) for e in ests))
            t = tgt8[:k]
            m8 = mix8[:k]
            si = [float(si_snri(e[:k].view(1, -1), t.view(1, -1), m8.view(1, -1)))
                  for e in ests]
            mix_si = float(si_snr(m8.view(1, -1), t.view(1, -1)))
            rows.append({"uid": n, "si": si, "mixture": mix_si,
                         "oracle": max(si)})
            n += 1
            if n % 50 == 0:
                print(f"{n} done", flush=True)
            if a.limit and n >= a.limit:
                break

    ok = [r for r in rows if "oracle" in r]
    res = {
        "model": "MossFormer (damo/speech_separation_mossformer_8k_pytorch, zero-shot)",
        "protocol": "8 kHz, oracle stream selection, same test mixtures as main table",
        "n": len(ok),
        "n_failed": len(rows) - len(ok),
        "si_snri_oracle": float(np.mean([r["oracle"] - r["mixture"] for r in ok]))
        if ok else None,
        "si_snr_oracle": float(np.mean([r["oracle"] for r in ok])) if ok else None,
        "per_utterance": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_utterance"},
                     ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
