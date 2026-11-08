"""
视觉增益的定标实验（论文的 claim 实验，不是刷点数的实验）
=========================================================
回答一个问题：视觉线索带来的 SI-SNRi 增益，随「目标与干扰的基频间隔」如何变化？

做法：固定模型权重，在验证/测试集上逐句算出 SI-SNRi，再按 (F0 间隔, 混合 SNR) 分箱统计。
      F0 间隔越小时，音频单模态在原理上越难分辨说话人，视觉应当是唯一可用的线索。

用法：
    python scripts/analyze_visual_gain.py --tag av_rf --split test --gpu 0
    python scripts/analyze_visual_gain.py --tag av_rf --split test --gpu 0 --compare-tag audio_rf

输出：
    runs/<tag>/gain_by_f0.csv      分箱统计明细
    runs/<tag>/gain_by_f0.png      增益曲线图
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from dataset import make_loader
from models import build_model, forward_batch
from utils import si_snri

F0_EDGES = [0, 10, 20, 40, 80, 1e9]
F0_LABELS = ["<10Hz", "10-20Hz", "20-40Hz", "40-80Hz", ">80Hz"]
SNR_EDGES = [-5, -2.5, 0, 2.5, 5]
SNR_LABELS = ["-5~-2.5", "-2.5~0", "0~2.5", "2.5~5"]


def load_run(tag: str, device):
    """按 runs/<tag>/model_info.json 里的 config 重建模型，避免架构漂移"""
    run_dir = config.run_dir / tag
    info = json.loads((run_dir / "model_info.json").read_text())
    for k, v in info["config"].items():
        if isinstance(v, list):
            v = tuple(v)
        setattr(config, k, v)
    model = build_model(info["kind"], config).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, info["kind"], run_dir, ckpt.get("best", float("nan"))


@torch.no_grad()
def run_eval(model, kind, split, device, batch_size=8):
    loader = make_loader(config.cache_dir, split, config, deterministic=True,
                         shuffle=False, batch_size=batch_size)
    ds = loader.dataset
    rows = []
    for b in loader:
        mix = b["mix"].to(device)
        vid = b["video"].to(device)
        tgt = b["target"].to(device)
        enr = b.get("enroll")
        enr = enr.to(device) if enr is not None else None
        est, _ = forward_batch(model, kind, mix, vid, enr)
        per = si_snri(est.float(), tgt, mix).cpu().numpy()
        for k, i in enumerate(b["index"]):
            j = b["partner_index"][k]
            rows.append({
                "uid": b["uid"][k],
                "si_snri": float(per[k]),
                "snr": float(b["snr"][k]),
                "f0_gap": abs(float(ds.f0[i]) - float(ds.f0[j])),
                "is_hard": bool(b["is_hard"][k]),
            })
    return rows


def summarise(rows, key):
    out = {k: [] for k in F0_LABELS}
    for r in rows:
        idx = int(np.clip(np.digitize(r["f0_gap"], F0_EDGES) - 1, 0, len(F0_LABELS) - 1))
        out[F0_LABELS[idx]].append(r["si_snri"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="要分析的 run，如 av_rf")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--compare-tag", default="", help="可选：对照 run（如 audio-only），用于算净增益")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    model, kind, run_dir, best = load_run(a.tag, device)
    print(f"[{a.tag}] kind={kind} best_val_SI-SNRi={best:+.2f} dB device={device}")

    rows = run_eval(model, kind, a.split, device, a.batch)
    n = len(rows)
    overall = float(np.mean([r["si_snri"] for r in rows]))
    print(f"{a.split} 集 {n} 句，SI-SNRi = {overall:+.2f} ± "
          f"{float(np.std([r['si_snri'] for r in rows])):.2f} dB")

    ref = None
    if a.compare_tag:
        model2, kind2, _, _ = load_run(a.compare_tag, device)
        ref = {r["uid"]: r["si_snri"] for r in run_eval(model2, kind2, a.split, device, a.batch)}
        print(f"对照 [{a.compare_tag}] 已载入，按 uid 对齐")

    by_f0 = summarise(rows, "f0_gap")
    lines = ["f0_bin,n,si_snri_mean,si_snri_std,net_gain_vs_control"]
    print(f"\n{'F0 间隔':<10}{'句数':>6}{'SI-SNRi':>12}{'净增益':>10}")
    for lab in F0_LABELS:
        v = by_f0[lab]
        if not v:
            continue
        gain = ""
        if ref is not None:
            sub = [r for r in rows if
                   F0_LABELS[int(np.clip(np.digitize(r["f0_gap"], F0_EDGES) - 1, 0, len(F0_LABELS) - 1))] == lab]
            d = [r["si_snri"] - ref[r["uid"]] for r in sub if r["uid"] in ref]
            gain = f"{np.mean(d):+.2f}" if d else ""
        print(f"{lab:<10}{len(v):>6}{np.mean(v):>+11.2f}{gain:>10}")
        lines.append(f"{lab},{len(v)},{np.mean(v):.4f},{np.std(v):.4f},{gain}")

    (run_dir / "gain_by_f0.csv").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6))
        xs, ys, ns = [], [], []
        for lab in F0_LABELS:
            if by_f0[lab]:
                xs.append(lab); ys.append(np.mean(by_f0[lab])); ns.append(len(by_f0[lab]))
        ax.plot(range(len(xs)), ys, marker="o", color="#1f5fa8")
        for i, (y, c) in enumerate(zip(ys, ns)):
            ax.annotate(f"{y:+.1f} dB\nn={c}", (i, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="#444")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
        ax.set_xlabel("Target-interferer F0 gap"); ax.set_ylabel("SI-SNRi (dB)")
        ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(run_dir / "gain_by_f0.png", dpi=160)
        print(f"\n图已保存: {run_dir/'gain_by_f0.png'}")
    except Exception as e:
        print(f"（绘图跳过: {e}）")


if __name__ == "__main__":
    main()
