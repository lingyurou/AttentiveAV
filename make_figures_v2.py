"""
修改稿用图（英文标注，300 dpi）—— 直接读 paper/numbers_v2.json
================================================================
    python make_figures_v2.py

输出 results/figs/：
    fig_arch.png        架构与张量维度
    fig_bars.png        各线索条件 SI-SNRi（95% CI）
    fig_curve.png       训练曲线
    fig_sync.png        视频帧偏移 vs SI-SNRi（对齐证据）
    fig_occl.png        三条视觉对照
    fig_f0.png          F0 间隔 vs 视觉净增益
    fig_stream.png      分块延迟—精度
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).parent
FIG = BASE / "results" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
NUM = BASE / "paper" / "numbers_v2.json"

C_AV, C_EN, C_AU, C_PIT, C_AB = "#c0392b", "#1f5fa8", "#6b7280", "#7b4fa8", "#b7791f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans"})


def load():
    return json.loads(NUM.read_text()) if NUM.exists() else {}


def fig_arch():
    """架构图：两条支路的连接点必须画准（审稿意见 R2-1 就是冲着描述不清来的）"""
    fig, ax = plt.subplots(figsize=(7.6, 3.35))
    ax.set_xlim(0, 9.65); ax.set_ylim(0, 4.0); ax.axis("off")

    def b(x, y, w, h, t, fc="#f4f6f9", ec="#3b4450", fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    lw=1.0, ec=ec, fc=fc))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    def a(x1, y1, x2, y2, c="#3b4450", ls="-", lw=1.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, lw=lw, color=c, linestyle=ls,
                                     shrinkA=0, shrinkB=0))

    # 音频通路
    b(0.10, 2.75, 1.10, 0.70, "Mixture\n$1\\times32000$")
    b(1.45, 2.75, 1.00, 0.70, "Encoder\n$192\\times2000$")
    b(2.70, 2.75, 1.40, 0.70, "TCN\n32 blocks")
    b(4.35, 2.75, 1.15, 0.70, "VGM\n$\\times(1+a_t)$", fc="#fdece9", ec=C_AV)
    b(5.75, 2.75, 1.00, 0.70, "Mask\n$\\sigma$")
    b(7.00, 2.75, 0.90, 0.70, "Decoder")
    b(8.15, 2.75, 1.15, 0.70, "Estimate\n$1\\times32000$")
    for x1, x2 in [(1.20, 1.45), (2.45, 2.70), (4.10, 4.35), (5.50, 5.75),
                   (6.75, 7.00), (7.90, 8.15)]:
        a(x1, 3.10, x2, 3.10)

    # 视觉通路
    b(1.45, 0.70, 1.15, 0.70, "Lip crop\n$50\\times96\\times96$", fc="#eaf1fa", ec=C_EN)
    b(2.85, 0.70, 1.20, 0.70, "Visual enc.\n(3-D conv)", fc="#eaf1fa", ec=C_EN)
    b(4.35, 0.70, 1.05, 0.70, "Mel head", fc="#eaf1fa", ec=C_EN)
    a(2.60, 1.05, 2.85, 1.05, c=C_EN)
    a(4.05, 1.05, 4.35, 1.05, c=C_EN, ls="--")
    a(5.40, 1.05, 5.62, 1.05, c=C_EN, ls="--")
    ax.text(5.66, 1.05, "$\\mathcal{L}_{mel}$  (training only)", fontsize=6.4,
            color=C_EN, va="center")

    # 视觉 -> 音频的两条连接（这是全文的关键，必须准确）
    a(3.30, 1.40, 4.62, 2.75, c=C_EN, ls="--")
    a(3.88, 1.40, 6.05, 2.75, c=C_EN, ls="-")
    ax.text(3.02, 1.66, "gate", fontsize=6.4, color=C_EN, rotation=52)
    ax.text(5.28, 1.98, "concat", fontsize=6.4, color=C_EN, rotation=32)

    ax.text(0.10, 3.62, "audio path", fontsize=7)
    ax.text(1.45, 0.44, "visual path", fontsize=7, color=C_EN)
    fig.tight_layout(); fig.savefig(FIG / "fig_arch.png", bbox_inches="tight"); plt.close(fig)


def fig_bars(d):
    order = ["au_rf", "pit_rf", "en_rf", "av_rf_noattn", "av_rf", "av_s8", "av_s32"]
    order = [t for t in order if t in d["runs"]]
    if not order:
        return
    lab = d["labels"]
    vals, errs, names, cols = [], [], [], []
    for t in order:
        r = d["runs"][t]
        n = max(r.get("n") or 1, 1)
        vals.append(r["si_snri"])
        errs.append(1.96 * (r["si_snri_std"] or 0) / n ** 0.5)
        names.append(lab.get(t, t).replace(" (proposed)", "\n(proposed)"))
        cols.append(C_PIT if t.startswith("pit") else C_EN if t.startswith("en")
                    else C_AU if t.startswith("au") else C_AV)

    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, xerr=errs, color=cols, height=0.58,
            error_kw=dict(lw=1, capsize=3, ecolor="#3b4450"))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.6)
    ax.axvline(0, color="#3b4450", lw=0.8)
    ax.set_xlabel("SI-SNRi (dB)")
    for yy, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.08, yy, f"{v:+.2f}", va="center", fontsize=7.2)
    ax.margins(x=0.30)
    fig.tight_layout(); fig.savefig(FIG / "fig_bars.png", bbox_inches="tight"); plt.close(fig)


def fig_curve(d):
    cur = d.get("curves") or {}
    if not cur:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style = {"av_rf": ("AttentiveAV", C_AV), "en_rf": ("Audio + enrollment", C_EN),
             "au_rf": ("Audio only", C_AU), "pit_rf": ("Audio only + PIT", C_PIT),
             "av_rf_noattn": ("w/o VGM", C_AB)}
    for t, (lb, c) in style.items():
        if t in cur:
            ax.plot(cur[t]["epoch"], cur[t]["val"], color=c, lw=1.6, label=lb)
    ax.axhline(0, color="#9aa3ad", lw=0.8, ls=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation SI-SNRi (dB)")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7.6)
    fig.tight_layout(); fig.savefig(FIG / "fig_curve.png", bbox_inches="tight"); plt.close(fig)


def fig_sync(d):
    sync = (d.get("sync") or {}).get("av_rf")
    if not sync:
        return
    ds = sorted(int(k) for k in sync)
    vals = [sync[str(k)]["si_snri"] for k in ds]
    errs = [1.96 * sync[str(k)]["std"] / max(sync[str(k)]["n"], 1) ** 0.5 for k in ds]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.errorbar(ds, vals, yerr=errs, marker="o", ms=4, lw=1.6, color=C_AV,
                ecolor="#c9a19b", capsize=2.5)
    ax.axvline(0, color="#9aa3ad", lw=0.8, ls=":")
    ax.axhline(0, color="#9aa3ad", lw=0.8, ls=":")
    ax.set_xlabel("Video frame offset (frames, 40 ms each)")
    ax.set_ylabel("SI-SNRi (dB)")
    ax.set_title("Temporal alignment sensitivity", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "fig_sync.png", bbox_inches="tight"); plt.close(fig)


def fig_occl(d):
    ctl = (d.get("controls") or {}).get("av_rf")
    if not ctl:
        return
    keys = [("correct", "Correct video"), ("zero", "Video zeroed"),
            ("shuffled", "Other speaker's video")]
    vals = [ctl[k]["si_snri"] for k, _ in keys]
    names = [n for _, n in keys]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    cols = [C_AV, C_AU, C_AB]
    bars = ax.bar(range(3), vals, color=cols, width=0.55)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7.6)
    ax.axhline(0, color="#3b4450", lw=0.8)
    ax.set_ylabel("SI-SNRi (dB)")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.25 if v >= 0 else -0.55), f"{v:+.2f}", ha="center", fontsize=7.6)
    ax.margins(y=0.18)
    fig.tight_layout(); fig.savefig(FIG / "fig_occl.png", bbox_inches="tight"); plt.close(fig)


def fig_f0(d):
    cue = d.get("cue") or {}
    buckets = cue.get("by_f0") or {}
    if not buckets:
        return
    if not buckets:
        return
    keys = [k for k in buckets]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for name, col in (("av", C_AV), ("en", C_EN), ("au", C_AU), ("pit", C_PIT)):
        ys = [buckets[k].get(name) for k in keys]
        if any(v is not None for v in ys):
            ax.plot(range(len(keys)), [np.nan if v is None else v for v in ys],
                    marker="o", ms=4, lw=1.5, color=col, label=name.upper())
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, fontsize=7.4)
    ax.set_xlabel("Target–interferer F0 separation")
    ax.set_ylabel("SI-SNRi (dB)")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7.6)
    fig.tight_layout(); fig.savefig(FIG / "fig_f0.png", bbox_inches="tight"); plt.close(fig)


def fig_stream(d):
    st = (d.get("stream") or {}).get("av_rf")
    if not st:
        return
    blocks = st.get("blocks") or {}
    if not blocks:
        return
    pts = sorted(((v["latency_s"], v["si_snri"]) for v in blocks.values()))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, lw=1.5, color=C_AV)
    full = st.get("full_utterance", {}).get("si_snri")
    if full is not None:
        ax.axhline(full, color="#9aa3ad", ls="--", lw=1.0)
        ax.text(pts[0][0], full + 0.08, f"full utterance {full:+.2f}", fontsize=7, color="#6b7280")
    ax.set_xlabel("Algorithmic latency (s)")
    ax.set_ylabel("SI-SNRi (dB)")
    ax.set_title("Block-wise (streaming) inference", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "fig_stream.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    d = load()
    fig_arch()
    if d:
        fig_bars(d); fig_curve(d); fig_sync(d); fig_occl(d); fig_f0(d); fig_stream(d)
    print("图已输出到", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)
