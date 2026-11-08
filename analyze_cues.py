"""
线索增益的定标分析（不需要 GPU：直接读 per_sample CSV + 缓存的 F0）
==================================================================
回答：视觉线索的增益随「声学歧义程度」如何变化？
    - 目标/干扰基频间隔越小，音色越接近，音频端越难分辨 → 视觉应更值钱
    - 混合 SNR 越低，干扰越强

输出 results/cue_gain.json：
    by_f0   : {"<10Hz": {"av":..., "en":..., "au":..., "n":...}, ...}
    by_snr  : 同上，按混合 SNR 分桶
    net     : av - en / av - au（同一样本子集上的配对差）
"""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
RES = BASE / "results"
META = BASE / "data" / "cache" / "lrs2" / "test" / "meta.json"

TAGS = {"av": "av_rf", "en": "en_rf", "au": "au_rf", "pit": "pit_rf"}
F0_EDGES = [0, 10, 20, 40, 80, 1e9]
F0_LABELS = ["<10", "10-20", "20-40", "40-80", ">80"]
SNR_LABELS = ["[-6,-2)", "[-2,1)", "[1,6)"]


def load_per(tag):
    p = RES / f"per_sample_{tag}.csv"
    if not p.exists():
        return None
    rows = {}
    for line in p.read_text().splitlines()[1:]:
        c = line.split(",")
        if len(c) < 7:
            continue
        rows[c[0]] = {"partner": c[1], "is_hard": int(c[2]), "snr": float(c[3]),
                      "si_snr": float(c[4]), "si_snri": float(c[5])}
    return rows


def main():
    meta = json.loads(META.read_text())
    f0 = {m["uid"]: float(m["f0"]) for m in meta}

    data = {}
    for k, tag in TAGS.items():
        r = load_per(tag)
        if r:
            data[k] = r
    if "av" not in data:
        print("缺少 av_rf 的 per_sample，先跑 evaluate.py")
        return

    common = set(data["av"])
    for k, r in data.items():
        common &= set(r)
    common = sorted(common)
    print(f"公共子集 {len(common)} 句")

    def gap(uid, partner):
        a, b = f0.get(uid), f0.get(partner)
        return abs(a - b) if a and b else None

    out = {"n_common": len(common), "by_f0": {}, "by_snr": {}, "overall": {}, "net": {}}

    for lab_i, lab in enumerate(F0_LABELS):
        lo, hi = F0_EDGES[lab_i], F0_EDGES[lab_i + 1]
        sel = [u for u in common
               if (g := gap(u, data["av"][u]["partner"])) is not None and lo <= g < hi]
        if not sel:
            continue
        row = {"n": len(sel)}
        for k in data:
            row[k] = round(float(np.mean([data[k][u]["si_snri"] for u in sel])), 3)
        if "av" in row and "en" in row:
            row["net_av_en"] = round(row["av"] - row["en"], 3)
        if "av" in row and "au" in row:
            row["net_av_au"] = round(row["av"] - row["au"], 3)
        out["by_f0"][f"{lab} Hz"] = row

    for lab in SNR_LABELS:
        lo, hi = [float(x) for x in lab.strip("[]()").replace(")", "").split(",")]
        sel = [u for u in common if lo <= data["av"][u]["snr"] < hi]
        if not sel:
            continue
        row = {"n": len(sel)}
        for k in data:
            row[k] = round(float(np.mean([data[k][u]["si_snri"] for u in sel])), 3)
        out["by_snr"][lab] = row

    for k in data:
        out["overall"][k] = round(float(np.mean([data[k][u]["si_snri"] for u in common])), 3)
    for a, b in (("av", "en"), ("av", "au"), ("av", "pit"), ("en", "au")):
        if a in data and b in data:
            d = np.array([data[a][u]["si_snri"] - data[b][u]["si_snri"] for u in common])
            rng = np.random.default_rng(11)
            bs = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(4000)]
            try:
                from scipy import stats as ss
                p = float(ss.wilcoxon(d).pvalue)
            except Exception:
                p = None
            out["net"][f"{a}-{b}"] = {
                "diff": round(float(d.mean()), 3),
                "lo": round(float(np.percentile(bs, 2.5)), 3),
                "hi": round(float(np.percentile(bs, 97.5)), 3),
                "p": p, "win": round(float((d > 0).mean()), 3), "n": int(d.size),
            }

    (RES / "cue_gain.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n按 F0 间隔：")
    for k, v in out["by_f0"].items():
        print(f"  {k:10s} n={v['n']:4d}  " +
              "  ".join(f"{m}={v[m]:+.2f}" for m in ("av", "en", "au", "pit") if m in v))
    print("\n按混合 SNR：")
    for k, v in out["by_snr"].items():
        print(f"  {k:10s} n={v['n']:4d}  " +
              "  ".join(f"{m}={v[m]:+.2f}" for m in ("av", "en", "au", "pit") if m in v))
    print("\n配对差：")
    for k, v in out["net"].items():
        print(f"  {k}: {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']}")


if __name__ == "__main__":
    main()
