"""
把所有实验结果汇成一个 numbers_v2.json，供论文/回应/报告生成器读取。
任何一项缺失都不报错，只留 None —— 生成器会显示 "—"。

    python collect_numbers.py
"""
import glob
import json
from pathlib import Path

import numpy as np

from config import config

R = config.result_dir
OUT = config.base_dir / "paper" / "numbers_v2.json"

RUNS = ["av_rf", "en_rf", "au_rf", "pit_rf", "av_rf_noattn", "av_rf_nodistill", "av_s32k"]
LABEL = {
    "mixture": "Unprocessed mixture",
    "au_rf": "Audio-only (no cue)",
    "pit_rf": "Audio-only PIT + oracle selection",
    "en_rf": "Enrollment-conditioned (VoiceFilter-style)",
    "av_rf": "AttentiveAV (proposed)",
    "av_rf_noattn": "AttentiveAV w/o visual gain modulation",
    "av_rf_nodistill": "AttentiveAV w/o mel distillation",
    "av_s32k": "AttentiveAV, 500 Hz frame rate (stride 32, kernel 32)",
}


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    d = {"runs": {}, "labels": LABEL}

    for tag in RUNS + ["mixture"]:
        js = load(R / f"eval_{tag}.json")
        if not js:
            continue
        d["runs"][tag] = {
            "params": js.get("params"),
            "n": js.get("all", {}).get("n"),
            "si_snri": js.get("all", {}).get("si_snri"),
            "si_snri_std": js.get("all", {}).get("si_snri_std"),
            "si_snr": js.get("all", {}).get("si_snr"),
            "pesq": js.get("all", {}).get("pesq"),
            "stoi": js.get("all", {}).get("stoi"),
            "pos_ratio": js.get("all", {}).get("pos_ratio"),
            "easy": js.get("easy", {}).get("si_snri"),
            "hard": js.get("hard", {}).get("si_snri"),
            "easy_hard_gap": (js.get("easy", {}).get("si_snri", 0) or 0)
                             - (js.get("hard", {}).get("si_snri", 0) or 0),
            "enroll_subset": js.get("all_enroll", {}).get("si_snri"),
            "enroll_subset_n": js.get("all_enroll", {}).get("n"),
            "snr_buckets": {k: v.get("si_snri") for k, v in (js.get("snr_buckets") or {}).items()},
            "best_epoch": js.get("best_epoch"),
            "best_val": js.get("best_val"),
            "mix_si_snr": js.get("mix_si_snr"),
        }

    # 统计检验：与 av_rf 逐样本配对
    stats = load(R / "stats.json") or {}
    d["stats_raw"] = stats
    d["sig"] = {}
    per = {}
    for tag in RUNS:
        p = R / f"per_sample_{tag}.csv"
        if p.exists():
            rows = {}
            for line in p.read_text().splitlines()[1:]:
                c = line.split(",")
                if len(c) >= 5:
                    rows[c[0]] = float(c[4])
            per[tag] = rows
    if "av_rf" in per:
        base = per["av_rf"]
        for tag, rows in per.items():
            if tag == "av_rf":
                continue
            common = sorted(set(base) & set(rows))
            if len(common) < 50:
                continue
            a = np.array([base[k] for k in common])
            b = np.array([rows[k] for k in common])
            diff = a - b
            rng = np.random.default_rng(7)
            boots = [diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(4000)]
            try:
                from scipy import stats as ss
                p_val = float(ss.wilcoxon(a, b).pvalue)
            except Exception:
                p_val = None
            d["sig"][tag] = {
                "n": int(diff.size), "diff": round(float(diff.mean()), 3),
                "lo": round(float(np.percentile(boots, 2.5)), 3),
                "hi": round(float(np.percentile(boots, 97.5)), 3),
                "p": p_val, "win": round(float((diff > 0).mean()), 3),
            }

    d["controls"] = load(R / "controls.json")
    d["attention"] = load(R / "attention_analysis.json")
    d["eff"] = load(R / "efficiency.json")
    for f in glob.glob(str(R / "sync_shift_*.json")):
        d.setdefault("sync", {})[Path(f).stem.replace("sync_shift_", "")] = load(f)
    for f in glob.glob(str(R / "stream_*.json")):
        d.setdefault("stream", {})[Path(f).stem.replace("stream_", "")] = load(f)
    d["noise"] = load(R / "noise_robust.json")
    d["cue"] = load(R / "cue_gain.json")
    d["eff_layers"] = load(R / "eff_layers_av_rf.json")
    d["mossformer"] = load(R / "mossformer_eval.json")
    d["ours_8k"] = load(R / "ours_8k_eval.json")
    # LRS3 第二语料库：零样本 + 域内重训（同一 test 集）
    lrs3 = {"zero_shot": {}, "in_domain": {}}
    for t in ("av_rf", "pit_rf", "en_rf"):
        e = load(R / f"eval_{t}_zs.json")
        if e and e.get("all"):
            lrs3["zero_shot"][t] = e["all"]
    for t in ("av_rf_lrs3", "en_rf_lrs3"):
        e = load(R / f"eval_{t}.json")
        if e and e.get("all"):
            lrs3["in_domain"][t] = e["all"]
    m = load(R / "eval_mixture_lrs3_test.json")
    if m and m.get("all"):
        lrs3["mixture"] = m["all"]
    if lrs3["zero_shot"] or lrs3["in_domain"]:
        d["lrs3"] = lrs3
    # seed 方差（av_rf_s1/s2 训练完成后才有）
    seeds = {}
    for t in ("av_rf", "av_rf_s1", "av_rf_s2"):
        e = load(R / f"eval_{t}.json")
        if e and e.get("all"):
            seeds[t] = e["all"]["si_snri"]
    d["seed_si_snri"] = seeds

    # 训练曲线（用于画学习曲线 / 说明 RF 的影响）
    curves = {}
    for tag in RUNS:
        p = config.run_dir / tag / "log.csv"
        if p.exists():
            lines = p.read_text().strip().splitlines()
            if len(lines) > 1:
                head = lines[0].split(",")
                cols = {h: [] for h in head}
                for r in lines[1:]:
                    for h, v in zip(head, r.split(",")):
                        cols[h].append(float(v))
                curves[tag] = cols
    d["curves"] = {k: {"epoch": v["epoch"], "val": v["val_si_snri"]} for k, v in curves.items()}

    d["meta"] = {
        "clip_seconds": config.clip_seconds, "sample_rate": config.sample_rate,
        "fps": config.fps, "mouth": config.mouth_size,
        "data_hours": 25.5, "train": 45839, "val": 1082, "test": 1243,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=float))
    print(f"写入 {OUT}")
    for tag, r in d["runs"].items():
        print(f"  {tag:12s} {str(r['si_snri']):>8s} dB  n={r['n']}  params={r['params']}")
    for tag, s in d["sig"].items():
        print(f"  sig vs {tag}: {s['diff']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}] p={s['p']}")


if __name__ == "__main__":
    main()
