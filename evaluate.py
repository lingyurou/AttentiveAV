"""
测试集评估（多模型、多指标）
============================
    python evaluate.py --ckpt runs/av_s1/best.pt --ckpt runs/en_s1/best.pt --gpu 1

输出:
    results/eval_<tag>.json        总体 + easy/hard + SNR 分桶指标
    results/per_sample_<tag>.csv   逐样本指标（含 PESQ / STOI）
    results/comparison.md          多模型对比表

指标约定：
    SI-SNRi  相对混合信号的改善（dB），本篇主指标
    SI-SNR   绝对 SI-SNR（dB）
    PESQ     wideband，16 kHz（ITU-T P.862.2）
    STOI     短时客观可懂度
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from config import config
from dataset import make_loader
from models import NEEDS_ENROLL, build_model, forward_batch
from utils import load_run_model, si_snr, si_snri

BUCKETS = [(-6.0, -2.0), (-2.0, 1.0), (1.0, 6.0)]


def _pesq_stoi(est, ref, sr=16000):
    """逐条计算 PESQ / STOI，任何失败都返回 nan（不能因为个别样本崩掉整轮评估）"""
    from pesq import pesq
    from pystoi import stoi
    e = np.asarray(est, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(e), len(r))
    e, r = e[:n], r[:n]
    # PESQ 对电平敏感，先对齐到参考的 RMS（不改变 SI-SNR）
    e = e / (np.sqrt(np.mean(e ** 2)) + 1e-8) * (np.sqrt(np.mean(r ** 2)) + 1e-8)
    try:
        p = float(pesq(sr, r, e, "wb"))
    except Exception:
        p = float("nan")
    try:
        s = float(stoi(r, e, sr, extended=False))
    except Exception:
        s = float("nan")
    return p, s


@torch.no_grad()
def run_one(ckpt_path, device, split="test", limit=0, amp=True, want_pesq=True,
            cache_dir=None):
    model, kind, ckpt = load_run_model(ckpt_path, device)

    use_enroll = kind in NEEDS_ENROLL
    loader = make_loader(cache_dir or config.cache_dir, split, config, deterministic=True,
                         limit=limit, shuffle=False, batch_size=8, use_enroll=use_enroll)

    rows = []
    for batch in loader:
        mix = batch["mix"].to(device)
        tgt = batch["target"].to(device)
        vid = batch["video"].to(device)
        enr = batch["enroll"].to(device) if "enroll" in batch else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            est, aux = forward_batch(model, kind, mix, vid, enr)
        est = est.float()
        if "alt" in aux:                  # PIT：oracle 选择（分离能力上界）
            alt = aux["alt"].float()
            pick = si_snri(alt, tgt, mix) > si_snri(est, tgt, mix)
            est = torch.where(pick.view(-1, 1, 1), alt, est)
        si = si_snri(est, tgt, mix)
        ss = si_snr(est, tgt)
        mix_s = si_snr(mix, tgt)

        for k in range(mix.size(0)):
            rec = {
                "uid": batch["uid"][k], "partner": batch["partner_uid"][k],
                "is_hard": int(batch["is_hard"][k]), "snr": round(float(batch["snr"][k]), 2),
                "si_snr": round(float(ss[k]), 3), "si_snri": round(float(si[k]), 3),
                "mix_si_snr": round(float(mix_s[k]), 3),
                "has_enroll": int(batch["has_enroll"][k]) if "has_enroll" in batch else 1,
            }
            if want_pesq:
                p, s = _pesq_stoi(est[k, 0].cpu().numpy(), tgt[k, 0].cpu().numpy())
                rec["pesq"] = round(p, 3)
                rec["stoi"] = round(s, 4)
            rows.append(rec)

    def agg(subset, with_pesq=want_pesq):
        if not subset:
            return {"n": 0}
        d = {"n": len(subset)}
        for key in ("si_snri", "si_snr"):
            v = np.array([r[key] for r in subset], dtype=float)
            d[key] = round(float(np.nanmean(v)), 3)
            d[key + "_std"] = round(float(np.nanstd(v)), 3)
            d[key + "_median"] = round(float(np.nanmedian(v)), 3)
        if with_pesq:
            for key in ("pesq", "stoi"):
                v = np.array([r[key] for r in subset], dtype=float)
                d[key] = round(float(np.nanmean(v)), 4)
        d["pos_ratio"] = round(float(np.mean([r["si_snri"] > 0.1 for r in subset])), 4)
        return d

    easy = [r for r in rows if not r["is_hard"]]
    hard = [r for r in rows if r["is_hard"]]
    buckets = {}
    for lo, hi in BUCKETS:
        sub = [r for r in rows if lo <= r["snr"] < hi]
        if sub:
            buckets[f"[{lo:+.0f},{hi:+.0f})"] = agg(sub)
    # 注册语音基线只在"有同说话人参考"的样本上才公平，全体模型都算这个子集
    sub_enroll = [r for r in rows if r.get("has_enroll", 1) == 1]

    summary = {
        "checkpoint": str(ckpt_path), "kind": kind, "split": split,
        "best_epoch": ckpt.get("epoch"), "best_val": ckpt.get("best"),
        "params": sum(p.numel() for p in model.parameters()),
        "all": agg(rows), "easy": agg(easy), "hard": agg(hard),
        "all_enroll": agg(sub_enroll),
        "easy_enroll": agg([r for r in sub_enroll if not r["is_hard"]]),
        "hard_enroll": agg([r for r in sub_enroll if r["is_hard"]]),
        "snr_buckets": buckets,
        "mix_si_snr": round(float(np.nanmean([r["mix_si_snr"] for r in rows])), 3),
    }
    return summary, rows


@torch.no_grad()
def run_mixture(device, split="test", limit=0, want_pesq=True, cache_dir=None):
    """未处理混音的参考行（SI-SNRi 恒为 0，但 PESQ/STOI 有意义）"""
    loader = make_loader(cache_dir or config.cache_dir, split, config, deterministic=True,
                         limit=limit, shuffle=False, batch_size=8)
    rows = []
    for batch in loader:
        mix = batch["mix"]
        tgt = batch["target"]
        for k in range(mix.size(0)):
            rec = {"uid": batch["uid"][k], "partner": batch["partner_uid"][k],
                   "is_hard": int(batch["is_hard"][k]), "snr": round(float(batch["snr"][k]), 2),
                   "si_snr": 0.0, "si_snri": 0.0, "mix_si_snr": 0.0,
                   "has_enroll": int(batch["has_enroll"][k]) if "has_enroll" in batch else 1}
            if want_pesq:
                p, s = _pesq_stoi(mix[k, 0].numpy(), tgt[k, 0].numpy())
                rec["pesq"] = round(p, 3)
                rec["stoi"] = round(s, 4)
            rows.append(rec)

    def agg(subset):
        if not subset:
            return {"n": 0}
        d = {"n": len(subset)}
        for key in ("si_snri", "si_snr"):
            v = np.array([r[key] for r in subset], dtype=float)
            d[key] = round(float(np.mean(v)), 3)
            d[key + "_std"] = round(float(np.std(v)), 3)
            d[key + "_median"] = 0.0
        if want_pesq:
            for key in ("pesq", "stoi"):
                d[key] = round(float(np.nanmean([r[key] for r in subset])), 4)
        d["pos_ratio"] = 0.0
        return d

    easy = [r for r in rows if not r["is_hard"]]
    hard = [r for r in rows if r["is_hard"]]
    return {"tag": "mixture", "kind": "mixture", "split": split, "params": 0,
            "all": agg(rows), "easy": agg(easy), "hard": agg(hard),
            "all_enroll": agg([r for r in rows if r.get("has_enroll", 1) == 1]),
            "snr_buckets": {}, "mix_si_snr": 0.0}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                    help="可多次指定；也支持 path=label 形式")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pesq", action="store_true")
    ap.add_argument("--with-mixture", action="store_true", help="额外输出未处理混音参考行")
    ap.add_argument("--cache", default=None,
                    help="覆盖缓存目录（如 data/cache/lrs3 做跨库零样本探测）；"
                         "设置后汇总写入 comparison_<库名>.md，不覆盖主表 comparison.md")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    config.result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(a.cache).resolve() if a.cache else None
    comp_name = f"comparison_{cache_dir.name}.md" if cache_dir else "comparison.md"

    summaries = []
    if a.with_mixture:
        s, rows = run_mixture(device, a.split, a.limit, want_pesq=not a.no_pesq,
                              cache_dir=cache_dir)
        summaries.append(s)
        mix_name = f"eval_mixture_{cache_dir.name}.json" if cache_dir else "eval_mixture.json"
        (config.result_dir / mix_name).write_text(
            json.dumps(s, ensure_ascii=False, indent=2))
        with open(config.result_dir /
                  (f"per_sample_mixture_{cache_dir.name}.csv" if cache_dir
                   else "per_sample_mixture.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[mixture] PESQ={s['all'].get('pesq', float('nan')):.2f} "
              f"STOI={s['all'].get('stoi', float('nan')):.3f}")
    for item in a.ckpt:
        ck, _, label = item.partition("=")
        tag = label or (Path(ck).parent.name if "/" in ck else Path(ck).stem)
        s, rows = run_one(ck, device, split=a.split, limit=a.limit,
                          want_pesq=not a.no_pesq, cache_dir=cache_dir)
        s["tag"] = tag
        summaries.append(s)
        (config.result_dir / f"eval_{tag}.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2))
        with open(config.result_dir / f"per_sample_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[{tag}] SI-SNRi={s['all'].get('si_snri', float('nan')):+.2f} ± {s['all'].get('si_snri_std', 0):.2f} "
              f"| PESQ={s['all'].get('pesq', float('nan')):.2f} | STOI={s['all'].get('stoi', float('nan')):.3f} "
              f"| easy={s['easy'].get('si_snri', float('nan')):+.2f} hard={s['hard'].get('si_snri', float('nan')):+.2f} | n={s['all']['n']}")

    head = ("| 模型 | 参数量 | SI-SNRi (dB) | SI-SNR (dB) | PESQ | STOI | easy | hard | 正增益比例 |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(
            f"| {s['tag']} | {s['params']/1e6:.2f}M | {s['all'].get('si_snri', float('nan')):+.2f} ± {s['all'].get('si_snri_std', 0):.2f} "
            f"| {s['all'].get('si_snr', float('nan')):+.2f} | {s['all'].get('pesq', float('nan')):.2f} "
            f"| {s['all'].get('stoi', float('nan')):.3f} | {s['easy'].get('si_snri', float('nan')):+.2f} "
            f"| {s['hard'].get('si_snri', float('nan')):+.2f} | {100*s['all'].get('pos_ratio', 0):.0f}% |")
    (config.result_dir / comp_name).write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
