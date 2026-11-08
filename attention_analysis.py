"""
视觉注意力与口型运动的定量关系
==============================
    python attention_analysis.py --ckpt runs/av_s1/best.pt --gpu 1

对测试样本逐帧计算：
    attention a_t     模型给出的视觉门控权重（上采样到 25 fps）
    mouth motion      相邻帧口型区域的平均绝对差（说话时明显变大）
    target energy     目标语音在该帧（40 ms）上的对数能量

报告 Pearson / Spearman 相关，用来支撑"注意力跟随发音动作"这一论断。
"""
import argparse
import json

import numpy as np
import torch
from scipy import stats as sstats

from config import config
from dataset import make_loader
from models import build_model
from utils import load_run_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/av_s1/best.pt")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n", type=int, default=240, help="参与统计的样本数")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    model, _kind, ck = load_run_model(a.ckpt, device)

    loader = make_loader(config.cache_dir, "test", config, deterministic=True,
                         limit=a.n, shuffle=False, batch_size=8)
    rows_attn_motion, rows_attn_energy = [], []
    for batch in loader:
        mix = batch["mix"].to(device)
        vid = batch["video"].to(device)
        tgt = batch["target"].to(device)
        est, aux = model(mix, vid)
        if "attention_weights" not in aux:
            raise SystemExit("该权重没有注意力分支")
        att = aux["attention_weights"]                      # [B,1,T']
        att = torch.nn.functional.interpolate(att, size=config.clip_frames,
                                              mode="linear", align_corners=False)[:, 0]

        v = vid.float()
        if v.max() > 1.5:                                    # uint8
            v = v / 255.0
        motion = (v[:, 0, 1:] - v[:, 0, :-1]).abs().mean(dim=(2, 3))   # [B, F-1]

        # 目标语音的帧能量（25 fps => 每帧 640 点）
        e = tgt[:, 0] ** 2
        e = torch.nn.functional.avg_pool1d(e.unsqueeze(1), kernel_size=640, stride=640)
        e = torch.log(e.squeeze(1) + 1e-8)
        k = min(e.shape[1], att.shape[1], motion.shape[1])
        att_c = att[:, :k].float().cpu().numpy()
        e_c = e[:, :k].float().cpu().numpy()
        m_c = motion[:, :k].float().cpu().numpy()

        for i in range(att_c.shape[0]):
            rows_attn_motion.append(sstats.pearsonr(att_c[i], m_c[i])[0])
            rows_attn_energy.append(sstats.pearsonr(att_c[i], e_c[i])[0])

    def summ(x, name):
        x = np.array([v for v in x if np.isfinite(v)])
        return {"name": name, "mean_r": round(float(x.mean()), 3),
                "median_r": round(float(np.median(x)), 3),
                "frac_positive": round(float(np.mean(x > 0)), 3), "n": int(x.size)}

    out = {"attn_vs_mouth_motion": summ(rows_attn_motion, "attention~mouth motion"),
           "attn_vs_target_energy": summ(rows_attn_energy, "attention~target energy")}
    (config.result_dir / "attention_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    for k, v in out.items():
        print(f"{v['name']}: 平均 r={v['mean_r']:+.3f} 中位 r={v['median_r']:+.3f} "
              f"正相关样本占比 {100*v['frac_positive']:.0f}% (n={v['n']})")


if __name__ == "__main__":
    main()
