"""
评估指标与绘图工具
==================
SI-SNR / SI-SNRi 采用 utterance-level 定义（Le Roux et al., 2019 的尺度不变形式）。
"""
import json
from pathlib import Path

import numpy as np
import torch


# ---------------- 指标 ----------------
def si_snr(estimate, target, eps=1e-8):
    """SI-SNR (dB)，estimate/target: [B, T] 或 [B, 1, T]"""
    est = estimate.reshape(estimate.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    # 最优缩放投影 s_target = <est, tgt> / ||tgt||^2 * tgt
    alpha = (est * tgt).sum(dim=-1, keepdim=True) / (tgt.pow(2).sum(dim=-1, keepdim=True) + eps)
    s_target = alpha * tgt
    e_noise = est - s_target

    ratio = s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def si_snri(estimate, target, mixture, eps=1e-8):
    """SI-SNR 改善量：分离结果相对混合信号的提升"""
    return si_snr(estimate, target, eps) - si_snr(mixture, target, eps)


def stft_mag(x, n_fft=512, hop=128, eps=1e-8):
    """STFT 幅度谱，x: [B, 1, T]"""
    if x.dim() == 3:
        x = x.squeeze(1)
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft, hop, window=win, return_complex=True, center=True)
    return spec.abs() + eps


def spectral_l1(est, tgt, n_fft=512, hop=128):
    """幅度谱 L1：非尺度不变，能直接惩罚"没减干净的干扰"，用于打破 SI-SNR 平台期"""
    return torch.nn.functional.l1_loss(stft_mag(est, n_fft, hop), stft_mag(tgt, n_fft, hop))


# ---------------- 训练/评估记录 ----------------
class CsvLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("epoch,train_loss,val_si_snri,val_si_snr,lr,time_s\n")

    def log(self, epoch, train_loss, val_si_snri, val_si_snr, lr, elapsed):
        with open(self.path, "a") as f:
            f.write(f"{epoch},{train_loss:.5f},{val_si_snri:.4f},{val_si_snr:.4f},{lr:.3e},{elapsed:.1f}\n")

    def read(self):
        rows = self.path.read_text().strip().splitlines()
        if len(rows) < 2:
            return {}
        head = rows[0].split(",")
        cols = {h: [] for h in head}
        for r in rows[1:]:
            for h, v in zip(head, r.split(",")):
                cols[h].append(float(v))
        return cols


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


# ---------------- 从 run 目录恢复架构 ----------------
def apply_run_config(run_dir):
    """把 runs/<tag>/model_info.json 里的架构超参写回全局 config。

    训练脚本支持 --stride / --dilations / --repeats 覆盖架构，若评估时只按
    config.py 的默认值建模型，state_dict 会直接对不上。所有评估入口都必须先调它。
    """
    from config import config
    info_path = Path(run_dir) / "model_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    for k, v in info.get("config", {}).items():
        if isinstance(v, list):
            v = tuple(v)
        if hasattr(config, k):
            setattr(config, k, v)
    return info


def load_run_model(ckpt_path, device):
    """按 checkpoint 所在 run 目录的 model_info.json 重建模型并载入权重"""
    from config import config
    from models import build_model
    ckpt_path = Path(ckpt_path)
    info = apply_run_config(ckpt_path.parent)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    kind = ckpt.get("kind") or (info or {}).get("kind", "av")
    model = build_model(kind, config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, kind, ckpt


# ---------------- 绘图 ----------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager  # noqa: F401
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_training_curves(csv_path, out_path, title="Training curve"):
    plt = _mpl()
    rows = CsvLogger(csv_path).read()
    if not rows:
        return None
    ep = rows["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(ep, rows["train_loss"], color="#c0392b", lw=1.8)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (-SI-SNR)")
    axes[0].set_title("Training loss"); axes[0].grid(alpha=.3)

    axes[1].plot(ep, rows["val_si_snri"], color="#1f5fa8", lw=1.8, label="SI-SNRi")
    axes[1].plot(ep, rows["val_si_snr"], color="#8a919c", lw=1.4, ls="--", label="SI-SNR")
    axes[1].axhline(0, color="#b7791f", lw=1, ls=":")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("dB")
    axes[1].set_title("Validation separation"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_separation_demo(mix, target, est, attn=None, out_path="demo.png", sr=16000,
                         title="", zoom=1.0, smooth=60):
    """波形对比 + 视觉注意力权重（三个面板共用同一条时间轴）

    peak 归一只用于显示：SI-SNR 本身是尺度不变的，估计的绝对幅度没有意义。
    """
    plt = _mpl()
    mix = np.asarray(mix).squeeze()
    target = np.asarray(target).squeeze()
    est = np.asarray(est).squeeze()
    n_show = min(mix.size, int(round(zoom * sr)))
    t = np.arange(n_show) / sr
    dur = mix.size / sr

    def nz(x):
        x = np.asarray(x, dtype=np.float64)
        return x / (np.abs(x).max() + 1e-9)

    n_rows = 3 if attn is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.15 * n_rows), sharex=True)

    axes[0].plot(t, nz(mix[:n_show]), color="#b8bec7", lw=.6, label="Mixture")
    axes[0].plot(t, nz(target[:n_show]), color="#1e7a46", lw=1.0, alpha=.9, label="Target")
    axes[0].plot(t, nz(est[:n_show]), color="#c0392b", lw=.9, alpha=.85, label="Estimate")
    axes[0].set_ylabel("amp (peak-norm.)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    axes[0].set_title(title or "Waveform comparison", fontsize=10)
    axes[0].margins(x=0)

    axes[1].plot(t, nz((est - target)[:n_show]), color="#b7791f", lw=.7)
    axes[1].set_ylabel("error"); axes[1].margins(x=0)

    if attn is not None:
        a = np.asarray(attn).squeeze().astype(np.float64)
        if smooth and smooth > 1 and a.size > smooth:
            a = np.convolve(a, np.ones(smooth) / smooth, mode="same")
        a_t = np.linspace(0, dur, a.size)
        axes[2].plot(a_t, a, color="#1f5fa8", lw=1.1)
        axes[2].fill_between(a_t, 0, a, color="#1f5fa8", alpha=.18)
        axes[2].set_ylim(0, 1.02); axes[2].set_ylabel("visual attention")
        axes[2].margins(x=0)

    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(0, n_show / sr)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path
