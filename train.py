"""
训练入口（真实 LRS2，SI-SNR + 频谱 L1 + 梅尔蒸馏）
==================================================
    python train.py --kind av     --epochs 30 --tag av_s1   --gpu 1 --seed 2024
    python train.py --kind enroll --epochs 30 --tag en_s1   --gpu 0 --seed 2024
    python train.py --kind audio  --epochs 30 --tag au_s1   --gpu 0 --seed 2024

kind: av | av_noattn | av_nodistill | enroll | audio

输出:
    runs/<tag>/log.csv      每个 epoch 的 loss / SI-SNRi
    runs/<tag>/best.pt      验证集 SI-SNRi 最优权重
    runs/<tag>/curve.png    训练曲线
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import config
from dataset import make_loader
from models import NEEDS_ENROLL, build_model, count_parameters, forward_batch
from utils import CsvLogger, plot_training_curves, si_snr, si_snri, spectral_l1


class MelTarget:
    """把干净目标音频转成与视频同帧率(25fps)的对数梅尔谱，作为视觉分支的蒸馏目标"""

    def __init__(self, cfg, device):
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=1024, hop_length=cfg.mel_hop,
            n_mels=cfg.n_mels, power=2.0).to(device)

    def __call__(self, wav):
        return self.mel(wav.squeeze(1)).clamp_min(1e-6).log()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, kind, amp=True, collect=None):
    model.eval()
    tot_i, tot_s, n = 0.0, 0.0, 0
    per_sample = []
    for batch in loader:
        mix = batch["mix"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        vid = batch["video"].to(device, non_blocking=True)
        enr = batch["enroll"].to(device, non_blocking=True) if "enroll" in batch else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            est, aux = forward_batch(model, kind, mix, vid, enr)
        est = est.float()
        if "alt" in aux:                       # PIT：验证时用 oracle 选择
            alt = aux["alt"].float()
            pick = si_snri(alt, tgt, mix) > si_snri(est, tgt, mix)
            est = torch.where(pick.view(-1, 1, 1), alt, est)
        val_i = si_snri(est, tgt, mix)
        val_s = si_snr(est, tgt)
        tot_i += val_i.sum().item()
        tot_s += val_s.sum().item()
        n += mix.size(0)
        if collect is not None:
            for k in range(mix.size(0)):
                per_sample.append({
                    "uid": batch["uid"][k],
                    "partner": batch["partner_uid"][k],
                    "is_hard": bool(batch["is_hard"][k]),
                    "snr": float(batch["snr"][k]),
                    "si_snri": float(val_i[k]),
                    "si_snr": float(val_s[k]),
                })
    return tot_i / max(n, 1), tot_s / max(n, 1), per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", "--model", dest="kind", default="av",
                    choices=["av", "audio", "enroll", "av_noattn", "av_nodistill", "pit"])
    ap.add_argument("--epochs", type=int, default=config.num_epochs)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=config.learning_rate)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--seed", type=int, default=config.seed)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--stride", type=int, default=config.enc_stride,
                    help="编码器下采样步长；16 对应 1000Hz 帧率")
    ap.add_argument("--kernel", type=int, default=config.enc_kernel,
                    help="编码器卷积核长；必须 >= stride，否则每隔 stride 个采样只覆盖 kernel 个，"
                         "分析滤波器失真（500Hz 帧率应配 kernel=32）")
    ap.add_argument("--dilations", default="",
                    help="逗号分隔的膨胀率，如 1,2,4,8,16,32,64,128")
    ap.add_argument("--repeats", type=int, default=config.tcn_repeats)
    ap.add_argument("--mask-act", default="", choices=["", "relu", "sigmoid"])
    ap.add_argument("--cache", default="",
                    help="训练缓存目录覆盖（如 data/cache/lrs3_train），默认用 config.cache_dir")
    ap.add_argument("--val-cache", default="",
                    help="验证缓存目录覆盖，默认与 --cache 相同")
    a = ap.parse_args()

    # 架构覆盖：不改 config.py，保证历史实验可复现
    config.enc_stride = a.stride
    config.enc_kernel = a.kernel
    if a.dilations:
        config.tcn_dilations = tuple(int(x) for x in a.dilations.split(",") if x.strip())
    config.tcn_repeats = a.repeats
    if a.mask_act:
        config.mask_act = a.mask_act

    tag = a.tag or f"{a.kind}_{int(time.time())}"
    run_dir = config.run_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(a.seed)
    config.seed = a.seed
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    amp = (not a.no_amp) and device.type == "cuda"
    config.batch_size = a.batch
    config.num_workers = a.workers

    print(f"模型={a.kind} 设备={device} tag={tag} seed={a.seed}")
    print(config)

    model = build_model(a.kind, config).to(device)
    total, trainable = count_parameters(model)
    print(f"参数量: {total/1e6:.2f}M (可训练 {trainable/1e6:.2f}M)")
    (run_dir / "model_info.json").write_text(json.dumps({
        "kind": a.kind, "params": total, "seed": a.seed, "config": {
            k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, tuple, list))
        }}, ensure_ascii=False, indent=2))

    if a.cache:
        config.cache_dir = Path(a.cache)
    val_cache = Path(a.val_cache) if a.val_cache else config.cache_dir

    use_enroll = a.kind in NEEDS_ENROLL
    train_loader = make_loader(config.cache_dir, "train", config, limit=a.train_limit,
                               shuffle=True, use_enroll=use_enroll)
    val_loader = make_loader(val_cache, "val", config, deterministic=True,
                             limit=a.val_limit, shuffle=False,
                             batch_size=max(a.batch, 8), use_enroll=use_enroll)
    print(f"训练样本 {len(train_loader.dataset)} | 验证样本 {len(val_loader.dataset)}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=config.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    warm = max(a.warmup, 0) * steps_per_epoch
    total_steps = a.epochs * steps_per_epoch
    floor = config.min_lr / a.lr

    def lr_lambda(s):
        # 必须返回 python float：np.cos 会给 np.float64，写进 param_groups 后
        # torch 2.6 的 weights_only 加载会直接报错
        if s < warm:
            return float((s + 1) / max(warm, 1))
        prog = (s - warm) / max(total_steps - warm, 1)
        return float(floor + (1 - floor) * 0.5 * (1 + float(np.cos(np.pi * min(prog, 1.0)))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best = 1, -1e9
    if a.resume and Path(a.resume).exists():
        ckpt = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best", -1e9)
        print(f"从 {a.resume} 恢复，起始 epoch={start_epoch}，best={best:.2f}")

    mel_target = MelTarget(config, device) if a.kind in ("av", "av_noattn") else None
    logger = CsvLogger(run_dir / "log.csv")

    for epoch in range(start_epoch, a.epochs + 1):
        t0 = time.time()
        train_loader.dataset.epoch = epoch
        model.train()
        run_loss, seen = 0.0, 0

        pbar = tqdm(train_loader, desc=f"E{epoch}/{a.epochs}", ncols=100, leave=False)
        for batch in pbar:
            mix = batch["mix"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            vid = batch["video"].to(device, non_blocking=True)
            enr = batch["enroll"].to(device, non_blocking=True) if "enroll" in batch else None

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                est, aux = forward_batch(model, a.kind, mix, vid, enr)
                est_f = est.float()
                if "alt" in aux:                       # PIT：两输出 + 两源
                    alt = aux["alt"].float()
                    itf = (mix - tgt).float()
                    def branch(e1, e2):
                        return (-si_snr(e1, tgt).mean() + -si_snr(e2, itf).mean()) \
                            + config.loss_lam * (spectral_l1(e1, tgt, config.n_fft, config.hop)
                                                 + spectral_l1(e2, itf, config.n_fft, config.hop))
                    loss = torch.minimum(branch(est_f, alt), branch(alt, est_f))
                else:
                    # SI-SNR + 频谱幅度 L1：前者对齐论文指标，后者打破尺度不变导致的平台期
                    loss = -si_snr(est_f, tgt).mean() \
                        + config.loss_lam * spectral_l1(est_f, tgt, config.n_fft, config.hop)
                if mel_target is not None and "mel_pred" in aux:
                    tgt_mel = mel_target(tgt)
                    k = min(tgt_mel.shape[-1], aux["mel_pred"].shape[-1])
                    loss = loss + config.aux_weight * torch.nn.functional.l1_loss(
                        aux["mel_pred"][..., :k].float(), tgt_mel[..., :k])

            if not torch.isfinite(loss):
                print("  ! 出现非有限损失，跳过该 batch")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item() * mix.size(0)
            seen += mix.size(0)
            pbar.set_postfix(loss=f"{loss.item():.2f}")

        train_loss = run_loss / max(seen, 1)

        if epoch % a.eval_every == 0 or epoch == a.epochs:
            val_i, val_s, _ = evaluate(model, val_loader, device, a.kind, amp)
        else:
            val_i, val_s = float("nan"), float("nan")

        dt = time.time() - t0
        logger.log(epoch, train_loss, val_i, val_s, opt.param_groups[0]["lr"], dt)
        print(f"E{epoch:03d} loss={train_loss:7.3f} | val SI-SNRi={val_i:+6.2f} dB "
              f"SI-SNR={val_s:6.2f} dB | lr={opt.param_groups[0]['lr']:.2e} | {dt:.0f}s")

        if val_i == val_i and val_i > best:
            best = val_i
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "best": best,
                        "kind": a.kind, "tag": tag, "seed": a.seed}, run_dir / "best.pt")
            print(f"  ✓ 新的最优 SI-SNRi={best:+.2f} dB")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "kind": a.kind, "tag": tag, "seed": a.seed}, run_dir / "last.pt")

    plot_training_curves(run_dir / "log.csv", run_dir / "curve.png",
                         title=f"AttentiveAV · {a.kind} · {tag}")
    print(f"\n训练完成，最优验证 SI-SNRi = {best:+.2f} dB\n权重: {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
