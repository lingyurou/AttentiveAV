"""
模型冒烟测试（不需要数据集）
============================
    python smoke_test.py --gpu 1

检查：参数量、前向/反向是否通、输出长度是否对齐、显存占用、SI-SNR 数值范围。
"""
import argparse
import torch

from config import config
from models import build_model, count_parameters
from utils import si_snr, si_snri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()

    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  {torch.cuda.get_device_name(dev) if dev.type=='cuda' else ''}")

    B, T = a.batch, config.clip_samples
    mix = torch.randn(B, 1, T, device=dev) * 0.1
    tgt = torch.randn(B, 1, T, device=dev) * 0.1
    vid = torch.randn(B, 1, config.clip_frames, config.mouth_size, config.mouth_size, device=dev)

    for kind in ("audio", "av"):
        model = build_model(kind, config).to(dev)
        total, _ = count_parameters(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            est, aux = model(mix, vid)
        loss = -si_snr(est.float(), tgt).mean()
        loss.backward()

        grads = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"[{kind:5s}] params={total/1e6:.2f}M  out={tuple(est.shape)}  "
              f"len_ok={est.shape[-1]==T}  loss={loss.item():+.2f}  "
              f"grad_tensors={grads}  SI-SNRi={si_snri(est.float(), tgt, mix).mean().item():+.2f}")
        for k, v in aux.items():
            print(f"        aux[{k}] = {tuple(v.shape)}")
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            print(f"        显存峰值 {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats(dev)
    print("冒烟测试通过")


if __name__ == "__main__":
    main()
