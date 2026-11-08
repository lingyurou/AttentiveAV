"""测纯 GPU 计算占一步训练的比例：forward+backward 的耗时 vs 实测每步耗时"""
import time
import torch
from config import config
from models import build_model, prep_video, forward_batch

dev = torch.device("cuda:0")
cfg = config
cfg.enc_stride, cfg.enc_kernel = 16, 16
cfg.dilations, cfg.repeats = [1, 2, 4, 8, 16, 32, 64, 128], 4
m = build_model("av", cfg).to(dev)
n = sum(p.numel() for p in m.parameters())
print(f"params {n/1e6:.2f} M")

opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
B = 8
mix = torch.randn(B, 1, 16000, device=dev)
tgt = torch.randn(B, 1, 16000, device=dev)
vid = torch.randint(0, 255, (B, 1, 50, 96, 96), dtype=torch.uint8, device=dev)

def step():
    opt.zero_grad(set_to_none=True)
    v = prep_video(vid)
    out = forward_batch(m, "av", mix, v, None)
    if isinstance(out, (tuple, list)):
        out = out[0]
    loss = (out - tgt).pow(2).mean()
    loss.backward()
    opt.step()

for _ in range(10):
    step()
torch.cuda.synchronize()

for amp in (True, False):
    ts = []
    ctx = torch.cuda.amp.autocast(enabled=amp) if amp else torch.cuda.amp.autocast(enabled=False)
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.time()
        with ctx:
            step()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    ts.sort()
    print(f"amp={amp}: 中位 {ts[len(ts)//2]*1000:.1f} ms/step  最快 {ts[0]*1000:.1f} ms")
