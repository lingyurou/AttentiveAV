"""测数据管线吞吐：只跑 DataLoader，不碰 GPU"""
import sys
import time
from config import config
from dataset import make_loader

W = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = config
cfg.num_workers = W
loader = make_loader(cfg.cache_dir, "train", cfg, limit=0, shuffle=True, use_enroll=False)

t0 = time.time()
n = 0
nb = 0
for b in loader:
    n += b["mix"].shape[0]
    nb += 1
    if nb >= 300:
        break
dt = time.time() - t0
print(f"workers={W}: {n} 样本 / {dt:.1f}s = {n/dt:.0f} 样本/秒 | 每 batch(8) {dt/nb*1000:.1f} ms")
print(f"→ 推得 45839 条/epoch 需 {45839/(n/dt):.0f} 秒")
