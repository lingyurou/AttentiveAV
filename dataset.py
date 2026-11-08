"""
LRS2 双说话人混合数据集（目标说话人提取协议）
==============================================
从 prepare_cache.py 生成的 memmap 缓存中构造混合语音：

    mixture = target + interferer * 10^(-SNR/20)

- target    : 视频中那个人说的话（视觉线索指向的目标）
- interferer: 同一 split 内另一个说话人
- 混合 SNR 在 [snr_min, snr_max] dB 内均匀采样
- 难样本对（hard pair）：按基频相近原则配对，此时仅靠音色很难区分，
  视觉信息才是关键 —— 这是检验"视觉注意力是否真的有用"的核心子集
- enrollment: 同一说话人的**另一条**语音（不含本条），供注册式基线使用；
  这样 audio+enrollment 与 audio-visual 拿到的信息量是可比的（都是"目标线索"）

所有采样都由 (split, index, epoch) 决定，因此评估完全可复现。

视频以 uint8 返回，归一化 (x/255-0.5)/0.5 在 GPU 上做 —— 原先在 worker 里把
460800 个元素转 float32 再归一，实测是数据加载的主要瓶颈。
"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# memmap 是只读的，torch.from_numpy 会警告；我们只读不改，无需理会
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


class LRS2MixDataset(Dataset):
    def __init__(self, cache_dir, split="train", cfg=None, epoch=0,
                 deterministic=False, limit=0, use_enroll=False):
        self.split = split
        self.cfg = cfg
        self.epoch = epoch
        self.deterministic = deterministic          # True 用于 val/test：不随 epoch 变化
        self.use_enroll = use_enroll

        root = Path(cache_dir) / split
        if not (root / "audio.npy").exists():
            raise FileNotFoundError(f"缺少缓存 {root}，请先运行 prepare_cache.py --split {split}")

        self.audio = np.load(root / "audio.npy", mmap_mode="r")     # [N, T] float16
        self.video = np.load(root / "video.npy", mmap_mode="r")     # [N, F, H, W] uint8
        meta = json.loads((root / "meta.json").read_text())

        uid = np.array([m["uid"] for m in meta])
        n_meta = len(uid)
        # 缓存文件尾部可能比 meta 长（memmap 预分配），以 meta 为准
        self.audio = self.audio[:n_meta]
        self.video = self.video[:n_meta]

        f0 = np.array([m["f0"] for m in meta], dtype=np.float32)
        spk = np.array([u.split("/")[0] for u in uid])

        keep = np.arange(n_meta)
        if limit and limit < len(keep):
            keep = keep[:limit]

        self.uid = uid[keep]
        self.f0 = f0[keep]
        self.spk = spk[keep]

        # 按基频排序，用于快速找"难样本对"
        order = np.argsort(self.f0, kind="stable")
        self._f0_sorted = self.f0[order]
        self._idx_sorted = order

        # 说话人 -> 片段索引（enrollment 用）
        self._by_spk = {}
        for k, s in enumerate(self.spk):
            self._by_spk.setdefault(s, []).append(k)
        self._by_spk = {s: np.array(v) for s, v in self._by_spk.items()}

        self.n = len(self.uid)
        self.seed = cfg.seed if cfg is not None else 2024

    def __len__(self):
        return self.n

    # ---------- 内部工具 ----------
    def _hard_partner(self, i, rng):
        """在 ±f0_tol 范围内找一个不同说话人的片段"""
        tol = self.cfg.f0_tol
        lo = np.searchsorted(self._f0_sorted, self.f0[i] - tol, side="left")
        hi = np.searchsorted(self._f0_sorted, self.f0[i] + tol, side="right")
        cand = self._idx_sorted[lo:hi]
        cand = cand[(cand != i) & (self.spk[cand] != self.spk[i]) & (self.f0[cand] > 0)]
        if cand.size == 0:
            return None
        return int(cand[rng.integers(cand.size)])

    def _random_partner(self, i, rng):
        for _ in range(8):
            j = int(rng.integers(self.n))
            if j != i and self.spk[j] != self.spk[i]:
                return j
        return int((i + 1) % self.n)

    def _enroll_index(self, i, rng):
        """同说话人的另一条语音；没有则返回 None"""
        c = self._by_spk.get(self.spk[i])
        if c is None:
            return None
        c = c[c != i]
        if c.size == 0:
            return None
        return int(c[rng.integers(c.size)])

    @staticmethod
    def _rms_norm(x, eps=1e-6):
        return x / (np.sqrt(np.mean(x.astype(np.float32) ** 2)) + eps)

    def __getitem__(self, i):
        ep = 0 if self.deterministic else self.epoch
        rng = np.random.default_rng(self.seed + i * 97 + ep * 100003)

        want_hard = rng.random() < self.cfg.hard_pair_ratio
        j = self._hard_partner(i, rng) if want_hard else None
        is_hard = j is not None
        if j is None:
            j = self._random_partner(i, rng)

        tgt = np.asarray(self.audio[i], dtype=np.float32)
        itf = np.asarray(self.audio[j], dtype=np.float32)

        snr = float(rng.uniform(self.cfg.snr_min, self.cfg.snr_max))
        s = self._rms_norm(tgt)
        n = self._rms_norm(itf) * (10.0 ** (-snr / 20.0))
        mix = s + n

        peak = float(np.abs(mix).max())
        if peak > 0.95:                       # 防止削顶，三路同比例缩放（不改变 SI-SNRi）
            g = 0.95 / peak
            mix, s, n = mix * g, s * g, n * g

        item = {
            "mix": torch.from_numpy(mix).float().unsqueeze(0),        # [1, T]
            "target": torch.from_numpy(s).float().unsqueeze(0),       # [1, T]
            "video": torch.from_numpy(np.ascontiguousarray(self.video[i])).unsqueeze(0),  # uint8 [1,F,H,W]
            "is_hard": is_hard,
            "snr": snr,
            "index": i,
            "partner_index": j,
            "uid": self.uid[i],
            "partner_uid": self.uid[j],
        }

        # 是否具备同说话人参考语音：与是否真的喂给模型无关，始终记录，
        # 保证所有模型（含注册基线）能在同一子集上比较
        e = self._enroll_index(i, rng)
        item["has_enroll"] = e is not None
        if self.use_enroll:
            if e is None:
                item["enroll"] = torch.zeros(1, self.cfg.clip_samples)
            else:
                item["enroll"] = torch.from_numpy(
                    self._rms_norm(np.asarray(self.audio[e], dtype=np.float32))).float().unsqueeze(0)
        return item


def collate(batch):
    """把 dict 列表拼成 batch（字符串字段保留为 list）"""
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, 0)
        else:
            out[k] = vals
    return out


def make_loader(cache_dir, split, cfg, epoch=0, deterministic=False,
                batch_size=None, shuffle=None, limit=0, use_enroll=False):
    from torch.utils.data import DataLoader
    ds = LRS2MixDataset(cache_dir, split=split, cfg=cfg, epoch=epoch,
                        deterministic=deterministic, limit=limit, use_enroll=use_enroll)
    bs = batch_size or cfg.batch_size
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=cfg.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=shuffle,
        # 注意：不能开 persistent_workers —— 每个 epoch 会更新 dataset.epoch，
        # 常驻 worker 拿到的是 fork 时的旧副本，增强就会失效。
        persistent_workers=False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
