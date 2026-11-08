"""
LRS2 真实数据缓存构建器
========================
把 LRS2 的 mp4 片段解码成固定长度的 (音频, 口型视频) 张量，写成 memmap，
后续 DataLoader 直接读内存映射，避免训练时反复调用 ffmpeg。

每条样本:
    audio: float16 [32000]        -> 2.0s @ 16kHz 单声道
    video: uint8   [50, 96, 96]   -> 2.0s @ 25fps 灰度口型区域
    f0:    float32                -> 该片段的中位基频（用于构造难样本对）

用法:
    python prepare_cache.py --split train --limit 4000 --workers 24
    python prepare_cache.py --split val   --limit 300  --workers 24
    python prepare_cache.py --split test  --limit 300  --workers 24
"""
import argparse
import json
import os
import subprocess
import sys
import zlib
from multiprocessing import get_context
from pathlib import Path

# 必须在 numpy / OpenBLAS 初始化前压住线程数：
# 否则 fork 出来的子进程会在 OpenMP 线程池上死锁（实测 19 分钟零写入）。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# ---------------- 常量 ----------------
SR = 16000
FPS = 25
CLIP_SEC = 2.0
CLIP_SAMPLES = int(SR * CLIP_SEC)      # 32000
CLIP_FRAMES = int(FPS * CLIP_SEC)      # 50
MOUTH = 96                             # 口型 ROI 边长

ROOT = Path(__file__).parent / "data" / "lrs3" / "LRS2数据集"
CACHE = Path(__file__).parent / "data" / "cache" / "lrs2"


def _ffmpeg_audio(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _ffmpeg_video(path, start):
    """从 start 秒开始抓 50 帧灰度 96x96（-ss 放在 -i 前做快速定位）"""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path),
           "-vf", f"fps={FPS},scale={MOUTH}:{MOUTH},format=gray",
           "-frames:v", str(CLIP_FRAMES), "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    frames = np.frombuffer(out, dtype=np.uint8)
    n = len(frames) // (MOUTH * MOUTH)
    frames = frames[:n * MOUTH * MOUTH].reshape(n, MOUTH, MOUTH)
    if n == 0:
        raise RuntimeError("no video frames")
    if n < CLIP_FRAMES:                      # 不足则用最后一帧补齐
        frames = np.concatenate([frames, np.repeat(frames[-1:], CLIP_FRAMES - n, axis=0)], 0)
    return frames


def _estimate_f0(x):
    """中位基频估计（YIN），失败返回 0。用于配对难易度分层。"""
    try:
        import librosa
        if np.abs(x).max() < 1e-6:
            return 0.0
        f0 = librosa.yin(x, fmin=60, fmax=400, sr=SR, frame_length=1024)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            return 0.0
        return float(np.median(f0))
    except Exception:
        return 0.0


def _pick_start(uid, duration):
    """按样本 id 哈希决定起始点：可复现，且不同样本错开"""
    seed = zlib.crc32(uid.encode("utf-8"))
    span = max(0.0, duration - CLIP_SEC)
    if span <= 0:
        return 0.0
    return (seed % 10000) / 10000.0 * span


def _process(args):
    uid, mode = args
    path = ROOT / "mvlrs_v1" / mode / (uid + ".mp4")
    if not path.exists():
        return None
    try:
        audio = _ffmpeg_audio(path)
        if audio.size < int(0.4 * SR):        # 太短，丢弃
            return None
        duration = audio.size / SR
        start = _pick_start(uid, duration)
        frames = _ffmpeg_video(path, start)

        s = int(round(start * SR))
        seg = audio[s:s + CLIP_SAMPLES]
        if seg.size < CLIP_SAMPLES:           # 尾部不足则回退/补零
            seg = np.pad(seg, (0, CLIP_SAMPLES - seg.size))

        peak = float(np.abs(seg).max())
        if not np.isfinite(peak) or peak < 1e-5:
            return None
        # 仅在削顶时压一下，不做响度归一（响度在 Dataset 里按 RMS 统一处理）
        seg = seg / max(peak, 1.0)

        f0 = _estimate_f0(seg)
        return uid, seg.astype(np.float16), frames, f0
    except Exception:
        return None


def build(split, limit, workers, fresh=False):
    list_file = ROOT / f"{split}.txt"
    if not list_file.exists():
        sys.exit(f"找不到 {list_file}")
    uids = [ln.split()[0] for ln in list_file.read_text().splitlines() if ln.strip()]
    # 打散但固定顺序，保证不同 limit 下的子集是同一分布
    rng = np.random.default_rng(2024)
    uids = list(rng.permutation(uids))
    if limit > 0:
        uids = uids[:limit]

    out_dir = CACHE / split
    out_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for f in out_dir.glob("*"):
            f.unlink()

    mode = "pretrain" if split == "pretrain" else "main"
    print(f"[{split}] 待处理 {len(uids)} 条（mode={mode}）...", flush=True)

    audio_mm = np.lib.format.open_memmap(out_dir / "audio_raw.npy", mode="w+",
                                         dtype=np.float16, shape=(len(uids), CLIP_SAMPLES))
    video_mm = np.lib.format.open_memmap(out_dir / "video_raw.npy", mode="w+",
                                         dtype=np.uint8, shape=(len(uids), CLIP_FRAMES, MOUTH, MOUTH))

    meta, n_ok, n_fail = [], 0, 0
    # spawn 而不是 fork：子进程里要调 ffmpeg + librosa，fork 后极易死锁
    ctx = get_context("spawn")
    import time
    t0 = time.time()
    with ctx.Pool(workers, maxtasksperchild=200) as pool:
        for i, res in enumerate(pool.imap(_process, [(u, mode) for u in uids], chunksize=4)):
            if res is None:
                n_fail += 1
            else:
                uid, seg, frames, f0 = res
                audio_mm[n_ok] = seg
                video_mm[n_ok] = frames
                meta.append({"uid": uid, "f0": round(f0, 2)})
                n_ok += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                rate = (i + 1) / max(el, 1e-6)
                eta = (len(uids) - i - 1) / max(rate, 1e-6) / 60
                print(f"  {i+1}/{len(uids)}  ok={n_ok} fail={n_fail} "
                      f"{rate:.1f}/s ETA {eta:.1f}min", flush=True)

    audio_mm.flush(); video_mm.flush()
    del audio_mm, video_mm
    # 文件尾部可能多出未使用的槽位，由 meta 里的条数决定有效范围（Dataset 侧切片）
    os.replace(out_dir / "audio_raw.npy", out_dir / "audio.npy")
    os.replace(out_dir / "video_raw.npy", out_dir / "video.npy")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(f"[{split}] 完成: {n_ok} 条成功 / {n_fail} 条失败 -> {out_dir}")
    return n_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "pretrain"])
    ap.add_argument("--limit", type=int, default=4000, help="0 表示全部")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    build(a.split, a.limit, a.workers, a.fresh)
