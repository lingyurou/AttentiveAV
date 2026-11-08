"""
LRS3 (TED) 零样本跨库探测 —— 缓存构建器
==========================================
用途：把 LRS3 的 test（必要时 trainval）子集解码成与 LRS2 缓存完全相同的
memmap 格式，使 dataset.py / evaluate.py / models.py 全部零改动复用。

每条样本（与 prepare_cache.py 完全一致）:
    audio: float16 [32000]        -> 2.0s @ 16kHz 单声道
    video: uint8   [50, 96, 96]   -> 2.0s @ 25fps 灰度口型/人脸区域
    f0:    float32                -> 中位基频（难样本对配对用）

说话人 ID：写成 "<spk>/<stem>"，dataset.py 用 uid.split("/")[0] 取说话人，
因此 enrollment 配对与说话人互斥逻辑自动生效。

LRS3 目录结构（v0.4）：root/{pretrain,trainval,test}/*.mp4（部分版本按
说话人分子目录）。说话人提取规则：
    1) 相对路径含目录 -> 用第一级目录名
    2) 文件名含 "___" -> 用第一段
    3) 否则 -> 整个 stem（拿到真实数据后如发现规则不对，改 _speaker_of 即可）

用法:
    python prepare_cache_lrs3.py --root data/LRS3 --split test --workers 24
    # 只需 test 子集即可做零样本探测；若后续做方案 B 重训再加 --split trainval
"""
import argparse
import json
import os
import subprocess
import zlib
from multiprocessing import get_context
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

SR = 16000
FPS = 25
CLIP_SEC = 2.0
CLIP_SAMPLES = int(SR * CLIP_SEC)      # 32000
CLIP_FRAMES = int(FPS * CLIP_SEC)      # 50
MOUTH = 96


def _speaker_of(rel: Path) -> str:
    parts = rel.with_suffix("").parts
    if len(parts) > 1:                       # 按目录分说话人
        return parts[0]
    name = parts[-1]
    if "___" in name:                        # 平铺 + ___ 分隔
        return name.split("___")[0]
    if "_" in name and name.split("_")[0].isalnum() and len(name.split("_")[0]) >= 4:
        return name.split("_")[0]            # 兜底：下划线首段（拿到数据后核对）
    return name


def _ffmpeg_audio(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _ffmpeg_video(path, start):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path),
           "-vf", f"fps={FPS},scale={MOUTH}:{MOUTH},format=gray",
           "-frames:v", str(CLIP_FRAMES), "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    frames = np.frombuffer(out, dtype=np.uint8)
    n = len(frames) // (MOUTH * MOUTH)
    frames = frames[:n * MOUTH * MOUTH].reshape(n, MOUTH, MOUTH)
    if n == 0:
        raise RuntimeError("no video frames")
    if n < CLIP_FRAMES:
        frames = np.concatenate([frames, np.repeat(frames[-1:], CLIP_FRAMES - n, axis=0)], 0)
    return frames


def _estimate_f0(x):
    try:
        import librosa
        if np.abs(x).max() < 1e-6:
            return 0.0
        f0 = librosa.yin(x, fmin=60, fmax=400, sr=SR, frame_length=1024)
        f0 = f0[np.isfinite(f0)]
        return float(np.median(f0)) if f0.size else 0.0
    except Exception:
        return 0.0


def _pick_start(uid, duration):
    seed = zlib.crc32(uid.encode("utf-8"))
    span = max(0.0, duration - CLIP_SEC)
    return 0.0 if span <= 0 else (seed % 10000) / 10000.0 * span


def _process(args):
    path, uid = args
    if not path.exists():
        return None
    try:
        audio = _ffmpeg_audio(path)
        if audio.size < int(0.4 * SR):
            return None
        duration = audio.size / SR
        start = _pick_start(uid, duration)
        frames = _ffmpeg_video(path, start)

        s = int(round(start * SR))
        seg = audio[s:s + CLIP_SAMPLES]
        if seg.size < CLIP_SAMPLES:
            seg = np.pad(seg, (0, CLIP_SAMPLES - seg.size))

        peak = float(np.abs(seg).max())
        if not np.isfinite(peak) or peak < 1e-5:
            return None
        seg = seg / max(peak, 1.0)

        return uid, seg.astype(np.float16), frames, _estimate_f0(seg)
    except Exception:
        return None


def build(root, split, limit, workers, fresh=False, cache=None,
          talks_file=None, split_name=None):
    root = Path(root)
    split_dir = root / split
    if not split_dir.exists():
        # 有些发布版本把 test 放在更深层，自动往下找两层
        cand = [p for p in root.rglob(split) if p.is_dir()]
        if not cand:
            raise SystemExit(f"找不到 {split_dir}，请确认 LRS3 解压后的目录结构")
        split_dir = cand[0]
    print(f"使用目录: {split_dir}")

    vids = sorted(split_dir.rglob("*.mp4"))
    if not vids:
        raise SystemExit(f"{split_dir} 下没有 mp4")
    allowed = None
    if talks_file:
        allowed = set(Path(talks_file).read_text().split())
        print(f"按说话人清单过滤: {len(allowed)} 个 talk ID")
    items = []
    for p in vids:
        rel = p.relative_to(split_dir)
        spk = _speaker_of(rel)
        if allowed is not None and spk not in allowed:
            continue
        uid = f"{spk}/{rel.with_suffix('').as_posix()}"
        items.append((p, uid))
    rng = np.random.default_rng(2024)          # 与 LRS2 相同的打散种子
    rng.shuffle(items)
    if limit > 0:
        items = items[:limit]

    cache = Path(cache or (Path(__file__).parent / "data" / "cache" / "lrs3"))
    out_dir = cache / (split_name or split)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for f in out_dir.glob("*"):
            f.unlink()

    print(f"[lrs3/{split}] 待处理 {len(items)} 条 ...", flush=True)
    audio_mm = np.lib.format.open_memmap(out_dir / "audio_raw.npy", mode="w+",
                                         dtype=np.float16, shape=(len(items), CLIP_SAMPLES))
    video_mm = np.lib.format.open_memmap(out_dir / "video_raw.npy", mode="w+",
                                         dtype=np.uint8, shape=(len(items), CLIP_FRAMES, MOUTH, MOUTH))
    meta, n_ok, n_fail = [], 0, 0
    ctx = get_context("spawn")
    import time
    t0 = time.time()
    with ctx.Pool(workers, maxtasksperchild=200) as pool:
        for i, res in enumerate(pool.imap(_process, items, chunksize=4)):
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
                eta = (len(items) - i - 1) / max(rate, 1e-6) / 60
                print(f"  {i+1}/{len(items)}  ok={n_ok} fail={n_fail} "
                      f"{rate:.1f}/s ETA {eta:.1f}min", flush=True)

    audio_mm.flush(); video_mm.flush()
    del audio_mm, video_mm
    os.replace(out_dir / "audio_raw.npy", out_dir / "audio.npy")
    os.replace(out_dir / "video_raw.npy", out_dir / "video.npy")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

    n_spk = len({m["uid"].split("/")[0] for m in meta})
    print(f"[lrs3/{split}] 完成: {n_ok} 条成功 / {n_fail} 条失败, {n_spk} 个说话人 -> {out_dir}")
    return n_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).parent / "data" / "LRS3"),
                    help="LRS3 解压根目录（其下应有 test/ 等子目录）")
    ap.add_argument("--split", default="test", choices=["test", "trainval", "pretrain"])
    ap.add_argument("--limit", type=int, default=0, help="0 表示全部")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cache", default=None, help="缓存输出目录，默认 data/cache/lrs3")
    ap.add_argument("--talks-file", default=None,
                    help="说话人清单文件（每行一个 talk ID），只处理清单内的说话人")
    ap.add_argument("--split-name", default=None,
                    help="输出子目录名，默认与 --split 相同（可设为 train/val）")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    build(a.root, a.split, a.limit, a.workers, a.fresh, a.cache,
          a.talks_file, a.split_name)
