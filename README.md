# AttentiveAV

Code release for the manuscript **"Cue, Not Capacity: A Controlled Study of Visual and Speaker-Enrollment Cues for Target-Speaker Extraction"** (under review).

AttentiveAV is a lightweight audio-visual target-speaker extraction model that separates a target speaker's voice from a two-speaker mixture using only **visual cues** (lip motion), and compares it under a strictly matched budget against **speaker-enrollment cues** (reference-voice d-vectors). All comparisons use a single shared backbone and a single training budget, so the observed gaps measure the *cue*, not the *capacity*.

## Model family (equal-capacity comparison)

| kind | Description | Params |
|---|---|---|
| `av` | Ours: cross-modal visual-audio attention + concatenated fusion + auxiliary log-Mel head (dense cross-modal supervision during training) | 0.80 M |
| `av_noattn` | Ablation: attention gate removed (mask head still sees visual features) | 0.80 M |
| `av_nodistill` | Ablation: auxiliary Mel-distillation head removed | 0.80 M |
| `enroll` | Audio + same-speaker reference utterance (FiLM-conditioned d-vector, VoiceFilter-style) | 0.81 M |
| `audio` | Cue-free degenerate control | 0.54 M |
| `pit` | Permutation-invariant training variant | — |

The last four models share the encoder / TCN / mask head / decoder with `av`; differences between them are therefore clean ablation quantities.

## Data protocol

- Mixtures: `x = s_target + s_interferer`, SNR ~ `U(-5, 5)` dB; 50% of samples are "hard pairs" matched by similar fundamental frequency.
- Every clip is 2.0 s / 16 kHz mono; lip crops are 96×96 / 25 fps.
- All sampling (pairing, SNR, shift) is fully determined by `(split, index, epoch)` — evaluation is exactly reproducible, and with the same seed every model sees the same mixtures.
- **LRS3-TED**: speaker-disjoint talk lists are shipped in `data/lrs3_{train,val,test}_talks.txt` (see `SPLITS.md`).
- **LRS2**: the speaker-disjoint split is constructed deterministically by `prepare_cache.py` when building the memmap caches.
- The LRS2 / LRS3-TED corpora are **not** redistributed; obtain them from their original distributors under their licenses.

## Repository layout

```
config.py               all hyperparameters in one place
models.py               model family + unified forward entry (forward_batch)
dataset.py              mixing, hard-pair matching, enrollment sampling
utils.py                STFT/iSTFT, losses, deterministic sampling helpers
train.py                training (AMP, warmup + cosine, best ckpt by val SI-SNRi)
evaluate.py             evaluation (SI-SNRi / SI-SNR / PESQ / STOI, SNR-bucketed)
prepare_cache.py        LRS2: mp4 -> memmap caches (audio float16 / video uint8)
prepare_cache_lrs3.py   LRS3: same, driven by the talk lists in data/
controls.py             visual-cue controls (correct / zeroed / shuffled video)
noise_robust.py         robustness under additive noise
mossformer_model.py     MossFormer2 (42 M) reference baseline
mossformer_eval.py      zero-shot / oracle evaluation of the MossFormer baseline
ours_8k_eval.py         8 kHz band-limited evaluation (matched protocol vs MossFormer)
stream_eval.py          causal streaming / chunked inference evaluation
eff_layers.py           streaming-safe efficiency layers
analyze_cues.py         per-sample cue analysis
attention_analysis.py   attention weight vs lip-motion / target-energy correlation
bench_step.py           per-step timing benchmark
bench_loader.py         data-loader throughput benchmark
probe_eff.py            efficiency probing
collect_numbers.py      aggregate result JSONs into paper tables
make_figures_v2.py      paper figures (300 dpi)
smoke_test.py           tiny end-to-end sanity check (no data download needed)
data/                   LRS3 talk lists; caches and corpora are NOT included
```

## Setup

```bash
pip install -r requirements.txt   # torch >= 2.5 recommended; any CUDA build that suits your GPU

python smoke_test.py --gpu 0      # runs a tiny forward/backward on random tensors
```

`smoke_test.py` verifies the whole model stack without any corpus or checkpoint.

## Preparing data caches

```bash
# LRS3 (uses the talk lists shipped in data/)
python prepare_cache_lrs3.py --split test     --workers 16 --fresh
python prepare_cache_lrs3.py --split trainval --workers 24 --fresh

# LRS2 (split is derived deterministically during cache building)
python prepare_cache.py --split train --limit 0 --workers 24 --fresh
python prepare_cache.py --split val   --limit 0 --workers 16 --fresh
python prepare_cache.py --split test  --limit 0 --workers 16 --fresh
```

Point the corpus root / cache directory at your local copies (`--root`, `--cache` for LRS3; see `config.py` and `prepare_cache.py` for LRS2 defaults).

## Training

```bash
python train.py --kind av --epochs 16 --batch 16 --gpu 0 --tag av_s1 --seed 2024
```

Useful flags: `--kind` selects the model family (see table above), `--seed` controls every stochastic step of the pipeline, `--resume` continues from a checkpoint. The best checkpoint is selected by validation SI-SNRi and saved under `runs/<tag>/`.

## Evaluation

```bash
# Main metric suite (SI-SNRi / SI-SNR / PESQ / STOI, plus the unprocessed-mixture row)
python evaluate.py --ckpt runs/av_s1/best.pt --gpu 0 --with-mixture

# Visual-cue controls: correct vs zeroed vs shuffled speaker video
python controls.py --ckpt runs/av_s1/best.pt --gpu 0

# MossFormer2 42 M reference baseline (zero-shot / oracle)
python mossformer_eval.py --gpu 0

# Ours under the 8 kHz band-limited protocol matched to MossFormer
python ours_8k_eval.py --ckpt runs/av_s1/best.pt --gpu 0

# Causal streaming evaluation over chunk sizes
python stream_eval.py --ckpt runs/av_rf/best.pt --gpu 0

# Additive-noise robustness
python noise_robust.py --ckpt runs/av_s1/best.pt --gpu 0
```

Multiple `--ckpt` flags evaluate several checkpoints on exactly the same deterministic test mixtures.

## Reproducibility

- Mixtures are generated deterministically at load time by `dataset.py` / `utils.py`; the seed fully determines pairing, SNR and shift.
- `SPLITS.md` documents how the speaker-disjoint splits are fixed, so every number in the paper is reproducible from the caches and the shipped talk lists.

## Citation

If you use this code, please cite the manuscript:

```
Jian Guo and Qianqian Cheng.
"Cue, Not Capacity: A Controlled Study of Visual and Speaker-Enrollment Cues
 for Target-Speaker Extraction." Under review, 2026.
```

## License

Released under the [MIT License](LICENSE).
