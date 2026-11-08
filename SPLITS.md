# Data splits

- **LRS3-TED**: speaker-disjoint talk lists are provided in `data/lrs3_train_talks.txt`,
  `data/lrs3_val_talks.txt`, `data/lrs3_test_talks.txt` (4,365 talks in total;
  test = 250 talks / 5,918 clips, val = 200 talks / 4,729 clips, train = 3,913 talks / 90,892 clips).
- **LRS2**: the speaker-disjoint train/val/test assignment is constructed deterministically by
  `prepare_cache.py` when building the memmap caches (`data/cache/<corpus>/{train,val,test}`),
  so re-running the script reproduces the exact same split without a separate list file.
- Test mixtures are generated deterministically at load time by `dataset.py` /
  `utils.py`; the seed fully determines the pairing, SNR and shift, so every number in
  the paper is reproducible from the caches and these lists.
- The LRS2 and LRS3-TED corpora themselves are NOT redistributed; obtain them from their
  original distributors under their licences.
