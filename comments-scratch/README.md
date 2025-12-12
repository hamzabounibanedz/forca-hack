## comments-scratch — multi-expert “from-scratch” models + merging

This folder adds **scratch (non-pretrained) models** for the Social Media Comments task (9 classes),
trained **from your dataset only**, and designed to be **merged/ensembled** with the existing
transformer pipeline in `forca-hack/comments/`.

### What’s inside

- **4 expert models** trained “from scratch”:
  - **French** (word-level TextCNN)
  - **Arabic** (char-level TextCNN)
  - **Darija (Arabic script)** (char-level TextCNN)
  - **Darija (Latin/Arabizi)** (char-level TextCNN)
- **Language/script gating**: per-comment weights decide how much each expert contributes.
- **Final merging**: combine _scratch experts_ with your **existing transformer ensemble** (DziriBERT)
  via weighted probability averaging.

### Why this helps

- Your labeled train (~1.8k) is small and noisy; scratch models alone won’t beat DziriBERT, but:
  - **char models** are robust to typos/Arabizi and can add complementary signal
  - **expert gating** can reduce “wrong-language” noise in the ensemble
  - **merging** is the main win: transformer stays the backbone; experts are “specialists”

### Quick start (local)

0. Analyze train vs 30k test (recommended before augmenting):

```bash
python forca-hack/comments-scratch/src/analyze_data.py
```

1. Train scratch experts (creates a run directory with folds/models):

```bash
python forca-hack/comments-scratch/src/train_experts.py
```

2. Predict + merge with an existing transformer run directory:

```bash
python forca-hack/comments-scratch/src/predict.py --scratch_run_dir forca-hack/outputs/comments-scratch/models/experts_charcnn/<RUN_DIR> --transformer_run_dirs forca-hack/outputs/comments/models/transformers/<TRANSFORMER_RUN_DIR> --w_transformer 0.75 --output_csv submission.csv
```

### Data location (local)

By default the scratch scripts read:
- `forca-hack/comments-scratch/data/train.csv`
- `forca-hack/comments-scratch/data/test_file.csv`

### (Recommended) pick `w_transformer` using OOF optimization

If you have:

- scratch run dir (with experts `oof_proba.npy`) and
- transformer run dir(s) (with fold checkpoints + `run_summary.json`)

you can estimate the best blending weight offline (OOF):

```bash
python forca-hack/comments-scratch/src/optimize_merge.py --scratch_run_dir forca-hack/outputs/comments-scratch/models/experts_charcnn/<RUN_DIR> --transformer_run_dirs forca-hack/outputs/comments/models/transformers/<TRANSFORMER_RUN_DIR>
```

### (Recommended) “augmentation” via pseudo-labeling (test → pseudo → retrain)

1) Use your transformer ensemble to save test probabilities (example):

```bash
python forca-hack/comments/src/predict.py --data_dir forca-hack/comments-scratch/data --run_dirs <TRANSFORMER_RUN_DIR> --transformer_add_flags --save_proba_npy proba_test.npy --output_csv submission_tmp.csv
```

2) Convert probabilities to a high-precision pseudo-labeled CSV:

```bash
python forca-hack/comments-scratch/src/make_pseudo_labels.py --proba_npy proba_test.npy --mapping_dir <TRANSFORMER_RUN_DIR> --min_conf 0.98 --min_margin 0.10 --per_class_max 1500 --output_csv pseudo.csv
```

3) Retrain scratch experts using pseudo labels (lower weight than real labels):

```bash
python forca-hack/comments-scratch/src/train_experts.py --pseudo_csv pseudo.csv --pseudo_min_conf 0.98 --pseudo_weight 0.4 --pseudo_use_confidence --pseudo_per_class_max 1500
```

### Append pseudo-labeled rows into an augmented train.csv (requested workflow)

This produces a new file `train_augmented.csv` that contains:
- original labeled train rows
- selected pseudo-labeled test rows (with new unique ids)

```bash
python forca-hack/comments-scratch/src/augment_train_csv.py --data_dir forca-hack/comments-scratch/data --proba_npys proba_test.npy --mapping_dir <TRANSFORMER_RUN_DIR> --min_conf 0.99 --min_margin 0.10 --per_class_max 333 --target_total 3000 --dedup --add_meta_cols
```

If you really want to overwrite the original `train.csv` (a `.bak` backup is created), add `--inplace`.

### Colab/Drive

All scripts support `--team_dir /content/drive/MyDrive/FORSA_team` and will look for:

- comments data: `<team_dir>/data/comments/` (same convention as `forca-hack/comments`)
- outputs: `<team_dir>/outputs/comments-scratch/`
