# Colab Notebook - Comments Classification (Step-by-Step)

**Goal**: Train XLM-R + DziriBERT, ensemble, and generate Kaggle submission with Macro-F1 > 0.7

---

## Setup (Run First)

### Code Block 1: Mount Google Drive + Install Dependencies

```python
# Mount Google Drive
from google.colab import drive
import os
import shutil

# If you ever see: "Mountpoint must not already contain files"
# it means /content/drive is not empty. Clean it and remount.
shutil.rmtree('/content/drive', ignore_errors=True)
os.makedirs('/content/drive', exist_ok=True)
drive.mount('/content/drive', force_remount=True)

# Install dependencies (from repo requirements)
!pip install -r forca-hack/requirements.txt

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Code Block 2: Clone/Pull Repository

```python
!git clone <YOUR_REPO_URL>
%cd <YOUR_REPO_FOLDER>/forca-hack
```

---

## Data Validation & Preprocessing

### Code Block 3: Check Data Files Exist

```python
from pathlib import Path

# 1) Find where your dataset is on Drive (prints candidates)
!find /content/drive/MyDrive -type f -name "train.csv" | head -20
!find /content/drive/MyDrive -type f -name "test_file.csv" | head -20

# 2) Set your data folder here (the folder that contains train.csv and test_file.csv)
DATA_DIR = Path("/content/drive/MyDrive/<PUT_YOUR_FOLDER_HERE>")
print("DATA_DIR exists:", DATA_DIR.exists(), DATA_DIR)
print("Files:", [p.name for p in DATA_DIR.glob("*.csv")])
```

### Code Block 4: Validate Preprocessing (Sanity Checks)

```python
# Run validation (use --data_dir so folder naming doesn't matter)
!python comments/src/validate_preprocess.py \
  --data_dir "$DATA_DIR" \
  --train_filename train.csv \
  --test_filename test_file.csv
```

### Code Block 5: Generate Cleaned Datasets

```python
# Where to write outputs on Drive
OUT_DIR = Path("/content/drive/MyDrive/forsa_outputs/comments/clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

!python comments/src/make_clean_dataset.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --train_filename train.csv \
  --test_filename test_file.csv

print("Cleaned outputs:", list(OUT_DIR.glob("*.csv")))
```

### Code Block 6: (Optional) Dataset Analysis Report

```python
# Generate relationships report (helps understand class patterns)
!python comments/src/analyze_relations.py \
  --data_dir "$DATA_DIR" \
  --train_filename train.csv \
  --after_clean \
  --out_json /content/drive/MyDrive/forsa_outputs/comments/analysis/relations.json

print("\nAnalysis complete. Check the JSON file for detailed class relationships.")
```

---

## Model Training (Two Transformers)

### Code Block 7: Train XLM-RoBERTa (5-Fold CV)

```python
# Train XLM-R with metadata + flags
!python comments/src/train_transformer.py \
  --data_dir "$DATA_DIR" \
  --output_dir /content/drive/MyDrive/forsa_outputs/comments/models/transformers \
  --train_filename train.csv \
  --model_name xlm-roberta-base \
  --n_splits 5 \
  --max_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --epochs 4 \
  --lr 3e-5 \
  --add_meta --add_flags \
  --use_class_weights \
  --fp16

# This will print CV scores and save fold checkpoints
# Expected output: [fold 1] macro_f1=0.XXXXX, etc.
# Watch for: macro_f1 mean >= 0.72 (good sign for >0.7 on Kaggle)
# Training time: ~30-60 minutes depending on GPU
```

### Code Block 8: Train DziriBERT (5-Fold CV)

```python
# Train DziriBERT (Algerian dialect model)
!python comments/src/train_transformer.py \
  --data_dir "$DATA_DIR" \
  --output_dir /content/drive/MyDrive/forsa_outputs/comments/models/transformers \
  --train_filename train.csv \
  --model_name alger-ia/dziribert \
  --n_splits 5 \
  --max_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --epochs 4 \
  --lr 3e-5 \
  --add_meta --add_flags \
  --use_class_weights \
  --fp16

# Same as above - watch CV scores
# Training time: ~30-60 minutes depending on GPU
```

### Code Block 9: Check Training Results

```python
# List all trained model runs
import json
from pathlib import Path

models_dir = Path('/content/drive/MyDrive/FORSA_team/outputs/comments/models/transformers')
runs = sorted([d for d in models_dir.iterdir() if d.is_dir()]) if models_dir.exists() else []

if not runs:
    models_dir = Path('/content/drive/MyDrive/forsa_outputs/comments/models/transformers')
    runs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
runs = sorted([d for d in models_dir.iterdir() if d.is_dir()])

print("Trained model runs:")
for run_dir in runs:
    summary_file = run_dir / 'run_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"\n{run_dir.name}:")
        print(f"  CV Macro-F1: {summary.get('macro_f1_mean', 0):.5f} ± {summary.get('macro_f1_std', 0):.5f}")
        print(f"  Fold scores: {summary.get('fold_scores', [])}")
    else:
        print(f"\n{run_dir.name}: (no summary yet)")

# Find the two best runs (one XLM-R, one DziriBERT)
xlmr_runs = [r for r in runs if 'xlm-roberta-base' in r.name]
dziri_runs = [r for r in runs if 'dziribert' in r.name]

print(f"\nXLM-R runs: {len(xlmr_runs)}")
print(f"DziriBERT runs: {len(dziri_runs)}")

# Use the most recent run of each (or pick by highest CV score)
if xlmr_runs:
    xlmr_run = xlmr_runs[-1]  # Most recent
    print(f"\nSelected XLM-R run: {xlmr_run.name}")
if dziri_runs:
    dziri_run = dziri_runs[-1]  # Most recent
    print(f"Selected DziriBERT run: {dziri_run.name}")
```

---

## Generate Submission (Ensemble)

### Code Block 10: Create Final Ensemble Submission

```python
# Find the two run directories (adjust names based on Code Block 9 output)
from pathlib import Path

models_dir = Path('/content/drive/MyDrive/FORSA_team/outputs/comments/models/transformers')
runs = sorted([d for d in models_dir.iterdir() if d.is_dir()]) if models_dir.exists() else []

if not runs:
    models_dir = Path('/content/drive/MyDrive/forsa_outputs/comments/models/transformers')
    runs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
runs = sorted([d for d in models_dir.iterdir() if d.is_dir()])

# Find XLM-R and DziriBERT runs (use most recent)
xlmr_runs = [r for r in runs if 'xlm-roberta-base' in r.name and r.name.count('_') >= 2]
dziri_runs = [r for r in runs if 'dziribert' in r.name and r.name.count('_') >= 2]

if not xlmr_runs or not dziri_runs:
    print("❌ Need both XLM-R and DziriBERT runs. Check Code Block 9 output.")
else:
    xlmr_base = xlmr_runs[-1]  # Most recent
    dziri_base = dziri_runs[-1]  # Most recent

    print(f"Using XLM-R run: {xlmr_base.name}")
    print(f"Using DziriBERT run: {dziri_base.name}")

    # Build list of all 10 fold checkpoints (5 XLM-R + 5 DziriBERT)
    model_paths = []
    for fold in range(1, 6):
        xlmr_fold = xlmr_base / f'xlm-roberta-base_fold{fold}'
        dziri_fold = dziri_base / f'alger-ia_dziribert_fold{fold}'
        if xlmr_fold.exists():
            model_paths.append(str(xlmr_fold))
        if dziri_fold.exists():
            model_paths.append(str(dziri_fold))

    print(f"\nFound {len(model_paths)} model checkpoints")

    # Create submission (join paths with spaces for command line)
    model_paths_str = ' '.join(model_paths)

    !python comments/src/predict.py \
      --data_dir "$DATA_DIR" \
      --test_filename test_file.csv \
      --model_paths {model_paths_str} \
      --transformer_add_meta --transformer_add_flags \
      --output_csv /content/drive/MyDrive/forsa_outputs/comments/submissions/final_ensemble.csv

    print("\n✅ Submission file created!")
```

### Code Block 10b: Alternative - Manual Path Specification (If Code Block 10 fails)

```python
# If you know the exact run names, use this simpler version:
# Replace the timestamps with your actual run folder names

xlmr_run = "xlm-roberta-base_20241212_120000"  # Replace with actual
dziri_run = "alger-ia_dziribert_20241212_130000"  # Replace with actual

base_dir = "/content/drive/MyDrive/FORSA_team/outputs/comments/models/transformers"

# Build the command manually
cmd = f"""python /content/drive/MyDrive/FORSA_team/forca-hack/comments/src/predict.py \\
  --team_dir /content/drive/MyDrive/FORSA_team \\
  --test_filename test_file.csv \\
  --model_paths \\
    {base_dir}/{xlmr_run}/xlm-roberta-base_fold1 \\
    {base_dir}/{xlmr_run}/xlm-roberta-base_fold2 \\
    {base_dir}/{xlmr_run}/xlm-roberta-base_fold3 \\
    {base_dir}/{xlmr_run}/xlm-roberta-base_fold4 \\
    {base_dir}/{xlmr_run}/xlm-roberta-base_fold5 \\
    {base_dir}/{dziri_run}/alger-ia_dziribert_fold1 \\
    {base_dir}/{dziri_run}/alger-ia_dziribert_fold2 \\
    {base_dir}/{dziri_run}/alger-ia_dziribert_fold3 \\
    {base_dir}/{dziri_run}/alger-ia_dziribert_fold4 \\
    {base_dir}/{dziri_run}/alger-ia_dziribert_fold5 \\
  --transformer_add_meta --transformer_add_flags \\
  --output_csv /content/drive/MyDrive/FORSA_team/outputs/comments/submissions/final_ensemble.csv"""

print("Run this command:")
print(cmd)
# Uncomment to execute:
# !{cmd}
```

### Code Block 11: Verify Submission Format

```python
# Check submission file
sub_path = Path('/content/drive/MyDrive/FORSA_team/outputs/comments/submissions/final_ensemble.csv')
if sub_path.exists():
    df_sub = pd.read_csv(sub_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Columns: {list(df_sub.columns)}")
    print(f"\nFirst 10 rows:")
    print(df_sub.head(10))
    print(f"\nClass distribution:")
    print(df_sub['Class'].value_counts().sort_index())
    print(f"\nFile size: {sub_path.stat().st_size / 1024:.1f} KB")
    print(f"\n✅ Ready for Kaggle upload!")
else:
    print("❌ Submission file not found. Check the predict.py command.")
```

### Code Block 12: Download Submission File

```python
# The file is already in Drive, so you can download it from the file browser
# Or use this to copy to Colab's local filesystem for easy download
import shutil

sub_path = Path('/content/drive/MyDrive/FORSA_team/outputs/comments/submissions/final_ensemble.csv')
local_path = Path('/content/final_ensemble.csv')

if sub_path.exists():
    shutil.copy(sub_path, local_path)
    print(f"✅ Copied to: {local_path}")
    print("You can now download it from Colab's file browser (left sidebar)")
else:
    print("❌ Submission file not found")
```

---

## Troubleshooting

### If GPU OOM (Out of Memory):

```python
# Reduce batch size and use gradient accumulation
# Change: --train_batch_size 8 --gradient_accumulation_steps 2
# Or: --train_batch_size 4 --gradient_accumulation_steps 4
```

### If CV Score < 0.7:

```python
# Try these hyperparameter tweaks:
# 1. Lower learning rate: --lr 2e-5
# 2. Longer sequences: --max_length 160
# 3. More epochs: --epochs 5 (early stopping will still apply)
```

### If Model Download Fails:

```python
# DziriBERT might need authentication or different path
# Check: https://huggingface.co/alger-ia/dziribert
# You may need to login: !huggingface-cli login
```

---

## Quick Test Run (Before Full Training)

### Code Block 0: Quick Sanity Check (Optional)

```python
# Run a tiny 2-fold test to verify everything works
!python /content/drive/MyDrive/FORSA_team/forca-hack/comments/src/train_transformer.py \
  --team_dir /content/drive/MyDrive/FORSA_team \
  --train_filename train.csv \
  --model_name xlm-roberta-base \
  --n_splits 2 \
  --max_length 128 \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --epochs 1 \
  --lr 3e-5 \
  --add_meta --add_flags \
  --use_class_weights \
  --fp16

# If this completes without errors, the full training will work
# This takes ~5-10 minutes (vs 30-60 for full 5-fold)
```

---

## Summary Checklist

- [ ] Code Block 1: Mount Drive + Install deps
- [ ] Code Block 2: Navigate to code
- [ ] Code Block 3: Verify data files exist
- [ ] Code Block 4: Validate preprocessing
- [ ] Code Block 5: Generate cleaned datasets
- [ ] Code Block 6: (Optional) Run analysis
- [ ] Code Block 7: Train XLM-R (5-fold)
- [ ] Code Block 8: Train DziriBERT (5-fold)
- [ ] Code Block 9: Check CV scores (should be ≥ 0.72)
- [ ] Code Block 10: Create ensemble submission
- [ ] Code Block 11: Verify submission format
- [ ] Code Block 12: Download submission

**Expected timeline:**

- Setup: 5 min
- Preprocessing: 2 min
- XLM-R training: ~30-60 min (depends on GPU)
- DziriBERT training: ~30-60 min
- Ensemble + submission: 5 min

**Total: ~1.5-2 hours**
