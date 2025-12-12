## Comments Challenge (Social Media) — End-to-end workflow

This folder contains the full pipeline for **Challenge 1** (social media comments classification, **9 classes**, metric = **Macro‑F1**).

### 1) Data notes (important)

- **Your CSV headers are French** and may look “broken” on some machines (e.g. `R�seau Social`).
- The code auto-detects and renames columns to canonical names:
  - `id` → `id`
  - `Réseau Social` → `platform`
  - `Commentaire client` → `comment`
  - `Class` (train only) → `label`

### 2) Quick EDA & class relationships (before cleaning)

Run:

```bash
python forca-hack/comments/src/profile_data.py --after_clean
python forca-hack/comments/src/analyze_relations.py --after_clean --out_json forca-hack/outputs/comments/analysis/relations.json
```

This gives you:
- platform distribution
- label distribution
- comment length stats
- per-class token signatures + class similarity graph (Jaccard on top tokens)
- examples per class (raw + cleaned)

### 3) Generate clean datasets (reproducible)

```bash
python forca-hack/comments/src/make_clean_dataset.py
python forca-hack/comments/src/validate_preprocess.py
```

Outputs by default:
- `forca-hack/outputs/comments/clean/train_clean.csv`
- `forca-hack/outputs/comments/clean/test_clean.csv`

### 4) Fast baselines (CPU)

#### TF‑IDF + Logistic Regression (strong baseline)

```bash
python forca-hack/comments/src/train_tfidf_lr.py --n_splits 5
```

This prints per-fold Macro‑F1 and saves artifacts under:
`forca-hack/outputs/comments/models/tfidf_lr/`

#### CatBoost (text + platform + numeric features)

```bash
python forca-hack/comments/src/train_catboost.py --n_splits 5 --iterations 3000
```

Saves under:
`forca-hack/outputs/comments/models/catboost/`

### 5) Transformers fine‑tuning (GPU / Colab recommended)

#### Install (Colab)

```bash
pip install -r forca-hack/requirements.txt
```

#### Train XLM‑R

```bash
python forca-hack/comments/src/train_transformer.py \
  --model_name xlm-roberta-base \
  --n_splits 5 \
  --max_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --epochs 4 \
  --lr 3e-5 \
  --add_meta --add_flags \
  --use_class_weights \
  --fp16
```

#### Train DziriBERT (Algerian dialect)

If the Hugging Face id is available, use (most common name):
`alger-ia/dziribert`

```bash
python forca-hack/comments/src/train_transformer.py \
  --model_name alger-ia/dziribert \
  --n_splits 5 \
  --max_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --epochs 4 \
  --lr 3e-5 \
  --add_meta --add_flags \
  --use_class_weights \
  --fp16
```

Each run saves a timestamped folder under:
`forca-hack/outputs/comments/models/transformers/`

Inside each run:
- `*_fold1/`, `*_fold2/`, ...: checkpoints
- `label_mapping.json`, `preprocess_config.json`
- `run_summary.json` (contains Macro‑F1 per fold)

### 6) Predict + submission (single model or ensemble)

#### Baseline submission

```bash
python forca-hack/comments/src/predict.py \
  --model_paths forca-hack/outputs/comments/models/tfidf_lr/model.joblib \
  --output_csv forca-hack/outputs/comments/submissions/tfidf_lr_submission.csv
```

#### Ensemble submission (average probabilities)

```bash
python forca-hack/comments/src/predict.py \
  --model_paths \
    forca-hack/outputs/comments/models/tfidf_lr/model.joblib \
    forca-hack/outputs/comments/models/catboost/model.cbm \
  --weights 0.6 0.4 \
  --output_csv forca-hack/outputs/comments/submissions/ensemble_submission.csv
```

#### Transformer submission (fold ensemble)

After training, you will have fold folders like:
- `.../xlm-roberta-base_YYYYMMDD_HHMMSS/xlm-roberta-base_fold1`

To ensemble across folds:

```bash
python forca-hack/comments/src/predict.py \
  --model_paths \
    <RUN_DIR>/xlm-roberta-base_fold1 \
    <RUN_DIR>/xlm-roberta-base_fold2 \
    <RUN_DIR>/xlm-roberta-base_fold3 \
    <RUN_DIR>/xlm-roberta-base_fold4 \
    <RUN_DIR>/xlm-roberta-base_fold5 \
  --transformer_add_meta --transformer_add_flags \
  --output_csv forca-hack/outputs/comments/submissions/xlmr_fold_ensemble.csv
```


