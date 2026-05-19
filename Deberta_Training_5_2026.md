# DeBERTa Binary Classifier

Fine-tunes `deberta-v3-base` for binary flagging. Handles class imbalance (weighted loss), long texts (chunking), and synthetic data leakage (group-stratified splits by `base_id`).

## Setup

```bash
pip install torch transformers datasets scikit-learn optuna pandas
```

## Your data

CSV with at minimum these columns:

| Column | Description |
|--------|-------------|
| `text` | The text to classify |
| `label` | 0 (negative) or 1 (positive) — string labels get auto-mapped |
| `base_id` | Links synthetic variants to their parent sample |

Other columns are ignored.

## First run (search + train)

```bash
python deberta_binary_pipeline.py \
    --mode full \
    --data_path data.csv \
    --text_column text \
    --label_column label \
    --base_id_column base_id \
    --n_trials 15 \
    --cv_folds 3
```

This does two things back to back:
1. **Search** — runs 15 Optuna trials × 3-fold CV to find the best hyperparameters
2. **Train** — trains a final model with those params, tunes the flagging threshold, saves everything to `./results/final_model/`

Takes a few hours on a single GPU. To speed up the search, drop to `--n_trials 10 --cv_folds 2`.

You can also run search and train separately:

```bash
# Step 1: find best hyperparameters
python deberta_binary_pipeline.py --mode search --data_path data.csv

# Step 2: train final model (auto-loads best params from step 1)
python deberta_binary_pipeline.py --mode train --data_path data.csv
```

### What gets saved

```
results/
├── best_hyperparameters.json   # from search
├── final_results.json          # test set metrics
├── final_model/
│   ├── model.safetensors       # weights
│   ├── tokenizer files
│   ├── config.json             # full pipeline config
│   └── threshold.json          # tuned flagging threshold
```

## Inference

```bash
python deberta_binary_pipeline.py \
    --mode inference \
    --data_path new_data.csv \
    --text_column text
```

Outputs `results/predictions.csv` with columns: `flagged`, `prob_positive`, `pred_label`, `pred_confidence`.

The tuned threshold is loaded automatically. To override it:

```python
from deberta_binary_pipeline import Classifier

clf = Classifier("./results/final_model", threshold=0.35)
result = clf.predict_single("some text to check")
# {'label': 1, 'flagged': True, 'confidence': 0.91, 'prob_positive': 0.91, ...}
```

## Key config options

Edit defaults in the `Config` class or pass via CLI:

| Flag | Default | What it does |
|------|---------|--------------|
| `--n_trials` | 15 | Optuna search trials |
| `--cv_folds` | 3 | CV folds per trial |
| `--output_dir` | `./results` | Where everything saves |
| `--seed` | 42 | Reproducibility seed |

Inside `Config` (edit in the script):

| Setting | Default | Notes |
|---------|---------|-------|
| `use_class_weights` | `True` | Handles your 60/40 imbalance |
| `max_length` | 512 | Long texts get chunked, nothing lost |
| `chunk_overlap` | 128 | Overlap between chunks |
| `early_stopping_patience` | 3 | Stops if val F1 plateaus |

## After your first run

Check the **threshold sweep table** in the output. If you want to catch more positives at the cost of more false flags, lower the threshold. You can re-run inference with a different threshold without retraining.
