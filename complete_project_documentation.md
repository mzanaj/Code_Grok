# DeBERTa Text Classification System - Complete Documentation

**Project**: State-of-the-art 5-class Topic Classification  
**Model**: DeBERTa-v3 (Base/Large)  
**Dataset**: 20,000 labeled samples (balanced)  
**Version**: 1.0  
**Last Updated**: December 2024

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technical Specifications](#technical-specifications)
3. [Implementation Guide](#implementation-guide)
4. [Code Explanations](#code-explanations)
5. [Setup & Installation](#setup--installation)
6. [Usage Guide](#usage-guide)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## 1. Project Overview

### 1.1 Objective

Build a state-of-the-art text classification system using DeBERTa models with enterprise-grade data practices, training pipeline, and model selection framework.

### 1.2 Task Specifications

- **Task Type**: Multi-class topic categorization
- **Number of Classes**: 5
- **Dataset Size**: 20,000 samples
- **Class Distribution**: Balanced (4,000 samples per class)
- **Data Split**: 70% train / 15% validation / 15% test
- **Target Environment**: GPU-accelerated training with future production deployment

### 1.3 Expected Performance

| Model Variant | Expected Accuracy | Expected Macro F1 |
|---------------|-------------------|-------------------|
| DeBERTa-v3-base | 88-91% | 0.87-0.90 |
| DeBERTa-v3-large | 90-93% | 0.89-0.92 |
| Optimized + Ensemble | 92-95% | 0.91-0.94 |

**Success Criteria**:
- Minimum acceptable: 88% accuracy / 0.87 macro F1
- Target: 91% accuracy / 0.90 macro F1
- Stretch goal: 93% accuracy / 0.92 macro F1

---

## 2. Technical Specifications

### 2.1 Model Architecture

**Primary Model**: DeBERTa-v3 (Decoding-enhanced BERT with disentangled attention)

**Recommended Variants**:
- `microsoft/deberta-v3-large` (304M params) - Best accuracy/speed tradeoff
- `microsoft/deberta-v3-base` (86M params) - Faster training, good baseline
- `microsoft/deberta-v3-xlarge` (710M params) - Maximum accuracy if compute allows

**Why DeBERTa-v3**:
- Enhanced mask decoder with position-aware encoding
- Disentangled attention mechanism (separate content and position embeddings)
- Superior performance on GLUE, SuperGLUE benchmarks
- Better gradient flow than BERT/RoBERTa
- Efficient tokenization with SentencePiece

### 2.2 Data Strategy

#### 2.2.1 Data Quality Assessment

**Pre-processing Audit**:
- Check label distribution (verify balance)
- Detect duplicate texts (exact and near-duplicates)
- Identify outliers (text length, special characters, encoding issues)
- Validate label consistency

**Recommended Tools**:
- `ydata-profiling` - Data profiling and EDA
- `cleanlab` - Find label errors using confident learning
- `textdescriptives` - Compute linguistic features

#### 2.2.2 Data Cleaning Pipeline

**Step-by-step Cleaning Process**:

1. **Text Normalization**:
   - Unicode normalization (NFKC)
   - Remove/replace invisible characters
   - Standardize whitespace
   - Handle HTML entities and markup
   - URL/email handling (replace with tokens or remove)

2. **Deduplication**:
   - Remove exact duplicates
   - Flag near-duplicates (>95% similarity)
   - Ensure train/val/test independence

3. **Label Validation**:
   - Run label error detection (cleanlab)
   - Manual review of low-confidence predictions
   - Ensure label schema consistency

4. **Quality Filters**:
   - Remove texts below minimum length (e.g., <3 tokens)
   - Handle extremely long texts (>512 tokens)
   - Remove corrupted encodings

#### 2.2.3 Data Augmentation (Optional)

**Techniques to Boost Robustness**:

- **Back-translation**: Translate to another language and back
- **Synonym replacement**: Replace words with contextual synonyms
- **Random insertion/deletion**: Strategic word manipulation
- **Paraphrasing**: Use T5/GPT models for semantic-preserving rewrites
- **Mixup**: Interpolate embeddings between examples

**Important**: Apply augmentation only to training set, not validation/test.

#### 2.2.4 Data Splitting Strategy

**Recommended Splits** (for 20k samples):
```
Training:   14,000 samples (70%) - 2,800 per class
Validation:  3,000 samples (15%) - 600 per class
Test:        3,000 samples (15%) - 600 per class
```

**Stratification**: Use stratified split to maintain class distribution across all sets.

### 2.3 Training Configuration

#### 2.3.1 Optimizer Configuration

**AdamW** with decoupled weight decay:
```python
optimizer = AdamW(
    params=model.parameters(),
    lr=3e-5,           # typical range: 1e-5 to 5e-5
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # L2 regularization
)
```

#### 2.3.2 Learning Rate Schedule

**Linear warmup + linear decay**:
- Warmup steps: 10% of total training steps
- Helps stabilize early training
- Gradual decay prevents overfitting

```python
num_warmup_steps = 0.1 * total_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps
)
```

#### 2.3.3 Training Hyperparameters

**Optimized for 20k samples, 5 classes**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 32 | Balance speed/memory |
| Gradient accumulation | 1 | Effective batch = 32 |
| Max sequence length | 256 | Start here, increase to 512 if needed |
| Number of epochs | 5 | ~2,185 total training steps |
| Mixed precision (fp16) | True | 2x speedup on modern GPUs |
| Gradient checkpointing | True | Reduces memory usage |
| Max gradient norm | 1.0 | Prevents exploding gradients |

**Training Steps Calculation**:
```
Steps per epoch = 14,000 samples / 32 batch size = 437 steps
Total steps = 437 steps × 5 epochs = 2,185 steps
Warmup steps = 2,185 × 0.1 = 218 steps
```

#### 2.3.4 Regularization Techniques

- **Dropout**: 0.1 (default in DeBERTa)
- **Weight decay**: 0.01 (L2 regularization)
- **Gradient clipping**: max_grad_norm = 1.0
- **Early stopping**: Patience = 3 evaluations

#### 2.3.5 Advanced Training Techniques

**1. Gradual Unfreezing**:
```python
# Epoch 1: Freeze all transformer layers, train only classifier
# Epoch 2: Unfreeze top 6 layers
# Epoch 3+: Unfreeze all layers
```

**2. Stochastic Weight Averaging (SWA)**:
- Average model weights over last few epochs
- Often improves generalization by 0.5-1%

**3. Early Stopping**:
```python
patience = 3
monitor = 'eval_f1_macro'
```

### 2.4 Evaluation Metrics

**Classification Metrics to Track**:

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness |
| Macro F1 | Average F1 across classes (handles imbalance) |
| Weighted F1 | F1 weighted by class support |
| Per-class F1 | Individual class performance |
| Confusion Matrix | Error pattern analysis |
| ROC-AUC | Probability calibration |

**Primary Metric**: Macro F1 (best for balanced multi-class)

### 2.5 Model Selection & Hyperparameter Optimization

#### 2.5.1 Hyperparameter Search Strategy

**Use Optuna for Bayesian Optimization**

**Parameters to Tune** (prioritized for 20k dataset):

| Parameter | Search Range | Priority |
|-----------|--------------|----------|
| Learning rate | [2e-5, 3e-5, 5e-5] | Critical |
| Epochs | [4, 5, 6] | Critical |
| Batch size | [16, 32] | Important |
| Warmup ratio | [0.1, 0.2] | Important |
| Weight decay | [0.01, 0.05] | Moderate |
| Max length | [256, 384] | Moderate |

**Search Budget**: 15-25 trials (sufficient for 20k dataset)

**Expected Training Time per Trial**: 30-60 minutes on single A100

#### 2.5.2 Cross-Validation Strategy

**5-Fold Stratified Cross-Validation**:
- Use stratified k-fold to maintain class distribution
- Average metrics across folds
- Select hyperparameters based on mean ± std dev

**Nested CV** (optional for robust estimates):
- Outer loop: Model evaluation
- Inner loop: Hyperparameter selection

#### 2.5.3 Ensemble Methods

**Model Ensemble for Production**:

1. Train 3-5 models with different:
   - Random seeds
   - Hyperparameters
   - Model sizes (base, large)

2. Combine predictions:
   - **Soft voting**: Average class probabilities
   - **Weighted voting**: Weight by validation performance
   - **Stacking**: Train meta-classifier on model outputs

**Expected Gain**: 1-3% improvement over single best model

### 2.6 Experiment Tracking & Reproducibility

#### 2.6.1 Logging Requirements

**Track for Every Experiment**:
- All hyperparameters
- Train/val/test metrics per epoch
- Model checkpoints (best + last)
- Data version/hash
- Random seeds (Python, NumPy, PyTorch)
- Hardware specs (GPU type, CUDA version)

**Recommended Tools**: Weights & Biases (wandb) or MLflow

#### 2.6.2 Reproducibility Checklist

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

#### 2.6.3 Model Versioning

- Semantic versioning: `deberta-v3-large-v1.0.0`
- Tag with metadata: dataset version, metrics, date
- Store in model registry (Hugging Face Hub, MLflow, cloud storage)

### 2.7 Validation & Testing Protocol

#### 2.7.1 Validation During Training

- Evaluate on validation set every 100 steps
- Save checkpoint if validation metric improves
- Use `evaluate()` without gradient computation

#### 2.7.2 Final Model Evaluation

**Test Set Evaluation** (run once on final model):
- Report all metrics with confidence intervals
- Generate confusion matrix
- Error analysis: Inspect misclassifications
- Adversarial testing: Test on challenging examples

#### 2.7.3 Robustness Testing

**Test Model Robustness**:
- Text perturbations (typos, synonym replacement)
- Out-of-distribution samples
- Adversarial examples (TextFooler, BERT-Attack)
- Fairness metrics (if applicable)

### 2.8 Infrastructure & Tooling

#### 2.8.1 Recommended Stack

**Compute**:
- NVIDIA A100 (40GB/80GB) or V100 (32GB)
- PyTorch + Hugging Face Transformers
- Accelerate or DeepSpeed for multi-GPU

**Data**:
- Storage: Cloud storage (S3, GCS) or shared filesystem
- Processing: Datasets library (memory-mapped)
- Versioning: DVC or Pachyderm

**Experiment Tracking**:
- Weights & Biases or MLflow

**Orchestration**:
- Training jobs: Kubernetes with Kubeflow or Ray
- Workflow: Airflow or Prefect

#### 2.8.2 Code Structure

```
project/
├── data/
│   ├── raw/              # Original data
│   ├── processed/        # Cleaned data
│   └── augmented/        # Augmented data (optional)
├── src/
│   ├── data_processing.py  # Cleaning & preprocessing
│   ├── model.py           # Model definitions
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation utilities
│   └── utils.py           # Helper functions
├── configs/
│   └── training_config.yaml  # Configuration files
├── notebooks/
│   └── eda.ipynb          # Exploratory analysis
├── scripts/
│   ├── prepare_data.sh    # Data preparation
│   └── run_training.sh    # Training launcher
├── tests/                 # Unit tests
├── results/               # Training outputs
├── models/                # Saved models
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── .gitignore
```

---

## 3. Implementation Guide

### 3.1 Complete Training Pipeline

See `train.py` in the repository for the full implementation. Key components:

#### 3.1.1 Configuration Management

```python
@dataclass
class Config:
    """Central configuration for training pipeline."""
    # Model
    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = 5
    
    # Data
    data_path: str = "data/labeled_data.csv"
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 256
    
    # Training
    output_dir: str = "./results"
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Optimization
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    metric_for_best_model: str = "eval_f1_macro"
    
    # Reproducibility
    seed: int = 42
```

#### 3.1.2 Reproducibility Function

```python
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

### 3.2 Data Processing Pipeline

#### 3.2.1 DataCleaner Class

The `DataCleaner` class provides comprehensive text preprocessing:

```python
class DataCleaner:
    """Comprehensive text cleaning for topic classification."""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters to standard form."""
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace, tabs, newlines."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def handle_urls_emails(text: str, replace: bool = True) -> str:
        """Replace or remove URLs and emails."""
        if replace:
            text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
            text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        else:
            text = re.sub(r'http\S+|www\.\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
        return text
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags and entities."""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&\w+;', '', text)
        return text
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Apply all cleaning steps."""
        if not isinstance(text, str):
            return ""
        
        text = DataCleaner.normalize_unicode(text)
        text = DataCleaner.remove_html_tags(text)
        text = DataCleaner.handle_urls_emails(text, replace=True)
        text = DataCleaner.remove_extra_whitespace(text)
        
        # Minimum length filter
        if len(text.split()) < 3:
            return ""
        
        return text
```

#### 3.2.2 Data Loading & Preparation

```python
def load_and_prepare_data(config: Config):
    """
    Load data, perform cleaning, splitting, and quality checks.
    """
    # Load data
    if config.data_path.endswith('.csv'):
        df = pd.read_csv(config.data_path)
    elif config.data_path.endswith(('.json', '.jsonl')):
        df = pd.read_json(config.data_path, 
                         lines=config.data_path.endswith('.jsonl'))
    
    # Clean text
    df['text_cleaned'] = df[config.text_column].apply(
        DataCleaner.clean_text
    )
    
    # Remove empty texts
    df = df[df['text_cleaned'].str.len() > 0].reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text_cleaned']).reset_index(drop=True)
    
    # Stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=config.seed,
        stratify=df[config.label_column]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=config.seed,
        stratify=temp_df[config.label_column]
    )
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
```

### 3.3 Model Training

#### 3.3.1 Metrics Computation

```python
def compute_metrics(eval_pred):
    """Compute comprehensive metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Overall metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }
    
    # Add per-class F1 scores
    for i, f1_class in enumerate(f1):
        metrics[f'f1_class_{i}'] = f1_class
    
    return metrics
```

#### 3.3.2 Training Function

```python
def train_model(config: Config):
    """Main training function."""
    
    # Set seed
    set_seed(config.seed)
    
    # Load and prepare data
    data_dict = load_and_prepare_data(config)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )
    
    # Prepare datasets
    tokenized_datasets = prepare_datasets(data_dict, tokenizer, config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience
        )]
    )
    
    # Train
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_datasets['test'])
    
    # Save final model
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    
    return trainer, test_results
```

---

## 4. Code Explanations

### 4.1 DataCleaner Class Deep Dive

#### Purpose
The `DataCleaner` class is a collection of text preprocessing methods designed to clean and normalize raw text data before feeding it to DeBERTa.

#### Why `@staticmethod`?
Static methods don't require an instance of the class. They're pure utility functions:

```python
# Direct call - no instance needed
clean_text = DataCleaner.clean_text("some messy text")

# Instead of:
# cleaner = DataCleaner()  ← NOT needed
# clean_text = cleaner.clean_text("some messy text")
```

This keeps the code clean and efficient since cleaning methods don't need to store state.

#### Method Breakdown

**1. `normalize_unicode(text)`**

```python
return unicodedata.normalize('NFKC', text)
```

**Problem**: Different ways to represent the same character
- "café" can be `café` (single é) or `café` (e + combining accent)
- URLs might have `%20` or weird encodings

**Solution**: NFKC normalization
- **N**ormalization **F**orm **K**ompatibility **C**omposition
- Converts all variations to single standard form
- Examples: `①` → `1`, `ﬁ` (ligature) → `fi`

```python
# Before: "hello①②③"
# After:  "hello123"
```

**2. `remove_extra_whitespace(text)`**

```python
text = re.sub(r'\s+', ' ', text)  # Any whitespace → single space
return text.strip()  # Remove leading/trailing
```

**Problem**: Inconsistent spacing breaks tokenization

```python
# Before: "hello    world\n\ntab\there"
# After:  "hello world tab here"
```

**3. `handle_urls_emails(text, replace=True)`**

```python
text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
text = re.sub(r'\S+@\S+', '[EMAIL]', text)
```

**Problem**: URLs/emails don't help topic classification and add noise

**Two Strategies**:
- **Replace** (default): `[URL]` and `[EMAIL]` tokens preserve that info existed
- **Remove**: Completely delete them

```python
# Replace mode:
# Before: "Check out https://example.com for info"
# After:  "Check out [URL] for info"

# Remove mode:
# Before: "Email me at john@example.com"
# After:  "Email me at"
```

**Why replace?** Context preservation:
- "Visit [URL] for more" ← meaningful
- "Visit for more" ← awkward

**4. `remove_html_tags(text)`**

```python
text = re.sub(r'<[^>]+>', '', text)  # Remove tags
text = re.sub(r'&\w+;', '', text)    # Remove entities
```

**Problem**: Web-scraped data often has HTML markup

```python
# Before: "<p>Hello <b>world</b></p>"
# After:  "Hello world"

# Before: "5 &lt; 10 &amp; 10 &gt; 5"
# After:  "5 10 10 5"
```

**5. `clean_text(text)` - Master Method**

Orchestrates all cleaning in optimal order:

```python
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""  # Handle nulls
    
    # Order matters!
    text = DataCleaner.normalize_unicode(text)
    text = DataCleaner.remove_html_tags(text)
    text = DataCleaner.handle_urls_emails(text, replace=True)
    text = DataCleaner.remove_extra_whitespace(text)
    
    # Quality filter
    if len(text.split()) < 3:
        return ""  # Flag too-short texts
    
    return text
```

**Order Rationale**:
1. Normalize unicode first (fix encoding)
2. Remove HTML (clean structure)
3. Handle URLs/emails (replace noise)
4. Clean whitespace (final polish)
5. Filter short texts (quality control)

**Real Example**:
```python
raw = """
  <div>Check out http://example.com! 
  Contact: admin@site.com    
  Price: ①⓪⓪ USD&nbsp;&nbsp;</div>
"""

clean = DataCleaner.clean_text(raw)
# Result: "Check out [URL]! Contact: [EMAIL] Price: 100 USD"
```

#### Why This Matters

1. **Consistent tokenization**: Clean text → predictable tokens
2. **Better embeddings**: DeBERTa focuses on meaningful words
3. **Reduced vocabulary**: `[URL]` instead of millions of unique URLs
4. **Improved accuracy**: Less noise = clearer signal

### 4.2 Training Configuration Explained

#### Config Dataclass

```python
@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = 5
    # ... more parameters
```

**Why dataclass?**
- Automatic `__init__`, `__repr__`, `__eq__`
- Type hints for IDE autocomplete
- Easy to serialize/deserialize
- Clean parameter management

**Key Parameters Explained**:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `max_length` | 256 | Truncate/pad sequences to this length. Shorter = faster, longer = more context |
| `batch_size` | 32 | Number of samples per gradient update. Larger = more stable but more memory |
| `learning_rate` | 3e-5 | Step size for weight updates. Too high = unstable, too low = slow |
| `weight_decay` | 0.01 | L2 regularization strength. Prevents overfitting |
| `warmup_ratio` | 0.1 | Percentage of steps for LR warmup. Stabilizes early training |
| `fp16` | True | Use 16-bit precision. 2x speedup, minimal accuracy loss |
| `eval_steps` | 100 | Evaluate every N steps. Balance speed vs monitoring |
| `early_stopping_patience` | 3 | Stop if no improvement for N evaluations |

### 4.3 Metrics Computation Explained

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
```

**What's happening**:
- `logits`: Raw model outputs (5 values per sample, not probabilities)
- `np.argmax`: Convert to class predictions (0-4)
- Example: `[0.1, -0.3, 2.5, 0.8, -1.2]` → class 2

**Metrics Breakdown**:

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | Correct / Total | Overall performance |
| Macro F1 | Average F1 per class | Balanced datasets, equal class importance |
| Weighted F1 | F1 weighted by support | Imbalanced datasets |
| Per-class F1 | F1 for each class | Identify weak classes |

**Why Macro F1 for us?**
- Dataset is balanced (4,000 per class)
- Each class equally important
- Penalizes poor performance on any class

### 4.4 Training Loop Explained

The `Trainer` class from Hugging Face handles the training loop automatically:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    compute_metrics=compute_metrics,
)

trainer.train()
```

**What happens under the hood**:

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Evaluation (every eval_steps)
        if step % eval_steps == 0:
            metrics = evaluate(val_dataset)
            if metrics[metric_for_best_model] > best_metric:
                save_model()
```

**Key Components**:
1. **Forward pass**: Compute predictions and loss
2. **Backward pass**: Compute gradients
3. **Gradient clipping**: Prevent exploding gradients
4. **Optimizer step**: Update weights
5. **LR scheduling