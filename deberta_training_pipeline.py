# ============================================================================
# DeBERTa Text Classification - Complete Training Pipeline
# ============================================================================

import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset as HFDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import wandb
from collections import Counter
import re
import unicodedata

# ============================================================================
# 1. REPRODUCIBILITY & CONFIGURATION
# ============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class Config:
    """Central configuration for training pipeline."""
    # Model
    model_name: str = "microsoft/deberta-v3-large"  # or deberta-v3-base
    num_labels: int = 5
    
    # Data
    data_path: str = "data/labeled_data.csv"  # or .json, .jsonl
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 256  # increase to 384/512 if needed
    
    # Training
    output_dir: str = "./results"
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Optimization
    fp16: bool = True  # use bf16 if you have Ampere GPUs (A100)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1_macro"
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    logging_steps: int = 50
    report_to: str = "wandb"  # or "tensorboard" or "none"
    run_name: Optional[str] = None


# ============================================================================
# 2. DATA PREPROCESSING & CLEANING
# ============================================================================

class DataCleaner:
    """Comprehensive text cleaning for topic classification."""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
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


def load_and_prepare_data(config: Config):
    """
    Load data, perform cleaning, splitting, and quality checks.
    
    Expected input format (CSV/JSON):
    - text column: raw text
    - label column: integer labels (0-4) or string labels
    """
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)
    
    # Load data
    if config.data_path.endswith('.csv'):
        df = pd.read_csv(config.data_path)
    elif config.data_path.endswith(('.json', '.jsonl')):
        df = pd.read_json(config.data_path, lines=config.data_path.endswith('.jsonl'))
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON/JSONL.")
    
    print(f"\n📊 Original dataset size: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic validation
    assert config.text_column in df.columns, f"Text column '{config.text_column}' not found"
    assert config.label_column in df.columns, f"Label column '{config.label_column}' not found"
    
    # Convert labels to integers if needed
    if df[config.label_column].dtype == 'object':
        label_map = {label: idx for idx, label in enumerate(df[config.label_column].unique())}
        df['label_int'] = df[config.label_column].map(label_map)
        print(f"\n🏷️  Label mapping: {label_map}")
        config.label_column = 'label_int'
    
    # Clean text
    print("\n🧹 Cleaning text...")
    df['text_cleaned'] = df[config.text_column].apply(DataCleaner.clean_text)
    
    # Remove empty texts
    original_len = len(df)
    df = df[df['text_cleaned'].str.len() > 0].reset_index(drop=True)
    print(f"Removed {original_len - len(df)} empty texts after cleaning")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['text_cleaned']).sum()
    if duplicates > 0:
        print(f"⚠️  Found {duplicates} duplicate texts - removing...")
        df = df.drop_duplicates(subset=['text_cleaned']).reset_index(drop=True)
    
    # Class distribution
    print("\n📈 Class distribution:")
    class_counts = df[config.label_column].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Text length statistics
    df['text_length'] = df['text_cleaned'].str.split().str.len()
    print(f"\n📏 Text length statistics:")
    print(f"  Mean: {df['text_length'].mean():.1f} tokens")
    print(f"  Median: {df['text_length'].median():.1f} tokens")
    print(f"  Min: {df['text_length'].min()} tokens")
    print(f"  Max: {df['text_length'].max()} tokens")
    print(f"  95th percentile: {df['text_length'].quantile(0.95):.1f} tokens")
    
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
    
    print(f"\n✂️  Data split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return {
        'train': train_df[['text_cleaned', config.label_column]].rename(columns={'text_cleaned': 'text', config.label_column: 'label'}),
        'val': val_df[['text_cleaned', config.label_column]].rename(columns={'text_cleaned': 'text', config.label_column: 'label'}),
        'test': test_df[['text_cleaned', config.label_column]].rename(columns={'text_cleaned': 'text', config.label_column: 'label'})
    }


# ============================================================================
# 3. DATASET PREPARATION
# ============================================================================

def prepare_datasets(data_dict: Dict, tokenizer, config: Config):
    """Tokenize datasets for training."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.max_length
        )
    
    # Convert to HuggingFace datasets
    datasets = {
        'train': HFDataset.from_pandas(data_dict['train']),
        'val': HFDataset.from_pandas(data_dict['val']),
        'test': HFDataset.from_pandas(data_dict['test'])
    }
    
    # Tokenize
    print("\n🔤 Tokenizing datasets...")
    tokenized_datasets = {
        split: ds.map(tokenize_function, batched=True, remove_columns=['text'])
        for split, ds in datasets.items()
    }
    
    # Set format for PyTorch
    for ds in tokenized_datasets.values():
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_datasets


# ============================================================================
# 4. METRICS COMPUTATION
# ============================================================================

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


# ============================================================================
# 5. TRAINING FUNCTION
# ============================================================================

def train_model(config: Config):
    """Main training function."""
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize wandb
    if config.report_to == "wandb":
        wandb.init(
            project="deberta-topic-classification",
            name=config.run_name or f"deberta-lr{config.learning_rate}-bs{config.batch_size}",
            config=config.__dict__
        )
    
    # Load and prepare data
    data_dict = load_and_prepare_data(config)
    
    # Load tokenizer and model
    print("\n🤖 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )
    
    print(f"Model: {config.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
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
        
        # Optimization
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        
        # Evaluation
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        
        # Logging
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        
        # Other
        seed=config.seed,
        dataloader_num_workers=4,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print("\nTest Set Results:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            print(f"  {key[5:]}: {value:.4f}")
    
    # Detailed classification report
    predictions = trainer.predict(tokenized_datasets['test'])
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    print("\n📊 Detailed Classification Report:")
    print(classification_report(true_labels, pred_labels, digits=4))
    
    print("\n🔀 Confusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)
    
    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")
    
    if config.report_to == "wandb":
        wandb.finish()
    
    return trainer, test_results


# ============================================================================
# 6. HYPERPARAMETER SEARCH (OPTUNA)
# ============================================================================

def hyperparameter_search():
    """
    Run hyperparameter optimization with Optuna.
    Uncomment and customize as needed.
    """
    import optuna
    
    def objective(trial):
        config = Config(
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
            batch_size=trial.suggest_categorical('batch_size', [16, 32]),
            num_epochs=trial.suggest_int('num_epochs', 4, 6),
            warmup_ratio=trial.suggest_float('warmup_ratio', 0.1, 0.2),
            weight_decay=trial.suggest_float('weight_decay', 0.01, 0.05),
            run_name=f"trial_{trial.number}"
        )
        
        trainer, results = train_model(config)
        return results['eval_f1_macro']
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    print(study.best_params)
    print(f"Best F1 Macro: {study.best_value:.4f}")
    
    return study


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Basic configuration
    config = Config(
        model_name="microsoft/deberta-v3-large",
        data_path="data/labeled_data.csv",  # UPDATE THIS PATH
        text_column="text",  # UPDATE IF DIFFERENT
        label_column="label",  # UPDATE IF DIFFERENT
        num_labels=5,
        
        # Training hyperparameters
        num_epochs=5,
        batch_size=32,
        learning_rate=3e-5,
        max_length=256,
        
        # Experiment tracking
        output_dir="./results/deberta-v3-large-run1",
        run_name="deberta-v3-large-baseline",
        report_to="wandb",  # change to "none" if not using wandb
    )
    
    # Run training
    trainer, results = train_model(config)
    
    # Uncomment to run hyperparameter search instead:
    # study = hyperparameter_search()
