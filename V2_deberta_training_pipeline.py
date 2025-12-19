"""
=================================================================================
DeBERTa Text Classification - Complete Implementation
=================================================================================
Project: State-of-the-art 5-class Topic Classification
Dataset: 20,000 labeled samples (balanced)
Model: DeBERTa-v3 (Base/Large)
Version: 1.0

Author: [Your Name]
Date: December 2024

Usage:
    python train.py                    # Run with default config
    python train.py --hyperparam       # Run hyperparameter search
=================================================================================
"""

import os
import sys
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import unicodedata
import re

# Transformers and datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed as hf_set_seed,
)
from datasets import Dataset as HFDataset, DatasetDict

# Sklearn utilities
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not installed. Install with: pip install wandb")

# Hyperparameter optimization
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not installed. Install with: pip install optuna")

warnings.filterwarnings('ignore')


# =============================================================================
# 1. REPRODUCIBILITY & CONFIGURATION
# =============================================================================

def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    hf_set_seed(seed)
    print(f"✓ Random seed set to {seed}")


@dataclass
class Config:
    """
    Central configuration for training pipeline.
    All hyperparameters and settings in one place.
    """
    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    model_name: str = "microsoft/deberta-v3-large"  # or deberta-v3-base
    num_labels: int = 5
    dropout: float = 0.1  # Default dropout rate
    
    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    data_path: str = "data/labeled_data.csv"  # Path to your data file
    text_column: str = "text"  # Column name containing text
    label_column: str = "label"  # Column name containing labels
    max_length: int = 256  # Max sequence length (256, 384, or 512)
    
    # Data split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data cleaning options
    replace_urls: bool = True  # Replace URLs with [URL] token
    replace_emails: bool = True  # Replace emails with [EMAIL] token
    min_text_length: int = 3  # Minimum number of tokens
    remove_duplicates: bool = True
    
    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    output_dir: str = "./results"
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # -------------------------------------------------------------------------
    # Optimization Configuration
    # -------------------------------------------------------------------------
    fp16: bool = True  # Mixed precision training (use bf16 for A100)
    bf16: bool = False  # Brain float 16 (for Ampere GPUs)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False  # Reduce memory usage
    
    # -------------------------------------------------------------------------
    # Evaluation Configuration
    # -------------------------------------------------------------------------
    eval_strategy: str = "steps"  # or "epoch"
    eval_steps: int = 100  # Evaluate every N steps
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3  # Keep only best N checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1_macro"
    greater_is_better: bool = True
    
    # -------------------------------------------------------------------------
    # Early Stopping Configuration
    # -------------------------------------------------------------------------
    early_stopping_patience: int = 3
    
    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    logging_dir: str = "./logs"
    logging_steps: int = 50
    report_to: str = "wandb"  # "wandb", "tensorboard", or "none"
    run_name: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # Advanced Training Techniques
    # -------------------------------------------------------------------------
    use_swa: bool = False  # Stochastic Weight Averaging
    swa_start_epoch: int = 3  # Start SWA from this epoch
    
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    seed: int = 42
    
    # -------------------------------------------------------------------------
    # Hyperparameter Search (Optuna)
    # -------------------------------------------------------------------------
    n_trials: int = 20  # Number of hyperparameter search trials
    
    # -------------------------------------------------------------------------
    # Device Configuration
    # -------------------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_num_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "Data split ratios must sum to 1.0"
        assert self.max_length in [128, 256, 384, 512], \
            "max_length should be one of: 128, 256, 384, 512"
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        
        # Set run name if not provided
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = self.model_name.split('/')[-1]
            self.run_name = f"{model_short}_lr{self.learning_rate}_bs{self.batch_size}_{timestamp}"
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"✓ Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# =============================================================================
# 2. DATA PREPROCESSING & CLEANING
# =============================================================================

class DataCleaner:
    """
    Comprehensive text cleaning for topic classification.
    All methods are static for easy use without instantiation.
    """
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode characters to standard form.
        
        Uses NFKC normalization:
        - Compatibility decomposition followed by canonical composition
        - Converts ligatures, special characters to standard forms
        
        Examples:
            "①②③" -> "123"
            "ﬁ" (ligature) -> "fi"
            "café" (any form) -> "café" (standard)
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        Remove extra whitespace, tabs, and newlines.
        
        Replaces any sequence of whitespace characters with a single space,
        then strips leading/trailing whitespace.
        
        Examples:
            "hello    world\n\ntab\there" -> "hello world tab here"
            "  spaces  " -> "spaces"
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized whitespace
        """
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def handle_urls_emails(text: str, replace: bool = True) -> str:
        """
        Replace or remove URLs and email addresses.
        
        Args:
            text (str): Input text
            replace (bool): If True, replace with tokens; if False, remove entirely
            
        Returns:
            str: Text with URLs/emails handled
            
        Examples:
            Replace mode:
                "Visit https://example.com" -> "Visit [URL]"
                "Email me@example.com" -> "Email [EMAIL]"
            Remove mode:
                "Visit https://example.com" -> "Visit"
        """
        if replace:
            text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
            text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        else:
            text = re.sub(r'http\S+|www\.\S+', '', text)
            text = re.sub(r'\S+@\S+', '', text)
        return text
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        Remove HTML tags and entities from text.
        
        Useful for web-scraped data or data from HTML sources.
        
        Examples:
            "<p>Hello <b>world</b></p>" -> "Hello world"
            "5 &lt; 10 &amp; text" -> "5  10  text"
        
        Args:
            text (str): Input text potentially containing HTML
            
        Returns:
            str: Text with HTML removed
        """
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'&\w+;', '', text)    # Remove HTML entities
        return text
    
    @staticmethod
    def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters while optionally keeping punctuation.
        
        Args:
            text (str): Input text
            keep_punctuation (bool): Whether to keep standard punctuation
            
        Returns:
            str: Cleaned text
        """
        if keep_punctuation:
            # Keep alphanumeric, spaces, and standard punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"]', '', text)
        else:
            # Keep only alphanumeric and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    @staticmethod
    def clean_text(text: str, 
                   min_length: int = 3,
                   replace_urls: bool = True,
                   replace_emails: bool = True) -> str:
        """
        Apply all cleaning steps in optimal order.
        
        Order of operations:
        1. Handle non-string input
        2. Normalize unicode (fix encoding issues)
        3. Remove HTML (clean structure)
        4. Handle URLs/emails (reduce noise)
        5. Remove extra whitespace (final polish)
        6. Length filter (quality control)
        
        Args:
            text (str): Input text
            min_length (int): Minimum number of tokens required
            replace_urls (bool): Replace URLs with [URL] token
            replace_emails (bool): Replace emails with [EMAIL] token
            
        Returns:
            str: Cleaned text, or empty string if below quality threshold
            
        Example:
            Input:
                "<div>Check out http://example.com! 
                Contact: admin@site.com    
                Price: ①⓪⓪ USD&nbsp;&nbsp;</div>"
            Output:
                "Check out [URL]! Contact: [EMAIL] Price: 100 USD"
        """
        # Handle missing or non-string input
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Apply cleaning steps in order
        text = DataCleaner.normalize_unicode(text)
        text = DataCleaner.remove_html_tags(text)
        text = DataCleaner.handle_urls_emails(text, replace=replace_urls or replace_emails)
        text = DataCleaner.remove_extra_whitespace(text)
        
        # Quality filter: remove very short texts
        if len(text.split()) < min_length:
            return ""
        
        return text


class DataProcessor:
    """
    Handle data loading, cleaning, splitting, and analysis.
    """
    
    def __init__(self, config: Config):
        """
        Initialize data processor with configuration.
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.df = None
        self.splits = {}
        self.stats = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file (CSV, JSON, or JSONL).
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        data_path = Path(self.config.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load based on file extension
        if data_path.suffix == '.csv':
            self.df = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            self.df = pd.read_json(data_path)
        elif data_path.suffix == '.jsonl':
            self.df = pd.read_json(data_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        print(f"✓ Loaded {len(self.df)} samples from {data_path}")
        print(f"✓ Columns: {self.df.columns.tolist()}")
        
        return self.df
    
    def validate_data(self):
        """Validate that required columns exist and data is properly formatted."""
        assert self.config.text_column in self.df.columns, \
            f"Text column '{self.config.text_column}' not found in data"
        assert self.config.label_column in self.df.columns, \
            f"Label column '{self.config.label_column}' not found in data"
        
        # Convert labels to integers if they're strings
        if self.df[self.config.label_column].dtype == 'object':
            unique_labels = sorted(self.df[self.config.label_column].unique())
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.df['label_int'] = self.df[self.config.label_column].map(label_map)
            
            print(f"\n✓ Converted string labels to integers:")
            for label, idx in label_map.items():
                print(f"  {label} -> {idx}")
            
            # Save label mapping
            label_map_path = Path(self.config.output_dir) / 'label_mapping.json'
            with open(label_map_path, 'w') as f:
                json.dump(label_map, f, indent=2)
            print(f"✓ Label mapping saved to {label_map_path}")
            
            self.config.label_column = 'label_int'
    
    def analyze_data(self):
        """Perform exploratory data analysis and print statistics."""
        print("\n" + "=" * 80)
        print("DATA ANALYSIS")
        print("=" * 80)
        
        # Basic statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Unique texts: {self.df[self.config.text_column].nunique()}")
        
        # Class distribution
        print(f"\n📈 Class Distribution:")
        class_counts = self.df[self.config.label_column].value_counts().sort_index()
        for label, count in class_counts.items():
            percentage = count / len(self.df) * 100
            print(f"  Class {label}: {count:,} samples ({percentage:.1f}%)")
        
        # Check for imbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 1.5:
            print(f"\n⚠️  Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print("  Consider using class weights or resampling techniques.")
        
        # Text length statistics
        self.df['text_length'] = self.df[self.config.text_column].str.split().str.len()
        
        print(f"\n📏 Text Length Statistics (in tokens):")
        print(f"  Mean: {self.df['text_length'].mean():.1f}")
        print(f"  Median: {self.df['text_length'].median():.1f}")
        print(f"  Min: {self.df['text_length'].min()}")
        print(f"  Max: {self.df['text_length'].max()}")
        print(f"  25th percentile: {self.df['text_length'].quantile(0.25):.1f}")
        print(f"  75th percentile: {self.df['text_length'].quantile(0.75):.1f}")
        print(f"  95th percentile: {self.df['text_length'].quantile(0.95):.1f}")
        
        # Check if max_length is appropriate
        pct_truncated = (self.df['text_length'] > self.config.max_length).mean() * 100
        if pct_truncated > 10:
            print(f"\n⚠️  Warning: {pct_truncated:.1f}% of texts exceed max_length={self.config.max_length}")
            print(f"  Consider increasing max_length to {int(self.df['text_length'].quantile(0.95))}")
        
        # Store statistics
        self.stats = {
            'total_samples': len(self.df),
            'num_classes': self.df[self.config.label_column].nunique(),
            'class_distribution': class_counts.to_dict(),
            'text_length_mean': float(self.df['text_length'].mean()),
            'text_length_median': float(self.df['text_length'].median()),
            'text_length_max': int(self.df['text_length'].max()),
        }
    
    def clean_data(self):
        """Apply comprehensive data cleaning."""
        print("\n" + "=" * 80)
        print("CLEANING DATA")
        print("=" * 80)
        
        original_len = len(self.df)
        
        # Apply text cleaning
        print("\n🧹 Applying text cleaning...")
        self.df['text_cleaned'] = self.df[self.config.text_column].apply(
            lambda x: DataCleaner.clean_text(
                x,
                min_length=self.config.min_text_length,
                replace_urls=self.config.replace_urls,
                replace_emails=self.config.replace_emails
            )
        )
        
        # Remove empty texts
        empty_mask = self.df['text_cleaned'].str.len() == 0
        n_empty = empty_mask.sum()
        if n_empty > 0:
            print(f"  Removed {n_empty} empty texts after cleaning")
            self.df = self.df[~empty_mask].reset_index(drop=True)
        
        # Check for duplicates
        if self.config.remove_duplicates:
            duplicates = self.df.duplicated(subset=['text_cleaned']).sum()
            if duplicates > 0:
                print(f"  Removed {duplicates} duplicate texts")
                self.df = self.df.drop_duplicates(
                    subset=['text_cleaned']
                ).reset_index(drop=True)
        
        # Report cleaning results
        cleaned_len = len(self.df)
        removed = original_len - cleaned_len
        print(f"\n✓ Cleaning complete:")
        print(f"  Original: {original_len:,} samples")
        print(f"  Cleaned: {cleaned_len:,} samples")
        print(f"  Removed: {removed:,} samples ({removed/original_len*100:.1f}%)")
    
    def split_data(self):
        """Split data into train, validation, and test sets with stratification."""
        print("\n" + "=" * 80)
        print("SPLITTING DATA")
        print("=" * 80)
        
        # Prepare final dataframe
        df_clean = self.df[[
            'text_cleaned', 
            self.config.label_column
        ]].rename(columns={
            'text_cleaned': 'text',
            self.config.label_column: 'label'
        })
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df_clean,
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=self.config.seed,
            stratify=df_clean['label']
        )
        
        # Second split: val vs test
        val_size = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=self.config.seed,
            stratify=temp_df['label']
        )
        
        # Store splits
        self.splits = {
            'train': train_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }
        
        # Print split statistics
        print(f"\n✓ Data split complete:")
        total = len(df_clean)
        for split_name, split_df in self.splits.items():
            n_samples = len(split_df)
            percentage = n_samples / total * 100
            print(f"  {split_name.capitalize():5s}: {n_samples:,} samples ({percentage:.1f}%)")
        
        # Verify class distribution in each split
        print(f"\n✓ Class distribution per split:")
        for split_name, split_df in self.splits.items():
            class_counts = split_df['label'].value_counts().sort_index()
            print(f"  {split_name.capitalize()}:")
            for label, count in class_counts.items():
                print(f"    Class {label}: {count} samples")
        
        # Save splits to disk
        splits_dir = Path(self.config.output_dir) / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        for split_name, split_df in self.splits.items():
            split_path = splits_dir / f'{split_name}.csv'
            split_df.to_csv(split_path, index=False)
        
        print(f"\n✓ Splits saved to {splits_dir}")
        
        return self.splits
    
    def prepare_datasets(self, tokenizer) -> DatasetDict:
        """
        Tokenize datasets for training.
        
        Args:
            tokenizer: HuggingFace tokenizer
            
        Returns:
            DatasetDict: Tokenized datasets ready for training
        """
        print("\n" + "=" * 80)
        print("TOKENIZING DATASETS")
        print("=" * 80)
        
        def tokenize_function(examples):
            """Tokenization function for batched processing."""
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length
            )
        
        # Convert pandas DataFrames to HuggingFace Datasets
        hf_datasets = {
            split: HFDataset.from_pandas(df)
            for split, df in self.splits.items()
        }
        
        # Tokenize
        print("\n🔤 Tokenizing...")
        tokenized_datasets = {}
        for split, dataset in hf_datasets.items():
            print(f"  Tokenizing {split} set...")
            tokenized = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
                desc=f"Tokenizing {split}"
            )
            tokenized.set_format(
                type='torch',
                columns=['input_ids', 'attention_mask', 'label']
            )
            tokenized_datasets[split] = tokenized
        
        print("✓ Tokenization complete")
        
        # Sample inspection
        print("\n📝 Sample tokenized example:")
        sample = tokenized_datasets['train'][0]
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Label: {sample['label']}")
        
        return DatasetDict(tokenized_datasets)
    
    def process_all(self, tokenizer):
        """
        Run complete data processing pipeline.
        
        Args:
            tokenizer: HuggingFace tokenizer
            
        Returns:
            DatasetDict: Tokenized datasets ready for training
        """
        self.load_data()
        self.validate_data()
        self.analyze_data()
        self.clean_data()
        self.split_data()
        
        # Save statistics
        stats_path = Path(self.config.output_dir) / 'data_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\n✓ Data statistics saved to {stats_path}")
        
        return self.prepare_datasets(tokenizer)


# =============================================================================
# 3. METRICS COMPUTATION
# =============================================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute comprehensive metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from trainer
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
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
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
    }
    
    # Add per-class F1 scores
    for i, f1_class in enumerate(f1):
        metrics[f'f1_class_{i}'] = float(f1_class)
    
    # Add per-class precision and recall
    for i, (prec, rec) in enumerate(zip(precision, recall)):
        metrics[f'precision_class_{i}'] = float(prec)
        metrics[f'recall_class_{i}'] = float(rec)
    
    return metrics


def detailed_evaluation(trainer, dataset, dataset_name="Test"):
    """
    Perform detailed evaluation with classification report and confusion matrix.
    
    Args:
        trainer: HuggingFace Trainer instance
        dataset: Dataset to evaluate
        dataset_name: Name for logging purposes
        
    Returns:
        Dict: Evaluation results
    """
    print("\n" + "=" * 80)
    print(f"DETAILED EVALUATION ON {dataset_name.upper()} SET")
    print("=" * 80)
    
    # Get predictions
    predictions = trainer.predict(dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    # Overall metrics
    results = predictions.metrics
    print(f"\n📊 Overall Metrics:")
    for key, value in results.items():
        if key.startswith('test_'):
            metric_name = key[5:]  # Remove 'test_' prefix
            print(f"  {metric_name}: {value:.4f}")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    report = classification_report(
        true_labels, pred_labels,
        digits=4,
        target_names=[f"Class {i}" for i in range(len(np.unique(true_labels)))]
    )
    print(report)
    
    # Confusion matrix
    print(f"\n🔀 Confusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(f"\n🔀 Normalized Confusion Matrix (%):")
    print((cm_normalized * 100).astype(int))
    
    # Save detailed results
    results_dir = Path(trainer.args.output_dir) / 'evaluation'
    results_dir.mkdir(exist_ok=True)
    
    # Save confusion matrix
    cm_path = results_dir / f'{dataset_name.lower()}_confusion_matrix.csv'
    pd.DataFrame(cm).to_csv(cm_path, index=False)
    
    # Save classification report
    report_dict = classification_report(
        true_labels, pred_labels,
        output_dict=True,
        target_names=[f"Class {i}" for i in range(len(np.unique(true_labels)))]
    )
    report_path = results_dir / f'{dataset_name.lower()}_classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {results_dir}")
    
    return results


# =============================================================================
# 4. TRAINING FUNCTION
# =============================================================================

def train_model(config: Config) -> Tuple[Trainer, Dict]:
    """
    Main training function.
    
    Args:
        config (Config): Configuration object
        
    Returns:
        Tuple[Trainer, Dict]: Trained trainer and test results
    """
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize experiment tracking
    if config.report_to == "wandb" and WANDB_AVAILABLE:
        wandb.init(
            project="deberta-topic-classification",
            name=config.run_name,
            config=asdict(config),
            reinit=True
        )
        print("✓ Weights & Biases initialized")
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("LOADING MODEL AND TOKENIZER")
    print("=" * 80)
    
    print(f"\n🤖 Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Process data
    data_processor = DataProcessor(config)
    tokenized_datasets = data_processor.process_all(tokenizer)
    
    # Load model
    print(f"\n🤖 Loading model: {config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
    )
    
    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Information:")
    print(f"  Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {n_trainable:,} ({n_trainable/1e6:.1f}M)")
    print(f"  Device: {config.device}")
    
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
        bf16=config.bf16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Evaluation
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if config.eval_strategy == "steps" else None,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy == "steps" else None,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # Logging
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        report_to=config.report_to if config.report_to != "none" else [],
        run_name=config.run_name,
        
        # Other
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ]
    )
    
    # Print training info
    print("\n" + "=" * 80)
    print("TRAINING INFORMATION")
    print("=" * 80)
    
    n_train = len(tokenized_datasets['train'])
    n_val = len(tokenized_datasets['val'])
    steps_per_epoch = n_train // (config.batch_size * config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    print(f"\n📈 Training Configuration:")
    print(f"  Training samples: {n_train:,}")
    print(f"  Validation samples: {n_val:,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total epochs: {config.num_epochs}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision (fp16): {config.fp16}")
    
    # Estimate training time
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
        if "A100" in gpu_name:
            estimated_time = total_steps * 0.8 / 60  # ~0.8 seconds per step on A100
        elif "V100" in gpu_name:
            estimated_time = total_steps * 1.2 / 60  # ~1.2 seconds per step on V100
        else:
            estimated_time = total_steps * 2.0 / 60  # ~2 seconds per step on other GPUs
        print(f"  Estimated training time: {estimated_time:.1f} minutes")
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()
    
    train_result = trainer.train()
    
    # Print training summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    metrics = train_result.metrics
    print(f"\n📊 Training Summary:")
    print(f"  Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Training time: {metrics.get('train_runtime', 0) / 60:.1f} minutes")
    print(f"  Samples per second: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)
    
    val_results = trainer.evaluate(tokenized_datasets['val'])
    print(f"\n📊 Validation Results:")
    for key, value in val_results.items():
        if key.startswith('eval_'):
            print(f"  {key[5:]}: {value:.4f}")
    
    # Evaluate on test set
    test_results = detailed_evaluation(
        trainer,
        tokenized_datasets['test'],
        dataset_name="Test"
    )
    
    # Save final model
    final_model_path = Path(config.output_dir) / "final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n💾 Model saved to: {final_model_path}")
    
    # Save configuration
    config.save(Path(config.output_dir) / "config.json")
    
    # Save training history
    history_path = Path(config.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'train_metrics': metrics,
            'val_metrics': val_results,
            'test_metrics': test_results,
        }, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")
    
    # Finish wandb run
    if config.report_to == "wandb" and WANDB_AVAILABLE:
        wandb.finish()
    
    return trainer, test_results


# =============================================================================
# 5. HYPERPARAMETER SEARCH WITH OPTUNA
# =============================================================================

def hyperparameter_search(base_config: Config, n_trials: int = 20) -> optuna.Study:
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        base_config (Config): Base configuration to modify
        n_trials (int): Number of optimization trials
        
    Returns:
        optuna.Study: Completed study object
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Install with: pip install optuna")
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 80)
    print(f"\nNumber of trials: {n_trials}")
    
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Metric to optimize (F1 macro)
        """
        # Sample hyperparameters
        config = Config(
            # Model
            model_name=base_config.model_name,
            num_labels=base_config.num_labels,
            
            # Data
            data_path=base_config.data_path,
            text_column=base_config.text_column,
            label_column=base_config.label_column,
            max_length=trial.suggest_categorical('max_length', [256, 384]),
            
            # Training
            output_dir=f"{base_config.output_dir}/trial_{trial.number}",
            num_epochs=trial.suggest_int('num_epochs', 4, 6),
            batch_size=trial.suggest_categorical('batch_size', [16, 32]),
            learning_rate=trial.suggest_float('learning_rate', 2e-5, 5e-5, log=True),
            weight_decay=trial.suggest_float('weight_decay', 0.01, 0.05),
            warmup_ratio=trial.suggest_float('warmup_ratio', 0.1, 0.2),
            
            # Optimization
            fp16=base_config.fp16,
            
            # Evaluation
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            
            # Logging
            report_to="none",  # Disable wandb for trials
            run_name=f"trial_{trial.number}",
            
            # Reproducibility
            seed=base_config.seed,
        )
        
        # Train model
        try:
            trainer, results = train_model(config)
            f1_macro = results['test_f1_macro']
            
            # Report intermediate values for pruning
            trial.report(f1_macro, step=config.num_epochs)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return f1_macro
            
        except Exception as e:
            print(f"\n❌ Trial {trial.number} failed with error: {e}")
            return 0.0
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='deberta_optimization',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\n🏆 Best Trial:")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  F1 Macro: {study.best_value:.4f}")
    
    print(f"\n🔧 Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save study results
    study_path = Path(base_config.output_dir) / "optuna_study.pkl"
    optuna.save_study(study, study_path)
    print(f"\n✓ Study saved to: {study_path}")
    
    # Save best parameters
    best_params_path = Path(base_config.output_dir) / "best_hyperparameters.json"
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
        }, f, indent=2)
    print(f"✓ Best parameters saved to: {best_params_path}")
    
    return study


# =============================================================================
# 6. MODEL INFERENCE
# =============================================================================

class TextClassifier:
    """
    Wrapper class for easy model inference.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize classifier.
        
        Args:
            model_path (str): Path to saved model directory
            device (str): Device to run inference on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def predict(self, texts: List[str], return_probs: bool = False) -> List:
        """
        Predict class labels for input texts.
        
        Args:
            texts (List[str]): List of input texts
            return_probs (bool): Whether to return probabilities
            
        Returns:
            List: Predictions (labels or probabilities)
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        else:
            predictions = torch.argmax(logits, dim=-1)
            return predictions.cpu().numpy()
    
    def predict_single(self, text: str, return_probs: bool = False):
        """
        Predict for a single text.
        
        Args:
            text (str): Input text
            return_probs (bool): Whether to return probabilities
            
        Returns:
            int or np.ndarray: Prediction
        """
        result = self.predict([text], return_probs=return_probs)
        return result[0]


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="DeBERTa Text Classification Training Pipeline"
    )
    
    # Training mode
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'hyperparam', 'eval'],
        help='Execution mode: train, hyperparam, or eval'
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--model_name', type=str, help='Model name or path')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_length', type=int, help='Max sequence length')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    
    # Flags
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = Config.load(args.config)
        print(f"✓ Configuration loaded from {args.config}")
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.model_name:
        config.model_name = args.model_name
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_length:
        config.max_length = args.max_length
    if args.seed:
        config.seed = args.seed
    if args.no_wandb:
        config.report_to = "none"
    if args.fp16:
        config.fp16 = True
    
    # Print banner
    print("\n" + "=" * 80)
    print(" " * 20 + "DeBERTa Text Classification")
    print(" " * 25 + "Training Pipeline v1.0")
    print("=" * 80)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Data: {config.data_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max length: {config.max_length}")
    print(f"  Seed: {config.seed}")
    print(f"  Device: {config.device}")
    
    # Execute based on mode
    if args.mode == 'train':
        print("\n🚀 Starting training...")
        trainer, results = train_model(config)
        print("\n✅ Training complete!")
        
    elif args.mode == 'hyperparam':
        print("\n🔍 Starting hyperparameter search...")
        study = hyperparameter_search(config, n_trials=args.n_trials)
        print("\n✅ Hyperparameter search complete!")
        
        # Optionally train with best parameters
        print("\n" + "=" * 80)
        response = input("Train final model with best parameters? (y/n): ")
        if response.lower() == 'y':
            # Update config with best parameters
            for key, value in study.best_params.items():
                setattr(config, key, value)
            config.output_dir = f"{config.output_dir}/final_model"
            config.report_to = "wandb" if not args.no_wandb else "none"
            
            print("\n🚀 Training final model...")
            trainer, results = train_model(config)
            print("\n✅ Final training complete!")
    
    elif args.mode == 'eval':
        print("\n📊 Evaluation mode...")
        model_path = Path(config.output_dir) / "final_model"
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("Please train a model first!")
            return
        
        # Load model
        classifier = TextClassifier(str(model_path), device=config.device)
        
        # Example predictions
        test_texts = [
            "This is a sample text for classification.",
            "Another example to demonstrate the model.",
        ]
        
        print("\n🔮 Example Predictions:")
        for text in test_texts:
            prediction = classifier.predict_single(text)
            probs = classifier.predict_single(text, return_probs=True)
            
            print(f"\nText: {text}")
            print(f"Predicted class: {prediction}")
            print(f"Probabilities: {probs}")
    
    print("\n" + "=" * 80)
    print("🎉 All done!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


# Hey! Solid follow-up on ONNX for production—it's a great choice for fast, portable inference across runtimes like ONNX Runtime (ORT), TensorRT, or even mobile/edge setups. To fit this seamlessly into your current script (e.g., after the trainer.save_model(final_model_path) line in train_model), add the following section. It'll export the trained DeBERTa to ONNX, then apply dynamic INT8 quantization (via Optimum + ORT) for ~2-4x speedup with minimal accuracy drop (~1-2% typically). For FP16, it uses graph optimization with mixed precision (GPU-only, for ~1.5-2x speedup).
# First, add these imports at the top of the script:
# Pythonfrom optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer, ORTOptimizer
# from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoOptimizationConfig
# Then, insert this right after trainer.save_model(final_model_path) and tokenizer.save_pretrained(final_model_path):
# Python
# =============================================================================
# 8. ONNX EXPORT, QUANTIZATION, AND OPTIMIZATION
# =============================================================================
print("\n" + "=" * 80)
print("EXPORTING TO ONNX AND OPTIMIZING FOR PRODUCTION")
print("=" * 80)

# Export to ONNX
onnx_path = Path(config.output_dir) / "onnx_model"
ort_model = ORTModelForSequenceClassification.from_pretrained(final_model_path, export=True)
ort_model.save_pretrained(onnx_path)
print(f"✓ ONNX model exported to {onnx_path}")

# Dynamic INT8 Quantization (CPU/GPU compatible, reduces model size ~4x)
quantizer = ORTQuantizer.from_pretrained(ort_model)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)  # Adjust for your hardware (e.g., avx512 or arm64)
quantized_path = Path(config.output_dir) / "quantized_int8_model"
quantizer.quantize(save_dir=quantized_path, quantization_config=dqconfig)
print(f"✓ INT8 quantized model saved to {quantized_path}")

# FP16 Optimization (GPU-only, for mixed precision inference)
optimizer = ORTOptimizer.from_pretrained(ort_model)
optimization_config = AutoOptimizationConfig.O4()  # Highest level: includes FP16, GELU approx, etc.
fp16_path = Path(config.output_dir) / "optimized_fp16_model"
optimizer.optimize(save_dir=fp16_path, optimization_config=optimization_config)
print(f"✓ FP16 optimized model saved to {fp16_path}")
# For production inference, update your TextClassifier to load the ONNX/quantized version (e.g., via ORTModelForSequenceClassification.from_pretrained(quantized_path) instead of AutoModelForSequenceClassification). Run it with ONNX Runtime sessions for low-latency preds—e.g., in a FastAPI endpoint or AWS Lambda. Test accuracy on your test set post-quantization to confirm the drop is acceptable.
# You'll need to install: pip install optimum[onnxruntime] onnxruntime. If you hit hardware-specific issues (e.g., no AVX512), swap the config to avx2 or test on your GPU env. What hardware are you targeting for prod? Any specific runtime (e.g., ORT on CPU vs. GPU)?
