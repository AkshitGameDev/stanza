#!/usr/bin/env python3
"""
Enhanced Bi-LSTM Language Identification System
- Modular architecture
- Mixed precision training
- Comprehensive metrics
- Reproducibility features
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stanza.langid')

class LanguageDataset(Dataset):
    """Improved data loading and preprocessing"""
    def __init__(self, data_dir: str, split: str = 'train', max_length: int = 100):
        self.data = self._load_data(data_dir, split)
        self.char_to_idx, self.tag_to_idx = self._build_vocab()
        self.max_length = max_length

    def _load_data(self, data_dir: str, split: str) -> List[dict]:
        """Load and validate data files"""
        files = [f for f in Path(data_dir).glob(f'*{split}*') if f.is_file()]
        if not files:
            raise FileNotFoundError(f"No {split} files found in {data_dir}")
        
        data = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in {file}: {line.strip()}")
        return data

    def _build_vocab(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build character and language vocabularies"""
        chars = set()
        tags = set()
        
        for item in self.data:
            chars.update(item['text'])
            tags.add(item['label'])
        
        char_to_idx = {c: i+2 for i, c in enumerate(chars)}  # 0: pad, 1: unk
        char_to_idx['<PAD>'] = 0
        char_to_idx['<UNK>'] = 1
        
        tag_to_idx = {t: i for i, t in enumerate(sorted(tags))}
        
        return char_to_idx, tag_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        text = item['text'][:self.max_length]
        label = item['label']
        
        # Convert to indices
        char_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in text]
        label_idx = self.tag_to_idx[label]
        
        # Padding
        padded = torch.zeros(self.max_length, dtype=torch.long)
        padded[:len(char_indices)] = torch.tensor(char_indices)
        
        return padded, torch.tensor(label_idx)

class BiLSTMLanguageID(nn.Module):
    """Enhanced model architecture"""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['embed_dim'],
            padding_idx=0
        )
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=config['embed_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['num_layers'],
            bidirectional=True,
            batch_first=True
        )
        
        # Classifier
        self.fc = nn.Linear(2 * config['hidden_dim'], config['num_classes'])
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        pooled = lstm_out.mean(dim=1)  # Mean pooling
        return self.fc(self.dropout(pooled))

class Trainer:
    """Enhanced training loop with AMP and early stopping"""
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize model
        self.model = BiLSTMLanguageID(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3
        )
        self.scaler = GradScaler(enabled=config['use_amp'])
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(config['class_weights'], device=self.device),
            label_smoothing=config['label_smoothing']
        )
        
        # Track best metric
        self.best_accuracy = 0.0
        self.early_stop_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config['use_amp']):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            if self.config['grad_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, dict]:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        report = classification_report(
            all_labels, all_preds,
            target_names=list(self.config['idx_to_tag'].values()),
            output_dict=True
        )
        
        return accuracy, report, total_loss / len(eval_loader)

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'config': self.config,
            'best_accuracy': self.best_accuracy
        }
        torch.save(state, path)
        if is_best:
            best_path = str(Path(path).with_name('model_best.pt'))
            torch.save(state, best_path)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='BiLSTM Language Identification')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train/dev/test files')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load datasets
    train_dataset = LanguageDataset(args.data_dir, 'train', args.max_length)
    dev_dataset = LanguageDataset(args.data_dir, 'dev', args.max_length)

    # Calculate class weights
    labels = [item['label'] for item in train_dataset.data]
    class_counts = np.bincount([train_dataset.tag_to_idx[l] for l in labels])
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model configuration
    config = {
        'vocab_size': len(train_dataset.char_to_idx),
        'num_classes': len(train_dataset.tag_to_idx),
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'class_weights': class_weights,
        'label_smoothing': args.label_smoothing,
        'grad_clip': args.grad_clip,
        'use_amp': args.use_amp,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'idx_to_tag': {v: k for k, v in train_dataset.tag_to_idx.items()}
    }

    # Initialize trainer
    trainer = Trainer(config)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        accuracy, report, val_loss = trainer.evaluate(dev_loader)
        logger.info(f"Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{json.dumps(report, indent=2)}")
        
        # Update learning rate
        trainer.scheduler.step(accuracy)
        
        # Check for early stopping
        if accuracy > trainer.best_accuracy:
            trainer.best_accuracy = accuracy
            trainer.early_stop_counter = 0
            trainer.save_checkpoint('checkpoint.pt', is_best=True)
        else:
            trainer.early_stop_counter += 1
            if trainer.early_stop_counter >= args.patience:
                logger.info(f"Early stopping after {args.patience} epochs without improvement")
                break

if __name__ == '__main__':
    main()