import argparse
import ast
import logging
import os
import random
import re
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import nlpaug.augmenter.word as naw

from stanza.models.common import loss, utils
from stanza.models.pos.vocab import CharVocab
from stanza.models.classifiers.data import read_dataset, dataset_labels
from stanza.models.classifiers.trainer import Trainer
from stanza.models.classifiers.utils import WVType, ExtraVectors, ModelType
from stanza.models.common.peft_config import add_peft_args, resolve_peft_args
from stanza.utils.confusion import format_confusion, confusion_to_accuracy, confusion_to_macro_f1

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger('stanza')
tlogger = logging.getLogger('stanza.classifiers.trainer')
logging.getLogger('elmoformanylangs').setLevel(logging.WARNING)

DEFAULT_TRAIN = 'data/sentiment/en_sstplus.train.txt'
DEFAULT_DEV = 'data/sentiment/en_sst3roots.dev.txt'
DEFAULT_TEST = 'data/sentiment/en_sst3roots.test.txt'

class Loss(Enum):
    CROSS = 1
    WEIGHTED_CROSS = 2
    LOG_CROSS = 3
    FOCAL = 4
    LABEL_SMOOTHING = 5

class DevScoring(Enum):
    ACCURACY = 'ACC'
    WEIGHTED_F1 = 'WF'
    MACRO_F1 = 'MF1'

class ModelArchitecture(Enum):
    CNN = 1
    TRANSFORMER = 2
    HYBRID = 3

# Initialize data augmenter
try:
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    AUGMENTATION_AVAILABLE = True
except:
    AUGMENTATION_AVAILABLE = False
    logger.warning("Text augmentation not available")

def augment_dataset(dataset, augment_factor=0.1):
    """Apply text augmentation to dataset"""
    if not AUGMENTATION_AVAILABLE:
        return dataset
        
    augmented = []
    for item in dataset:
        if random.random() < augment_factor:
            augmented_text = aug.augment(item.text)
            augmented.append(Example(augmented_text, item.label))
    return dataset + augmented

def build_argparse():
    """Enhanced argument parser with new options"""
    parser = argparse.ArgumentParser()
    
    # Existing arguments...
    # Add new arguments for improvements:
    
    # Model architecture
    parser.add_argument('--model_arch', type=lambda x: ModelArchitecture[x.upper()], 
                       default=ModelArchitecture.CNN,
                       help='Model architecture: CNN, TRANSFORMER, or HYBRID')
    
    # Transformer options
    if TRANSFORMERS_AVAILABLE:
        parser.add_argument('--transformer_model', type=str, default='distilbert-base-uncased',
                          help='Pretrained transformer model name')
        parser.add_argument('--freeze_transformer', action='store_true',
                          help='Freeze transformer weights during training')
    
    # Training improvements
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                      help='Label smoothing factor (0.0 to 0.5)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping value')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                      help='Number of epochs to wait before early stopping')
    
    # Data augmentation
    parser.add_argument('--augment_data', action='store_true',
                      help='Enable text data augmentation')
    parser.add_argument('--augment_factor', type=float, default=0.1,
                      help='Percentage of samples to augment')
    
    # Hyperparameter tuning
    parser.add_argument('--auto_tune', action='store_true',
                      help='Enable automatic hyperparameter tuning')
    
    add_peft_args(parser)
    utils.add_device_args(parser)
    
    return parser

class HybridClassifier(nn.Module):
    """Combines CNN and Transformer features"""
    def __init__(self, args, num_labels):
        super().__init__()
        self.args = args
        self.num_labels = num_labels
        
        # Transformer component
        if TRANSFORMERS_AVAILABLE and args.model_arch in [ModelArchitecture.TRANSFORMER, ModelArchitecture.HYBRID]:
            self.transformer = AutoModel.from_pretrained(args.transformer_model)
            if args.freeze_transformer:
                for param in self.transformer.parameters():
                    param.requires_grad = False
            self.transformer_hidden_size = self.transformer.config.hidden_size
        else:
            self.transformer = None
            
        # CNN component
        if args.model_arch in [ModelArchitecture.CNN, ModelArchitecture.HYBRID]:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
            self.convs = nn.ModuleList([
                nn.Conv1d(args.embedding_dim, args.filter_channels, fs)
                for fs in args.filter_sizes
            ])
            self.cnn_hidden_size = len(args.filter_sizes) * args.filter_channels
            
        # Classifier head
        if args.model_arch == ModelArchitecture.HYBRID:
            input_size = self.transformer_hidden_size + self.cnn_hidden_size
        elif args.model_arch == ModelArchitecture.TRANSFORMER:
            input_size = self.transformer_hidden_size
        else:
            input_size = self.cnn_hidden_size
            
        self.classifier = nn.Sequential(
            nn.Linear(input_size, args.fc_shapes[0]),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc_shapes[0], num_labels)
        )
    
    def forward(self, inputs):
        features = []
        
        # Transformer features
        if self.transformer is not None:
            transformer_outputs = self.transformer(**inputs)
            cls_feature = transformer_outputs.last_hidden_state[:, 0, :]
            features.append(cls_feature)
        
        # CNN features
        if hasattr(self, 'convs'):
            embedded = self.embedding(inputs['input_ids'])
            embedded = embedded.permute(0, 2, 1)
            
            conv_features = []
            for conv in self.convs:
                conv_out = F.relu(conv(embedded))
                pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                conv_features.append(pooled)
                
            cnn_features = torch.cat(conv_features, 1)
            features.append(cnn_features)
        
        # Combine features
        if len(features) > 1:
            combined = torch.cat(features, 1)
        else:
            combined = features[0]
            
        return self.classifier(combined)

def train_model(trainer, model_file, checkpoint_file, args, train_set, dev_set, labels):
    """Enhanced training loop with new features"""
    tlogger.setLevel(logging.DEBUG)
    
    # Initialize AMP
    scaler = GradScaler(enabled=args.use_amp)
    
    # Early stopping
    best_score = -np.inf
    epochs_no_improve = 0
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, device=trainer.device)
    
    # Enhanced loss functions
    if args.label_smoothing > 0:
        loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    elif args.loss == Loss.FOCAL:
        from focal_loss import FocalLoss
        loss_function = FocalLoss(gamma=args.loss_focal_gamma)
    else:
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(trainer.epochs_trained, args.max_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            inputs, labels = batch
            
            with autocast(enabled=args.use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            # AMP scaling and backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            running_loss += loss.item()
        
        # Evaluation
        dev_score, accuracy, macro_f1 = score_dev_set(model, dev_set, args.dev_eval_scoring)
        
        # Early stopping check
        if dev_score > best_score:
            best_score = dev_score
            epochs_no_improve = 0
            trainer.save(model_file)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                logger.info(f"Early stopping after {epochs_no_improve} epochs without improvement")
                break
        
        # Learning rate scheduling
        scheduler.step(dev_score)

def main():
    args = parse_args()
    
    # Data loading with augmentation
    train_set = read_dataset(args.train_file, args.wordvec_type, args.min_train_len)
    if args.augment_data:
        train_set = augment_dataset(train_set, args.augment_factor)
    
    # Initialize model
    if args.model_arch == ModelArchitecture.TRANSFORMER and not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available, falling back to CNN")
        args.model_arch = ModelArchitecture.CNN
    
    if args.model_arch == ModelArchitecture.TRANSFORMER:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.transformer_model, 
            num_labels=len(set(dataset_labels(train_set)))
        )
    elif args.model_arch == ModelArchitecture.HYBRID:
        model = HybridClassifier(args, len(set(dataset_labels(train_set))))
    else:
        # Original CNN model
        model = Trainer.build_new_model(args, train_set)
    
    # Training with enhanced features
    train_model(trainer, save_name, checkpoint_file, args, train_set, dev_set, labels)
    
    # Enhanced evaluation
    predictions = dataset_predictions(model, test_set)
    logger.info("\nClassification Report:\n%s", 
               classification_report(
                   [x.sentiment for x in test_set],
                   predictions,
                   target_names=model.labels
               ))

if __name__ == '__main__':
    main()