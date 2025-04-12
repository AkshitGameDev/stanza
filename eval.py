#!/usr/bin/env python3
"""
Evaluate Language Identification Model
"""
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import BiLSTMLanguageID  # Your model class
from data import LanguageDataset    # Your dataset class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('eval')

def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[BiLSTMLanguageID, dict]:
    """Load trained model from checkpoint"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = BiLSTMLanguageID(config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, config

def evaluate(model: BiLSTMLanguageID, 
             test_loader: DataLoader, 
             idx_to_tag: Dict[int, str]) -> Dict[str, float]:
    """Run evaluation and generate metrics"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=list(idx_to_tag.values()),
        output_dict=True
    )
    
    # Confusion matrix visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=idx_to_tag.values(), 
                yticklabels=idx_to_tag.values())
    plt.savefig('confusion_matrix.png')
    
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = load_model(args.checkpoint, device)
    
    # Load test data
    test_dataset = LanguageDataset(args.data_dir, 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Evaluate
    metrics = evaluate(model, test_loader, config['idx_to_tag'])
    
    # Save results
    with open('eval_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Full report saved to eval_results.json")
    logger.info(f"Confusion matrix saved to confusion_matrix.png")

if __name__ == '__main__':
    main()