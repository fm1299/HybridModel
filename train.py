"""
================================================================================
HYBRID CNN-TRANSFORMER EMOTION RECOGNITION - FINAL TRAINING SCRIPT
================================================================================
Consolidated training script for thesis implementation.

Features:
- YAML configuration support
- Checkpoint saving for training resume
- Separate train/val/test transforms
- F1 score tracking during training (Macro F1 for early stopping)
- Optional test evaluation during training
- Training history saved to CSV
- Per-class metrics during validation
- SAM optimizer option
- Complete reproducibility with seed workers

Author: [Your Name]
University: UNSA - Universidad Nacional de San Agustín de Arequipa
Thesis: Hybrid CNN-Transformer for Facial Emotion Recognition
================================================================================
"""

print("=" * 70)
print("HYBRID CNN-TRANSFORMER EMOTION RECOGNITION")
print("Final Training Script v1.0")
print("=" * 70)

# ==================== Imports ====================
import os
import copy
import yaml
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
from collections import Counter
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from models.HybridModel import HybridEmotionRecognition, print_model_summary
from models.sam import SAM
from get_dataset import Four4All

print("✓ All libraries imported successfully")


# ==================== Configuration ====================
def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


config = load_config()

# Extract commonly used config values
CLASS_NAMES = config['data']['class_names']
NUM_CLASSES = config['model']['num_classes']
OUTPUT_DIR = config['experiment']['output_dir']

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ==================== Reproducibility ====================
# def set_seed(seed=42):
#     """Set random seeds for reproducibility"""
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


# def seed_worker(worker_id):
#     """Set seed for DataLoader workers"""
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# set_seed(config['experiment']['seed'])
# print(f"✓ Random seed set to {config['experiment']['seed']}")


# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in emotion recognition.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    Args:
        alpha: Class weights tensor (optional)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        focal_loss = focal_term * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==================== Data Transforms ====================
def build_train_transform(config):
    """
    Build training transform with data augmentation.
    All augmentation parameters are extracted from config['augmentation'].
    """
    aug = config['augmentation']
    
    transform_list = [
        transforms.Resize((aug['img_size'], aug['img_size'])),
    ]
    
    # Grayscale to 3-channel if specified
    grayscale = aug.get('grayscale', False)
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    
    # Random rotation - extract from config
    rotation_deg = aug.get('rotation_deg', 10)
    transform_list.append(transforms.RandomRotation(degrees=rotation_deg))

    # Horizontal flip - extract from config
    h_flip_prob = aug.get('horizontal_flip_prob', 0.5)
    transform_list.append(transforms.RandomHorizontalFlip(p=h_flip_prob))

    # Random autocontrast - extract from config or use default
    autocontrast_prob = aug.get('autocontrast_prob', 0.3)
    if autocontrast_prob > 0:
        transform_list.append(transforms.RandomAutocontrast(p=autocontrast_prob))
    
    # Color jitter - extract all values from config
    cj = aug.get('color_jitter', {})
    if cj:
        brightness = cj.get('brightness', 0.0)
        contrast = cj.get('contrast', 0.0)
        saturation = cj.get('saturation', 0.0)
        #hue = cj.get('hue', 0.0)  # Also support hue if added to config
        transform_list.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        ))
    
    # Random affine - extract all values from config
    aff = aug.get('affine', {})
    if aff:
        translate_val = aff.get('translate', 0.1)
        scale_range = tuple(aff.get('scale', [0.9, 1.1]))
        shear = aff.get('shear', None)  # Also support shear if added to config
        transform_list.append(transforms.RandomAffine(
            degrees=0,
            translate=(translate_val, translate_val),
            scale=scale_range,
            shear=shear
        ))
    
    # Convert to tensor and normalize (ImageNet stats for pretrained Swin)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def build_eval_transform(config):
    """
    Build evaluation transform (no augmentation).
    Only resize, grayscale conversion (if needed), and normalization.
    """
    aug = config['augmentation']
    
    transform_list = [
        transforms.Resize((aug['img_size'], aug['img_size'])),
    ]
    
    if aug.get('grayscale', False):
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    
    # Use ImageNet normalization to match training (important for pretrained models)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


# ==================== Data Loaders ====================
def get_data_loaders(config, use_weighted_sampler=False):
    """Create train, validation, and test data loaders"""
    
    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)
    
    # Create datasets
    train_dataset = Four4All(
        config['data']['train_csv'],
        config['data']['train_dir'],
        transform=train_transform
    )
    val_dataset = Four4All(
        config['data']['val_csv'],
        config['data']['val_dir'],
        transform=eval_transform
    )
    test_dataset = Four4All(
        config['data']['test_csv'],
        config['data']['test_dir'],
        transform=eval_transform
    )
    
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"  Training samples:   {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Test samples:       {len(test_dataset):,}")
    print(f"  Total:              {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
    
    # # Weighted sampler for class imbalance
    # train_sampler = None
    # if use_weighted_sampler:
    #     print("\n  Using WeightedRandomSampler for balanced batch sampling...")
    #     labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    #     class_counts = Counter(labels)
    #     class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
    #     sample_weights = [class_weights_dict[label] for label in labels]
    #     train_sampler = WeightedRandomSampler(
    #         weights=sample_weights,
    #         num_samples=len(sample_weights),
    #         replacement=True
    #     )
    
    # Generator for reproducible DataLoader
    # g = torch.Generator()
    # g.manual_seed(config['experiment']['seed'])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        # worker_init_fn=seed_worker,
        # generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        #worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        #worker_init_fn=seed_worker
    )
    
    return train_loader, val_loader, test_loader


# ==================== Class Weights ====================
def compute_class_weights(train_loader, num_classes=7):
    """Compute class weights for handling imbalanced data"""
    print("\nComputing class weights from training data...")
    
    class_counts = torch.zeros(num_classes)
    for _, labels in tqdm(train_loader, desc="  Counting classes"):
        for label in labels:
            class_counts[label] += 1
    
    total_samples = class_counts.sum()
    
    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"\n  {'Class':<12} {'Samples':>8} {'Percentage':>12} {'Weight':>10}")
    print("  " + "-" * 44)
    for i, (name, count, weight) in enumerate(zip(CLASS_NAMES, class_counts, class_weights)):
        pct = count / total_samples * 100
        print(f"  {name:<12} {int(count):>8} {pct:>11.2f}% {weight:>10.4f}")
    print("  " + "-" * 44)
    
    return class_weights


# ==================== Checkpoint Management ====================
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint"""
    print(f"\n  Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, config, resume_checkpoint=None):
    """
    Main training function with all features.
    
    Args:
        model: HybridEmotionRecognition model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Configuration dictionary
        resume_checkpoint: Path to checkpoint for resuming training
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    
    # ============ Setup ============
    num_epochs = config['training']['epochs']
    patience = config['training']['patience']
    gradient_clip = config['training'].get('gradient_clip', 1.0)
    use_focal_loss = config['loss']['type'] == 'FocalLoss'
    focal_gamma = config['loss'].get('gamma', 2.0)
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader, NUM_CLASSES).to(device)
    
    # Loss function
    if use_focal_loss:
        print(f"\n✓ Using Focal Loss (gamma={focal_gamma}) with class weights")
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
    else:
        label_smoothing = config['loss'].get('label_smoothing', 0.0)
        print(f"\n✓ Using Cross Entropy Loss with class weights (label_smoothing={label_smoothing})")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    # Optimizer
    opt_cfg = config['optimizer']
    if opt_cfg['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_cfg['lr'],
            momentum=opt_cfg.get('momentum', 0.9),
            weight_decay=opt_cfg['weight_decay']
        )
        print(f"✓ Using SGD optimizer (lr={opt_cfg['lr']}, momentum={opt_cfg.get('momentum', 0.9)})")
    elif opt_cfg['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=tuple(opt_cfg.get('betas', [0.9, 0.999]))
        )
        print(f"✓ Using AdamW optimizer (lr={opt_cfg['lr']})")
    elif opt_cfg['type'] == 'SAM':
        base_optimizer = optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=opt_cfg['lr'],
            momentum=opt_cfg.get('momentum', 0.9),
            weight_decay=opt_cfg['weight_decay']
        )
        print(f"✓ Using SAM optimizer (lr={opt_cfg['lr']})")
    else:
        raise ValueError(f"Unknown optimizer type: {opt_cfg['type']}")
    
    # Learning rate scheduler
    sch_cfg = config['scheduler']
    if sch_cfg['type'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer if opt_cfg['type'] != 'SAM' else optimizer.base_optimizer,
            mode='max',  # Maximize F1 score
            factor=sch_cfg['factor'],
            patience=sch_cfg['patience'],
            min_lr=sch_cfg['min_lr']
        )
        print(f"✓ Using ReduceLROnPlateau scheduler (factor={sch_cfg['factor']})")

    elif sch_cfg['type'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_max=sch_cfg['T_max'],
            eta_min=sch_cfg['eta_min']
        )
        print(f"✓ Using CosineAnnealingLR scheduler (T_max={sch_cfg['T_max']}, eta_min={sch_cfg['eta_min']})")
    else:
        raise ValueError(f"Unknown scheduler type: {sch_cfg['type']}")
    
    # ============ History Tracking ============
    history = {
        'epoch': [],
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
        'learning_rate': []
    }
    
    # ============ Early Stopping ============
    best_val_f1 = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch, saved_metrics = load_checkpoint(model, optimizer, scheduler, resume_checkpoint)
        history = saved_metrics.get('history', history)
        best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
        print(f"  Resuming from epoch {start_epoch + 1}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"  Epochs: {num_epochs} | Early stopping patience: {patience}")
    print(f"  Metric for early stopping: Validation Macro F1")
    print(f"{'='*70}\n")
    
    # ============ Training Loop ============
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = datetime.now()
        
        # -------- Training Phase --------
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if opt_cfg['type'] == 'SAM':
                # SAM requires two forward-backward passes
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.first_step()
                
                outputs = model(inputs)
                criterion(outputs, labels).backward()
                optimizer.second_step()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        
        # -------- Validation Phase --------
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # -------- Update Learning Rate --------
        current_lr = optimizer.param_groups[0]['lr'] if opt_cfg['type'] != 'SAM' else optimizer.base_optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()
        
        # -------- Log Results --------
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # -------- Update History --------
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['learning_rate'].append(current_lr)
        
        # -------- Early Stopping Check --------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  ✓ New best model! (Val F1: {val_f1:.4f})")
            
            # Save best model
            torch.save(best_model_state, os.path.join(OUTPUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                {'history': history},
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
            print(f"  ✓ Checkpoint saved")
        
        if patience_counter >= patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING at epoch {epoch+1}")
            print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
            print(f"{'='*70}")
            break
    
    # ============ Restore Best Model ============
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model from epoch {best_epoch}")
    
    # ============ Save Training History ============
    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to {history_path}")
    
    return model, history


# ==================== Evaluation Function ====================
def evaluate_model(model, data_loader, split_name='Test'):
    """
    Comprehensive model evaluation with all metrics.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        split_name: Name of the split (for logging)
    
    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {split_name.upper()} SET")
    print(f"{'='*70}")
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    loss = running_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"  Loss:             {loss:.4f}")
    print(f"  Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision (Macro):{precision_macro:.4f}")
    print(f"  Recall (Macro):   {recall_macro:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    
    # Confusion matrix visualization
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Percentage'}, annot_kws={'size': 12})
    plt.title(f'{split_name} Confusion Matrix\nAccuracy: {acc*100:.2f}% | Macro F1: {f1_macro:.4f}', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{split_name.lower()}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Confusion matrix saved to {cm_path}")
    
    # Save detailed results
    results = {
        'loss': loss,
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    # Save per-class results to CSV
    results_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'F1_Score': f1_per_class,
        'Support': [cm[i].sum() for i in range(len(CLASS_NAMES))]
    })
    results_path = os.path.join(OUTPUT_DIR, f'{split_name.lower()}_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to {results_path}")
    
    return results


# ==================== Plotting Functions ====================
def plot_training_curves(history, save_path=None):
    """Plot training curves: Loss, Accuracy, and F1 Score"""
    
    epochs = history['epoch']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[1]
    ax.plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # F1 Score curves
    ax = axes[2]
    ax.plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_f1'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    
    # Log start time
    start_time = datetime.now()
    print(f"\nTraining started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ============ Load Data ============
    train_loader, val_loader, test_loader = get_data_loaders(
        config, 
        use_weighted_sampler=False
    )
    
    # ============ Initialize Model ============
    print(f"\n{'='*70}")
    print("INITIALIZING MODEL")
    print(f"{'='*70}")
    
    model = HybridEmotionRecognition(
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        pretrained_swin=True,
        aggregation=config['model']['aggregation']
    ).to(device)
    
    print_model_summary(model)
    
    # ============ Train Model ============
    resume_checkpoint = config['training'].get('resume_checkpoint', None)
    
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        resume_checkpoint=resume_checkpoint
    )
    
    # ============ Plot Training Curves ============
    plot_training_curves(history)
    
    # ============ Final Evaluation ============
    # Load best model
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n✓ Loaded best model for final evaluation")
    
    # Evaluate on all splits
    test_results = evaluate_model(model, test_loader, 'Test')
    
    # ============ Save Final Model ============
    final_model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_results': test_results
    }, final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # ============ Summary ============
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Duration: {duration}")
    print(f"  Best Validation F1: {max(history['val_f1']):.4f}")
    print(f"  Final Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Final Test F1 (Macro): {test_results['f1_macro']:.4f}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*70}")
