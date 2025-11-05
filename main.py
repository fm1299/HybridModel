"""
Enhanced Training Script for Hybrid Emotion Recognition Model
Includes all recommended improvements for class imbalance handling
"""

print("Starting Hybrid Model training script...")
print("Importing necessary libraries...")

import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                            recall_score, f1_score, classification_report,
                            precision_recall_fscore_support)
from collections import Counter
from models.HybridModel import HybridEmotionRecognition
from get_dataset import Four4All

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== Focal Loss Implementation ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha: Class weights (Tensor of size num_classes)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        # Use standard CE loss without reduction as basis
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, num_classes) logits
            targets: (B,) class labels
        """
        # Compute cross entropy loss
        ce_loss = self.ce(inputs, targets)
        
        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Focal loss
        focal_loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== Helper Functions ====================

def compute_class_weights(train_loader, num_classes=7):
    """
    Compute inverse frequency class weights for handling imbalance
    
    Args:
        train_loader: DataLoader for training set
        num_classes: Number of emotion classes
    
    Returns:
        weights: Tensor of class weights
    """
    print("\nComputing class weights from training data...")
    
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for _, labels in tqdm(train_loader, desc="Counting classes"):
        for label in labels:
            class_counts[label] += 1
    
    # Compute inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights to sum to num_classes
    class_weights = class_weights / class_weights.sum() * num_classes
    
    # Print class distribution
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print("\nClass Distribution in Training Set:")
    print("-" * 60)
    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
        print(f"{name:10s}: {int(count):6d} samples ({count/total_samples*100:5.2f}%) | Weight: {weight:.4f}")
    print("-" * 60)
    
    return class_weights


def get_data_loaders(batch_size=32, num_workers=4, use_weighted_sampler=False):
    """
    Create data loaders with enhanced augmentation
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        use_weighted_sampler: Use weighted random sampling for balancing
    """
    # Training transform with aggressive augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = Four4All(
        csv_file='rafdb/train_labels.csv',
        img_dir='rafdb/train',
        transform=train_transform
    )
    val_dataset = Four4All(
        csv_file='rafdb/valid_labels.csv',
        img_dir='rafdb/valid/',
        transform=val_transform
    )
    test_dataset = Four4All(
        csv_file='rafdb/test_labels.csv',
        img_dir='rafdb/test',
        transform=val_transform
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")
    print(f"  Test images: {len(test_dataset)}")
    
    # Create sampler if requested
    train_sampler = None
    if use_weighted_sampler:
        print("\nUsing WeightedRandomSampler for balanced batch sampling...")
        # Get all labels
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = Counter(labels)
        
        # Compute sample weights
        class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights_dict[label] for label in labels]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ==================== Checkpoint Management ====================

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, valid_losses, 
                   train_accuracies, valid_accuracies, train_f1, valid_f1, path):
    """Save model checkpoint with all training state"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies,
        'train_f1': train_f1,
        'valid_f1': valid_f1
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (model, optimizer, scheduler, checkpoint['epoch'],
            checkpoint['train_losses'], checkpoint['valid_losses'],
            checkpoint['train_accuracies'], checkpoint['valid_accuracies'],
            checkpoint.get('train_f1', []), checkpoint.get('valid_f1', []))


# ==================== Training Function ====================

def train_model(model, train_loader, val_loader, start_epoch=0, num_epochs=80,
                checkpoint_path='rafdb/checkpoints/checkpoint_epoch_{}.pth',
                use_focal_loss=True, focal_gamma=2.0):
    """
    Enhanced training loop with all improvements
    
    Args:
        model: HybridEmotionRecognition model
        train_loader: Training data loader
        val_loader: Validation data loader
        start_epoch: Starting epoch (for resuming)
        num_epochs: Total number of epochs
        checkpoint_path: Path template for saving checkpoints
        use_focal_loss: Use Focal Loss instead of CrossEntropy
        focal_gamma: Gamma parameter for Focal Loss
    """
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader, num_classes=7)
    class_weights = class_weights.to(device)
    
    # Loss function
    if use_focal_loss:
        print(f"\nUsing Focal Loss (gamma={focal_gamma}) with class weights")
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
    else:
        print("\nUsing Cross Entropy Loss with class weights")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - Using AdamW instead of SGD for better convergence
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Lower learning rate for fine-tuning
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - ReduceLROnPlateau for adaptive LR
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Early stopping parameters
    patience = 15
    best_val_f1 = 0.0  # Use F1 score instead of accuracy for imbalanced data
    patience_counter = 0
    
    # History tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    
    # Load checkpoint if resuming
    if start_epoch > 0:
        checkpoint_file = checkpoint_path.format(start_epoch)
        if os.path.exists(checkpoint_file):
            (model, optimizer, scheduler, _, train_losses, val_losses,
             train_accuracies, val_accuracies, train_f1_scores, val_f1_scores
            ) = load_checkpoint(model, optimizer, scheduler, checkpoint_file)
            best_val_f1 = max(val_f1_scores) if val_f1_scores else 0.0
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # ============ Training Phase ============
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)
        
        # ============ Validation Phase ============
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute validation metrics
        val_loss = val_running_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # Compute per-class metrics
        val_precision, val_recall, val_f1_per_class, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, zero_division=0
        )
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # ============ Print Epoch Summary ============
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val F1:   {val_f1:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Print per-class F1 scores
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        print(f"\n  Per-Class F1 Scores (Validation):")
        for name, f1_val in zip(class_names, val_f1_per_class):
            print(f"    {name:10s}: {f1_val:.4f}")
        
        # ============ Save Checkpoint ============
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            train_losses, val_losses, train_accuracies, val_accuracies,
            train_f1_scores, val_f1_scores,
            checkpoint_path.format(epoch + 1)
        )
        print(f"\n  ✓ Checkpoint saved")
        
        # ============ Early Stopping Logic ============
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'rafdb/best_hybrid_model.pth')
            print(f"  ✓ Best model saved (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement in validation F1 for {patience_counter}/{patience} epochs")
        
        if patience_counter >= patience:
            print(f"\n{'='*70}")
            print("Early stopping triggered - no improvement for {patience} epochs")
            print(f"{'='*70}")
            break
        
        print()
    
    # ============ Save Training History ============
    history_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Train_Accuracy': train_accuracies,
        'Val_Accuracy': val_accuracies,
        'Train_F1': train_f1_scores,
        'Val_F1': val_f1_scores
    })
    history_df.to_csv('rafdb/training_history.csv', index=False)
    print("\n✓ Training history saved to rafdb/training_history.csv")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores


# ==================== Evaluation Function ====================

def evaluate_model(model, test_loader, save_path='rafdb'):
    """
    Comprehensive model evaluation with detailed metrics
    """
    print("\n" + "="*70)
    print("Evaluating Model on Test Set")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute overall metrics
    acc = accuracy_score(all_labels, all_preds)
    
    # Compute per-class and macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("OVERALL TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:          {acc*100:.2f}%")
    print(f"  Macro Precision:   {precision_macro:.4f}")
    print(f"  Macro Recall:      {recall_macro:.4f}")
    print(f"  Macro F1-Score:    {f1_macro:.4f}")
    print(f"  Weighted F1-Score: {f1_weighted:.4f}")
    
    # Per-class results
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(f"\n{'='*70}")
    print("PER-CLASS METRICS")
    print(f"{'='*70}")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, name in enumerate(class_names):
        print(f"{name:<12} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support[i]:<10}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title(f'Hybrid Model - Test Accuracy: {acc*100:.2f}%\nMacro F1: {f1_macro:.4f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix_hybrid.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {save_path}/confusion_matrix_hybrid.png")
    plt.close()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'F1_Score': f1_per_class,
        'Support': support
    })
    results_df.to_csv(f'{save_path}/test_results_per_class.csv', index=False)
    print(f"✓ Per-class results saved to {save_path}/test_results_per_class.csv")
    
    return acc, precision_macro, recall_macro, f1_macro, cm


# ==================== Plotting Functions ====================

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies,
                        train_f1, val_f1, save_path='rafdb'):
    """Plot comprehensive training curves"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, [a*100 for a in train_accuracies], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in val_accuracies], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[2].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves_hybrid.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}/training_curves_hybrid.png")
    plt.close()


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('rafdb/checkpoints', exist_ok=True)
    os.makedirs('rafdb', exist_ok=True)
    
    print("\n" + "="*70)
    print("HYBRID CNN-TRANSFORMER EMOTION RECOGNITION")
    print("="*70)
    
    # ============ Load Data ============
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=32,
        num_workers=4,
        use_weighted_sampler=False  # Set to True for weighted sampling
    )
    
    # ============ Initialize Model ============
    print("\nInitializing Hybrid Model...")
    model = HybridEmotionRecognition(
        num_classes=7,
        embed_dim=512,
        num_heads=8,
        dropout=0.2,  # Increased from 0.1 for better regularization
        pretrained_swin=True,
        use_gradient_checkpointing=False,  # Set True if memory limited
        aggregation='mean'
    ).to(device)
    
    # Print model summary
    from models.HybridModel import print_model_summary
    print_model_summary(model)
    
    # ============ Train Model ============
    start_epoch = 0  # Change to resume from checkpoint
    (model, train_losses, val_losses, train_accuracies, 
     val_accuracies, train_f1, val_f1) = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        num_epochs=80,
        checkpoint_path='rafdb/checkpoints/checkpoint_epoch_{}.pth',
        use_focal_loss=True,  # Use Focal Loss for class imbalance
        focal_gamma=2.0
    )
    
    # ============ Load Best Model ============
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load('rafdb/best_hybrid_model.pth'))
    
    # ============ Evaluate on Test Set ============
    acc, prec, rec, f1, cm = evaluate_model(model, test_loader, save_path='rafdb')
    
    # ============ Plot Training Curves ============
    plot_training_curves(train_losses, val_losses, train_accuracies, 
                        val_accuracies, train_f1, val_f1, save_path='rafdb')
    
    # ============ Save Final Model ============
    torch.save(model.state_dict(), 'rafdb/hybrid_model_final.pth')
    print("\n✓ Final model saved to rafdb/hybrid_model_final.pth")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
