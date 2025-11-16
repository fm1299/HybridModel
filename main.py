print("Starting Hybrid Model training script...")
print("Importing necessary libraries...")

import torch
import pandas as pd
import numpy as np
import os
import random
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from models.HybridModel import HybridEmotionRecognition, print_model_summary
from get_dataset import Four4All

# ==================== Config Loading ====================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Class names, device, and random seed from config
class_names = config['data']['class_names']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config['experiment']['seed'])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==================== Focal Loss Implementation ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
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
        else:
            return focal_loss

# ==================== Helper Functions ====================
def compute_class_weights(train_loader, num_classes=7):
    print("\nComputing class weights from training data...")
    class_counts = torch.zeros(num_classes)
    for _, labels in tqdm(train_loader, desc="Counting classes"):
        for label in labels:
            class_counts[label] += 1
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    print("\nClass Distribution in Training Set:")
    print("-" * 60)
    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
        print(f"{name:10s}: {int(count):6d} samples ({count/total_samples*100:5.2f}%) | Weight: {weight:.4f}")
    print("-" * 60)
    return class_weights

def get_data_loaders(config, use_weighted_sampler=False):
    aug = config['augmentation']
    train_transform = transforms.Compose([
        transforms.Resize((aug['img_size'], aug['img_size'])),
        transforms.Grayscale(num_output_channels=3) if aug.get('grayscale', True) else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip(aug['horizontal_flip_prob']),
        transforms.RandomRotation(degrees=aug['rotation_deg']),
        transforms.ColorJitter(**aug['color_jitter']),
        transforms.RandomAffine(
            degrees=0,
            translate=(aug['affine']['translate'], aug['affine']['translate']),
            scale=tuple(aug['affine']['scale'])
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((aug['img_size'], aug['img_size'])),
        transforms.Grayscale(num_output_channels=3) if aug.get('grayscale', True) else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = Four4All(
        csv_file=config['data']['train_csv'],
        img_dir=config['data']['train_dir'],
        transform=train_transform
    )
    val_dataset = Four4All(
        csv_file=config['data']['val_csv'],
        img_dir=config['data']['val_dir'],
        transform=val_transform
    )
    test_dataset = Four4All(
        csv_file=config['data']['test_csv'],
        img_dir=config['data']['test_dir'],
        transform=val_transform
    )
    print(f"\nDataset Statistics:")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")
    print(f"  Test images: {len(test_dataset)}")
    train_sampler = None
    if use_weighted_sampler:
        print("\nUsing WeightedRandomSampler for balanced batch sampling...")
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = Counter(labels)
        class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights_dict[label] for label in labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    return train_loader, val_loader, test_loader

# ==================== Checkpoint Management ====================
def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, valid_losses, 
                   train_accuracies, valid_accuracies, train_f1, valid_f1, path):
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

# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, config,
                start_epoch=0, checkpoint_path='rafdb/checkpoints/checkpoint_epoch_{}.pth'):
    patience = config['training']['patience']
    num_epochs = config['training']['epochs']
    use_focal_loss = (config['loss']['type'] == "FocalLoss")
    focal_gamma = config['loss'].get('gamma', 2.0)

    class_weights = compute_class_weights(train_loader, num_classes=config['model']['num_classes']).to(device)
    criterion = (
        FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
        if use_focal_loss
        else nn.CrossEntropyLoss(weight=class_weights)
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=tuple(config['optimizer']['betas'])
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'], min_lr=config['scheduler']['min_lr']
    )

    best_val_f1 = 0.0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []

    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['gradient_clip'])
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        train_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss = val_running_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        save_checkpoint(
            model, optimizer, scheduler, epoch+1,
            train_losses, val_losses, train_accuracies, val_accuracies,
            train_f1_scores, val_f1_scores,
            checkpoint_path.format(epoch+1)
        )
        print(f"  ✓ Checkpoint saved")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['experiment']['output_dir'], 'best_hybrid_model.pth'))
            print(f"  ✓ Best model saved (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement in validation F1 for {patience_counter}/{patience} epochs")
        if patience_counter >= patience:
            print("\nEarly stopping triggered!")
            print(f"Best validation F1 was at epoch: {np.argmax(val_f1_scores)+1}")
            print(f"Stopped at epoch: {epoch+1}")
            break

    # Save training history
    history_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Train_Accuracy': train_accuracies,
        'Val_Accuracy': val_accuracies,
        'Train_F1': train_f1_scores,
        'Val_F1': val_f1_scores
    })
    history_path = os.path.join(config['experiment']['output_dir'], 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print("\n✓ Training history saved to", history_path)
    torch.save(model.state_dict(), os.path.join(config['experiment']['output_dir'], 'hybrid_model_final.pth'))
    print("✓ Final model saved to", os.path.join(config['experiment']['output_dir'], 'hybrid_model_final.pth'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores

# ==================== Evaluation Function ====================
def evaluate_model(model, test_loader, config):
    save_path = config['experiment']['output_dir']
    print("\n" + "="*70)
    print("Evaluating Model on Test Set")
    print("="*70)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\nTest Results:  Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nAccuracy: {acc*100:.2f}%, F1: {f1:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(save_path, 'confusion_matrix_hybrid.png')
    plt.savefig(cm_path)
    print(f"✓ Confusion matrix saved to {cm_path}")
    plt.close()
    return acc, prec, rec, f1, cm

# ==================== Plotting Functions ====================
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies,
                        train_f1, val_f1, config):
    save_path = config['experiment']['output_dir']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [a*100 for a in train_accuracies], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in val_accuracies], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves_hybrid.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}/training_curves_hybrid.png")
    plt.close()

# ==================== Main Execution ====================
if __name__ == "__main__":
    os.makedirs(config['experiment']['output_dir'] + '/checkpoints', exist_ok=True)
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    print("\n" + "="*70)
    print("HYBRID CNN-TRANSFORMER EMOTION RECOGNITION")
    print("="*70)
    train_loader, val_loader, test_loader = get_data_loaders(config, use_weighted_sampler=False)
    print("\nInitializing Hybrid Model...")
    model = HybridEmotionRecognition(
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        pretrained_swin=config['model']['pretrained_swin'],
        use_gradient_checkpointing=False,  # You can add this to config if needed
        aggregation=config['model']['aggregation']
    ).to(device)
    print_model_summary(model)
    start_epoch = 0
    model, train_losses, val_losses, train_accuracies, val_accuracies, train_f1, val_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        start_epoch=start_epoch,
        checkpoint_path=config['experiment']['output_dir'] + '/checkpoints/checkpoint_epoch_{}.pth'
    )
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(config['experiment']['output_dir'], 'best_hybrid_model.pth')))
    acc, prec, rec, f1, cm = evaluate_model(model, test_loader, config)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_f1, val_f1, config)
    print("\n✓ Final model saved to", os.path.join(config['experiment']['output_dir'], 'hybrid_model_final.pth'))
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
