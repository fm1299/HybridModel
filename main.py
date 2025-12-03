print("Starting Hybrid Model training script...")
print("Importing necessary libraries...")

import torch
import pandas as pd
import numpy as np
import os
import copy
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from models.HybridModel import HybridEmotionRecognition, print_model_summary
from get_dataset import Four4All

# ==================== Config Loading ====================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class_names = config['data']['class_names']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
def build_transform(config, is_train=True):
    aug = config['augmentation']
    transform = [
        transforms.Resize((aug['img_size'], aug['img_size'])),
        transforms.Grayscale(num_output_channels=3) if aug.get('grayscale', True) else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip(aug.get('horizontal_flip_prob', 0.5)),
        transforms.RandomRotation(aug.get('rotation_deg', 15))
    ]

    cj = aug.get('color_jitter', {})
    if cj:
        transform.append(transforms.ColorJitter(
            brightness=cj.get('brightness', 0.0),
            contrast=cj.get('contrast', 0.0),
            saturation=cj.get('saturation', 0.0)
        ))

    aff = aug.get('affine', {})
    if aff:
        transform.append(transforms.RandomAffine(
            degrees=0,
            translate=(aff.get('translate', 0), aff.get('translate', 0)),
            scale=tuple(aff.get('scale', [1.0, 1.0]))
        ))

    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform)


def get_data_loaders(config, use_weighted_sampler=False):
    # Apply the same transform to all (train/val/test)
    transform = build_transform(config, is_train=True)

    train_dataset = Four4All(config['data']['train_csv'],
                             config['data']['train_dir'],
                             transform=transform)
    val_dataset = Four4All(config['data']['val_csv'],
                           config['data']['val_dir'],
                           transform=transform)
    test_dataset = Four4All(config['data']['test_csv'],
                            config['data']['test_dir'],
                            transform=transform)

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
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


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
    for name, count, weight in zip(class_names, class_counts, class_weights):
        print(f"{name:10s}: {int(count):6d} samples ({count/total_samples*100:5.2f}%) | Weight: {weight:.4f}")
    print("-" * 60)

    return class_weights


def plot_training_curves(train_losses, val_losses, test_losses,
                         train_accs, val_accs, test_accs, config):
    out_dir = config['experiment']['output_dir']
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.plot(test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.plot(test_accs, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


# ==================== Main Training Function ====================
def train_model(model, train_loader, val_loader, test_loader, config, start_epoch=0):
    patience = config['training']['patience']
    num_epochs = config['training']['epochs']
    use_focal_loss = (config['loss']['type'] == "FocalLoss")
    focal_gamma = config['loss'].get('gamma', 2.0)
    aux_weight = config['loss'].get('aux_weight', 0.3)

    class_weights = compute_class_weights(
        train_loader, num_classes=config['model']['num_classes']
    ).to(device)

    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights,
                              gamma=focal_gamma,
                              reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    opt_type = config['optimizer']['type']
    if opt_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=float(config['optimizer'].get('momentum', 0)),
            weight_decay=config['optimizer']['weight_decay']
        )
    elif opt_type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
            betas=tuple(config['optimizer']['betas'])
        )
    else:
        raise NotImplementedError(
            f"Optimizer type '{opt_type}' is not recognized (use 'SGD' or 'AdamW')."
        )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr']
    )

    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    best_val_epoch = 0

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(start_epoch, num_epochs):
        # ==================== Training ====================
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # main + auxiliary logits from hybrid model
            logits_main, logits_cnn_aux, logits_tr_aux = model(
                inputs,
                return_embeddings=False,
                return_attention=False,
                return_aux=True
            )

            loss_main = criterion(logits_main, labels)
            loss_cnn_aux = criterion(logits_cnn_aux, labels)
            loss_tr_aux = criterion(logits_tr_aux, labels)

            loss = loss_main + aux_weight * (loss_cnn_aux + loss_tr_aux)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training']['gradient_clip']
            )
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits_main, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ==================== Validation ====================
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits_main = model(inputs)
                loss = criterion(logits_main, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(logits_main, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # ==================== Test (per-epoch monitoring) ====================
        test_running_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits_main = model(inputs)
                loss = criterion(logits_main, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(logits_main, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss = test_running_loss / len(test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # ==================== Logging & Scheduler ====================
        print(
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}"
        )

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience increased to {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! (Best val @ epoch {best_val_epoch})")
            break

    out_dir = config['experiment']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_model_state, os.path.join(out_dir, 'best_hybrid_model.pth'))
    print(f"✓ Best model saved to {out_dir}/best_hybrid_model.pth")

    plot_training_curves(
        train_losses, val_losses, test_losses,
        train_accuracies, val_accuracies, test_accuracies, config
    )

    return (train_losses, val_losses, test_losses,
            train_accuracies, val_accuracies, test_accuracies,
            best_val_epoch)


def final_test_evaluation(model, test_loader, config):
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (best model)")
    print("=" * 70)

    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits_main = model(images)
            loss = criterion(logits_main, labels)
            running_loss += loss.item()
            _, predicted = torch.max(logits_main, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\nFinal Test Set Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.title("Test Confusion Matrix (%)")
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(config['experiment']['output_dir'], 'confusion_matrix_hybrid.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {cm_path}/confusion_matrix_hybrid.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    print("\n" + "=" * 70)
    print("HYBRID CNN-TRANSFORMER EMOTION RECOGNITION")
    print("=" * 70)

    train_loader, val_loader, test_loader = get_data_loaders(
        config, use_weighted_sampler=False
    )

    print("\nInitializing Hybrid Model...")
    model = HybridEmotionRecognition(
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        pretrained_swin=config['model']['pretrained_swin'],
        use_gradient_checkpointing=False,
        aggregation=config['model']['aggregation']
    ).to(device)

    print_model_summary(model)

    start_epoch = 0
    (train_losses, val_losses, test_losses,
     train_accs, val_accs, test_accs,
     best_epoch) = train_model(
        model, train_loader, val_loader, test_loader, config,
        start_epoch=start_epoch
    )

    out_dir = config['experiment']['output_dir']
    best_model_path = os.path.join(out_dir, 'best_hybrid_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_test_evaluation(model, test_loader, config)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
