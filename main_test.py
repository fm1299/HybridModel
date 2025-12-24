# ================================================================
# MERGED HYBRID CNN–TRANSFORMER TRAINING SCRIPT
# Config-driven + strong evaluation (Accuracy, Precision, Recall, F1)
# Checkpoints removed, only best model saved at the end
# ================================================================

print("Starting Hybrid Model training script...")

# ==================== Imports ====================
import os
import copy
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import Counter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.HybridModel import HybridEmotionRecognition, print_model_summary
from get_dataset import Four4All

# ==================== Config & Reproducibility ====================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class_names = config['data']['class_names']
num_classes = config['model']['num_classes']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config['experiment']['seed'])

# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ==================== Data ====================
def build_transforms():
    aug = config['augmentation']
    train_tf = transforms.Compose([
        transforms.Resize((aug['img_size'], aug['img_size'])),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((aug['img_size'], aug['img_size'])),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf


def get_dataloaders():
    train_tf, eval_tf = build_transforms()

    train_ds = Four4All(config['data']['train_csv'], config['data']['train_dir'], transform=train_tf)
    val_ds   = Four4All(config['data']['val_csv'], config['data']['val_dir'], transform=eval_tf)
    test_ds  = Four4All(config['data']['test_csv'], config['data']['test_dir'], transform=eval_tf)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'], shuffle=True,
                              num_workers=config['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['data']['batch_size'], shuffle=False,
                            num_workers=config['data']['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config['data']['batch_size'], shuffle=False,
                             num_workers=config['data']['num_workers'], pin_memory=True)

    return train_loader, val_loader, test_loader


def compute_class_weights(loader):
    counts = torch.zeros(num_classes)
    for _, labels in loader:
        for l in labels:
            counts[l] += 1
    weights = counts.sum() / (num_classes * counts)
    return (weights / weights.sum() * num_classes).to(device)

# ==================== Training ====================
def train_model(model, train_loader, val_loader):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    patience = config['training']['patience']
    use_focal_loss = (config['loss']['type'] == "FocalLoss")
    focal_gamma = config['loss'].get('gamma', 2.0)

    class_weights = compute_class_weights(train_loader)

    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    opt_cfg = config['optimizer']

    if opt_cfg['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=float(config.get('momentum', 0)),
            weight_decay=config['weight_decay']
        )
    elif opt_cfg['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=tuple(config['betas'])
        )

    sch_cfg = config['scheduler']
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=sch_cfg['factor'],
        patience=sch_cfg['patience'],
        min_lr=sch_cfg['min_lr']
    )

    best_f1 = 0.0
    best_state = None
    patience_counter = 0
    best_model_state = None
    best_val_epoch = 0
    for epoch in range(config['training']['epochs']):
        # -------- Train --------
        model.train()
        tr_preds, tr_labels, tr_loss = [], [], 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item()
            tr_preds.extend(out.argmax(1).cpu().numpy())
            tr_labels.extend(y.cpu().numpy())

        train_loss = tr_loss / len(train_loader)
        train_acc = accuracy_score(tr_labels, tr_preds)
        train_f1 = f1_score(tr_labels, tr_preds, average='macro')

        # -------- Validation --------
        model.eval()
        val_preds, val_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: "
              f"Train Acc {train_acc:.4f} | Train F1 {train_f1:.4f} || "
              f"Val Acc {val_acc:.4f} | Val P {val_prec:.4f} R {val_rec:.4f} F1 {val_f1:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_val_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience increased to {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! (Best val @ epoch {best_val_epoch})")
            break

        model.load_state_dict(best_state)
        plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    return model

# ==================== Training Curves Plotting ====================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    if not config['evaluation'].get('plot_training_curves', True):
        return

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(config['experiment']['output_dir'], 'training_curves.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Training curves saved to {out_path}")

# ==================== Final Evaluation ====================
def evaluate(model, loader, split='Test'):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro')

    print(f"\n{split} Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title("Test Confusion Matrix (%) - Accuracy: {:.2f}%".format(acc * 100), fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(config['experiment']['output_dir'], 'confusion_matrix_hybrid.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {cm_path}/confusion_matrix_hybrid.png")
    plt.close()
    # cm = cm.astype('float') / cm.sum(axis=1)[:, None]

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
    #             xticklabels=class_names, yticklabels=class_names)
    # plt.title(f'{split} Confusion Matrix')
    # plt.tight_layout()
    # plt.savefig(os.path.join(config['experiment']['output_dir'], f'{split.lower()}_confusion_matrix.png'))
    # plt.close()


    

# ==================== Main ====================
if __name__ == '__main__':
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()

    model = HybridEmotionRecognition(
        num_classes=num_classes,
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        pretrained_swin=config['model']['pretrained_swin'],
        aggregation=config['model']['aggregation']
    ).to(device)

    print_model_summary(model)

    model = train_model(model, train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(config['experiment']['output_dir'], 'best_hybrid_model.pth'))
    print("✓ Best model saved")

    evaluate(model, test_loader, split='Test')
