print("Starting Hybrid Model training script...")
print("Importing necessary libraries...")

import torch
import pandas as pd
import numpy as np
import os
import random
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = Four4All(
        csv_file=config['data']['train_csv'],
        img_dir=config['data']['train_dir'],
        transform=transform
    )
    val_dataset = Four4All(
        csv_file=config['data']['val_csv'],
        img_dir=config['data']['val_dir'],
        transform=transform
    )
    test_dataset = Four4All(
        csv_file=config['data']['test_csv'],
        img_dir=config['data']['test_dir'],
        transform=transform
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

# ==================== Main Training Function ====================
def train_model(model, train_loader, val_loader, test_loader, config, start_epoch=0):
    patience = config['training']['patience']
    num_epochs = config['training']['epochs']
    # use_focal_loss = (config['loss']['type'] == "FocalLoss")
    # focal_gamma = config['loss'].get('gamma', 2.0)

    class_weights = compute_class_weights(train_loader, num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'])
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=config['optimizer']['lr'],
    #     weight_decay=config['optimizer']['weight_decay'],
    #     betas=tuple(config['optimizer']['betas'])
    # )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'], min_lr=config['scheduler']['min_lr']
    )
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    best_val_epoch = 0
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        # Test
        test_running_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_loss = test_running_loss / len(test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        # Print
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! (Best val @ epoch {best_val_epoch})")
            break
    out_dir = config['experiment']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_model_state, os.path.join(out_dir, 'best_hybrid_model.pth'))
    print(f"✓ Best model saved to {out_dir}/best_hybrid_model.pth")
    return train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, best_val_epoch

# ==================== Final Test Evaluation ====================
def final_test_evaluation(model, test_loader, config):
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION (best model)")
    print("="*70)
    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
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
    # --- Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(config['experiment']['output_dir'], 'confusion_matrix_hybrid.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")

# ==================== Main Execution ====================
if __name__ == "__main__":
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
        use_gradient_checkpointing=False,
        aggregation=config['model']['aggregation']
    ).to(device)
    print_model_summary(model)
    start_epoch = 0
    train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, best_epoch = train_model(
        model, train_loader, val_loader, test_loader, config, start_epoch=start_epoch
    )
    out_dir = config['experiment']['output_dir']
    # Evaluation of best model on test set (full metrics + confusion matrix)
    best_model_path = os.path.join(out_dir, 'best_hybrid_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    final_test_evaluation(model, test_loader, config)
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
