print("Starting ResEmoteNet training script...")
print("Importing necessary libraries...")
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from models.HybridModel import HybridEmotionRecognition
from get_dataset import Four4All

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def get_data_loaders():
    # Transform the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_dataset = Four4All(csv_file='fer2013/data/train_labels.csv',
                            img_dir='fer2013/data/train', transform=transform)
    val_dataset = Four4All(csv_file='fer2013/data/valid_labels.csv', 
                        img_dir='fer2013/data/valid/', transform=transform)
    test_dataset = Four4All(csv_file='fer2013/data/test_labels.csv', 
                            img_dir='fer2013/data/test', transform=transform)
    
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_data_loaders()

# Load the dataset

# train_image, train_label = next(iter(train_loader))
# val_image, val_label = next(iter(val_loader))
# test_image, test_label = next(iter(test_loader))

# print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
# print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
# print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

def save_checkpoint(model, optimizer, epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies
    }, path)

def load_checkpoint(model, optimizer, path):
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    train_accuracies = checkpoint['train_accuracies']
    valid_accuracies = checkpoint['valid_accuracies']
    return model, optimizer, epoch, train_losses, valid_losses, train_accuracies, valid_accuracies


def train_model(model, train_loader, val_loader,start_epoch, num_epochs=80, checkpoint_path='fer2013/checkpoints/checkpoint_epoch_{}.pth'):
    # Hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    patience = 15
    best_val_acc = 0
    patience_counter = 0
    epoch_counter = start_epoch

    num_epochs = 80

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    if start_epoch > 0:
        model, optimizer, _, train_losses, val_losses, train_accuracies, val_accuracies = load_checkpoint(model, optimizer, checkpoint_path.format(start_epoch))

    # Start training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
        scheduler.step()
        print(f"Scheduler stepped. Current LR: {scheduler.get_last_lr()[0]}")
        epoch_counter += 1

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch + 1, train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_path.format(epoch + 1))
        print(f"Checkpoint saved at epoch {epoch + 1}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epochs.")
        
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

    df = pd.DataFrame({
        'Epoch': range(1, epoch_counter+1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies
    })
    df.to_csv('result_fer2013.csv', index=False)

    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, average='weighted')
    rec = recall_score(true, preds, average='weighted')
    f1 = f1_score(true, preds, average='weighted')
    cm = confusion_matrix(true, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Convert to percentages

    print(f'\nTest Results:\nAccuracy: {acc*100:.2f}%\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}')

    # Emotion class labels (modify if you're using a different dataset)
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title(f'FER2013 Test Acc. {acc*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig('fer2013/confusion_matrix.png')
    plt.show()
    plt.close()

    return acc, prec, rec, f1, cm


def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(valid_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('fer2013/training_curves.png')
    plt.show()
    plt.close()



if __name__ == "__main__":
    os.makedirs('fer2013/checkpoints', exist_ok=True)
    # Initialize the model
    model = HybridEmotionRecognition(
        num_classes=7,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
        pretrained_swin=True
    ).to(device)

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    # Train the model (set start_epoch > 0 to resume from checkpoint)
    start_epoch = 0  # Change to resume from a specific epoch (e.g., 5)
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader,start_epoch, num_epochs=80,
        checkpoint_path='fer2013/checkpoints/checkpoint_epoch_{}.pth'
    )

    # Evaluate the model on test set
    acc, prec, rec, f1, cm = evaluate_model(model, test_loader)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    # Save the final model
    torch.save(model.state_dict(), 'fer2013/resemotenet_fer2013.pth')