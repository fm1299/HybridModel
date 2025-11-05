"""
Script to prepare RAF-DB dataset with validation split
Handles RAF-DB structure where images are organized in numbered emotion folders
"""

import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

def get_emotion_folder(label):
    """
    Map emotion label to folder number
    RAF-DB structure: folders 1-7 correspond to emotions
    """
    # RAF-DB folder mapping (adjust if your dataset uses different mapping)
    # Folders: 1=Surprise, 2=Fear, 3=Disgust, 4=Happy, 5=Sad, 6=Angry, 7=Neutral
    return str(label)


def create_rafdb_splits_with_subfolders(
    original_train_dir='rafdb/train',
    original_test_dir='rafdb/test',
    output_base_dir='rafdb_prepared',
    val_split=0.15,
    random_state=42
):
    """
    Split RAF-DB training data into train and validation sets
    Handles structure where images are in emotion subfolders (1, 2, 3, ..., 7)
    
    Args:
        original_train_dir: Path to original training images folder (contains subfolders 1-7)
        original_test_dir: Path to test images folder (contains subfolders 1-7)
        output_base_dir: Output directory for prepared dataset
        val_split: Fraction of training data for validation (0.15 = 15%)
        random_state: Random seed for reproducibility
    """
    
    print("="*70)
    print("RAF-DB Dataset Preparation (with emotion subfolders)")
    print("="*70)
    
    # Create output directories
    train_dir = os.path.join(output_base_dir, 'train')
    valid_dir = os.path.join(output_base_dir, 'valid')
    test_dir = os.path.join(output_base_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Emotion labels mapping
    emotion_names = {
        1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happy',
        5: 'Sad', 6: 'Angry', 7: 'Neutral'
    }
    
    # Step 1: Scan training folder and collect all images
    print("\n1. Scanning training folder structure...")
    train_data = []
    
    for emotion_folder in sorted(os.listdir(original_train_dir)):
        folder_path = os.path.join(original_train_dir, emotion_folder)
        
        # Skip if not a directory or not a number
        if not os.path.isdir(folder_path) or not emotion_folder.isdigit():
            continue
        
        label = int(emotion_folder)
        emotion = emotion_names.get(label, f'Unknown_{label}')
        
        # Get all images in this emotion folder
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   {emotion:10s} (folder {emotion_folder}): {len(images):5d} images")
        
        # Add to training data list
        for img_name in images:
            train_data.append({
                'filename': img_name,
                'label': label,
                'emotion_folder': emotion_folder,
                'full_path': os.path.join(folder_path, img_name)
            })
    
    train_df = pd.DataFrame(train_data)
    print(f"\n   Total training images found: {len(train_df)}")
    
    if len(train_df) == 0:
        print("\n❌ ERROR: No images found in training folder!")
        print(f"   Check if images exist in: {original_train_dir}/1, {original_train_dir}/2, etc.")
        return None
    
    # Step 2: Perform stratified split
    print(f"\n2. Splitting training data (train: {1-val_split:.0%}, valid: {val_split:.0%})...")
    train_split, valid_split = train_test_split(
        train_df,
        test_size=val_split,
        stratify=train_df['label'],  # Maintain class distribution
        random_state=random_state
    )
    
    print(f"   New training samples: {len(train_split)}")
    print(f"   Validation samples: {len(valid_split)}")
    
    # Step 3: Copy training images
    print("\n3. Copying training images...")
    train_labels = []
    for idx, row in tqdm(train_split.iterrows(), total=len(train_split), desc="   Train"):
        src = row['full_path']
        dst = os.path.join(train_dir, row['filename'])
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            train_labels.append({
                'filename': row['filename'],
                'label': row['label']
            })
        else:
            print(f"   Warning: Image not found: {src}")
    
    # Step 4: Copy validation images
    print("\n4. Copying validation images...")
    valid_labels = []
    for idx, row in tqdm(valid_split.iterrows(), total=len(valid_split), desc="   Valid"):
        src = row['full_path']
        dst = os.path.join(valid_dir, row['filename'])
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            valid_labels.append({
                'filename': row['filename'],
                'label': row['label']
            })
        else:
            print(f"   Warning: Image not found: {src}")
    
    # Step 5: Process test folder
    print("\n5. Processing test images...")
    test_labels = []
    
    for emotion_folder in sorted(os.listdir(original_test_dir)):
        folder_path = os.path.join(original_test_dir, emotion_folder)
        
        if not os.path.isdir(folder_path) or not emotion_folder.isdigit():
            continue
        
        label = int(emotion_folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(images, desc=f"   Test (folder {emotion_folder})"):
            src = os.path.join(folder_path, img_name)
            dst = os.path.join(test_dir, img_name)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                test_labels.append({
                    'filename': img_name,
                    'label': label
                })
    
    print(f"\n   Test samples: {len(test_labels)}")
    
    # Step 6: Save CSV files
    print("\n6. Saving label CSV files...")
    train_csv = pd.DataFrame(train_labels)
    valid_csv = pd.DataFrame(valid_labels)
    test_csv = pd.DataFrame(test_labels)
    
    train_csv_path = os.path.join(output_base_dir, 'train_labels.csv')
    valid_csv_path = os.path.join(output_base_dir, 'valid_labels.csv')
    test_csv_path = os.path.join(output_base_dir, 'test_labels.csv')
    
    train_csv.to_csv(train_csv_path, index=False)
    valid_csv.to_csv(valid_csv_path, index=False)
    test_csv.to_csv(test_csv_path, index=False)
    
    print(f"   ✓ {train_csv_path} ({len(train_csv)} samples)")
    print(f"   ✓ {valid_csv_path} ({len(valid_csv)} samples)")
    print(f"   ✓ {test_csv_path} ({len(test_csv)} samples)")
    
    # Print final summary
    print("\n" + "="*70)
    print("Dataset Preparation Complete!")
    print("="*70)
    print(f"\nNew flat structure:")
    print(f"  {output_base_dir}/")
    print(f"    ├── train/          ({len(train_csv)} images - all in one folder)")
    print(f"    ├── valid/          ({len(valid_csv)} images - all in one folder)")
    print(f"    ├── test/           ({len(test_csv)} images - all in one folder)")
    print(f"    ├── train_labels.csv")
    print(f"    ├── valid_labels.csv")
    print(f"    └── test_labels.csv")
    
    # Print class distribution in splits
    print(f"\n  Class distribution:")
    print(f"    {'Emotion':<12} {'Train':<8} {'Valid':<8} {'Test':<8}")
    print("    " + "-"*40)
    for label in sorted(emotion_names.keys()):
        emotion = emotion_names[label]
        train_count = (train_csv['label'] == label).sum()
        valid_count = (valid_csv['label'] == label).sum()
        test_count = (test_csv['label'] == label).sum()
        print(f"    {emotion:<12} {train_count:<8} {valid_count:<8} {test_count:<8}")
    
    return train_csv, valid_csv, test_csv


def verify_dataset(output_base_dir='rafdb_prepared'):
    """
    Verify that images were copied correctly
    """
    print("\n" + "="*70)
    print("Verifying Dataset")
    print("="*70)
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(output_base_dir, split)
        csv_file = os.path.join(output_base_dir, f'{split}_labels.csv')
        
        # Count images in folder
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            img_count = 0
        
        # Count rows in CSV
        if os.path.exists(csv_file):
            csv_count = len(pd.read_csv(csv_file))
        else:
            csv_count = 0
        
        status = "✓" if img_count == csv_count else "❌"
        print(f"  {split.capitalize():5s}: {img_count:5d} images, {csv_count:5d} CSV rows {status}")
        
        if img_count != csv_count:
            print(f"    Warning: Mismatch between images and CSV for {split}!")


if __name__ == "__main__":
    # IMPORTANT: Adjust these paths to match your RAF-DB location
    train_splits = create_rafdb_splits_with_subfolders(
        original_train_dir='rafdb/train',      # Folder with subfolders 1-7
        original_test_dir='rafdb/test',        # Folder with subfolders 1-7
        output_base_dir='rafdb_prepared',      # Output folder
        val_split=0.15,                        # 15% for validation
        random_state=42
    )
    
    # Verify the split was successful
    if train_splits is not None:
        verify_dataset('rafdb_prepared')
        print("\n✓ Dataset preparation successful!")
        print("\nYou can now use 'rafdb_prepared' folder for training.")
    else:
        print("\n❌ Dataset preparation failed. Check the error messages above.")
