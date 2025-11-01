import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_train_dir = 'raf_db/data/train'
new_train_dir = 'raf_db/data/split/train'
val_dir = 'raf_db/data/val'

# Create folders
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Load original CSV
df = pd.read_csv('raf_db/data/train_labels.csv')

# Split into 80% train, 20% val
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Save new CSVs
train_df.to_csv('raf_db/data/split/train_labels.csv', index=False)
val_df.to_csv('raf_db/data/val_labels.csv', index=False)

# Move images
for _, row in train_df.iterrows():
    shutil.copy(os.path.join(original_train_dir, row['image']),
                os.path.join(new_train_dir, row['image']))

for _, row in val_df.iterrows():
    shutil.copy(os.path.join(original_train_dir, row['image']),
                os.path.join(val_dir, row['image']))

print("✅ Split done. Train and validation sets created.")
