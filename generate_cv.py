import os
import csv
from pathlib import Path

# FER2013 label mapping
class_to_idx = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

def make_csv_from_folders(split_dir, output_csv):
    rows = []
    for class_name, label in class_to_idx.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                rows.append([fname, label])

    with open(output_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {output_csv}")

# Example usage:
if __name__ == "__main__":
    make_csv_from_folders('./RAFDB_balanced/train', 'train_labels.csv')
    make_csv_from_folders('./RAFDB_balanced/val', 'val_labels.csv')
    make_csv_from_folders('./RAFDB_balanced/test', 'test_labels.csv')
