import pandas as pd
from pathlib import Path

# 📌 Replace this with your actual file path
input_csv = Path("rafdb/valid_labels.csv")
output_csv = Path("rafdb/new_labels/valid_labels.csv")

# 🗺️ Label map: RAF-DB ➝ FER2013 format
raf_to_fer_map = {
    6: 0,  # Angry ➝ 0
    3: 1,  # Disgust ➝ 1
    2: 2,  # Fear ➝ 2
    4: 3,  # Happy ➝ 3
    5: 4,  # Sad ➝ 4
    1: 5,  # Surprise ➝ 5
    7: 6   # Neutral ➝ 6
}

try:
    # Check if input file exists
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    # 🧾 Load the CSV (assuming format: filename,label)
    df = pd.read_csv(input_csv)
    
    # Validate column exists
    if 'label' not in df.columns:
        raise ValueError(f"CSV must contain 'label' column. Found columns: {df.columns.tolist()}")
    
    # 🔁 Map the labels
    original_nulls = df['label'].isnull().sum()
    df['label'] = df['label'].map(raf_to_fer_map)
    mapped_nulls = df['label'].isnull().sum()
    
    # 🧪 Check mapping results
    if mapped_nulls > original_nulls:
        failed_mappings = mapped_nulls - original_nulls
        print(f"⚠️ Warning: {failed_mappings} label(s) could not be mapped. Check input CSV for invalid values.")
    
    # 💾 Create output directory and save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Done. Converted {len(df)} rows. Labels saved to: {output_csv}")

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
except ValueError as e:
    print(f"❌ Error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
