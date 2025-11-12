import pandas as pd

# 📌 Replace this with your actual file path
input_csv = "rafdb/train_labels.csv"
output_csv = "rafdb/new_labels/train_labels.csv"

# 🗺️ Label map: RAF-DB ➝ FER2013 format
raf_to_fer_map = {
    1: 5,  # Surprise ➝ 5
    2: 2,  # Fear ➝ 2
    3: 1,  # Disgust ➝ 1
    4: 3,  # Happy ➝ 3
    5: 4,  # Sad ➝ 4
    6: 0,  # Angry ➝ 0
    7: 6   # Neutral ➝ 6
}

# 🧾 Load the CSV (assuming format: filename,label)
df = pd.read_csv(input_csv)

# 🔁 Map the labels
df['label'] = df['label'].map(raf_to_fer_map)

# 🧪 Optional: check if any NaNs appeared
if df['label'].isnull().any():
    print("⚠️ Warning: Some labels could not be mapped. Check the CSV for invalid values.")

# 💾 Save the new CSV
df.to_csv(output_csv, index=False)

print(f"✅ Done. Converted labels saved to: {output_csv}")
