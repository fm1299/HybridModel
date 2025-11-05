import os
import pandas as pd
import numpy as np
from PIL import Image

# Load FER2013
df = pd.read_csv('fer2013.csv')
dir_name = 'fer2013'
os.makedirs(dir_name, exist_ok=True)
print("Data loaded successfully.")
# Map usage
usage_map = {
    'Training': 'train',
    'PublicTest': 'valid',
    'PrivateTest': 'test'
}

for index, row in df.iterrows():
    emotion = row['emotion']
    pixels = np.array(row['pixels'].split(), dtype='uint8')
    img = pixels.reshape(48, 48)
    usage = usage_map[row['Usage']]

    # Create directories if not exist
    out_dir = f'{dir_name}/data/{usage}/'
    os.makedirs(out_dir, exist_ok=True)

    # Save image
    img_name = f'{usage}_{index}.png'
    im = Image.fromarray(img)
    im = im.convert('RGB')  # convert to 3 channels
    im.save(out_dir + img_name)

    # Append to CSV
    with open(f'{dir_name}/data/{usage}_labels.csv', 'a') as f:
        if index == 0 or os.stat(f'{dir_name}/data/{usage}_labels.csv').st_size == 0:
            f.write('image,label\n')
        f.write(f'{img_name},{emotion}\n')

print('Done!')