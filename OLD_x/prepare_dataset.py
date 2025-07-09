
import os
import glob
import random
import shutil

# Path to the dataset
data_dir = '/home/utka/proj/semacomm/Eval_data'
# Percentage of data to use for validation
val_split = 0.2

# Create train and val directories
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all image paths
image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
# Shuffle the paths
random.shuffle(image_paths)

# Calculate the split point
split_point = int(len(image_paths) * (1 - val_split))

# Move files
for i, path in enumerate(image_paths):
    if i < split_point:
        shutil.move(path, os.path.join(train_dir, os.path.basename(path)))
    else:
        shutil.move(path, os.path.join(val_dir, os.path.basename(path)))

print(f"Moved {split_point} images to {train_dir}")
print(f"Moved {len(image_paths) - split_point} images to {val_dir}")
