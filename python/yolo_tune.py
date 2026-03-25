#!/usr/bin/env python3
"""I just asked Chat to do this."""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

def prepare_yolo_dataset(images_dir, labels_dir, output_dir, split_ratio=0.8):
    """
    Takes a folder of images and a folder of text labels, matches them up,
    splits them into Train/Val sets, and builds the YOLO folder structure.
    Also automatically generates the required data.yaml file.
    """
    print(f"--- Preparing Dataset in '{output_dir}' ---")
    
    # 1. Create the strict YOLO folder hierarchy
    folders_to_create = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'val')
    ]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    # 2. Grab all images and shuffle them for a random split
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)

    # 3. Calculate the split index
    split_idx = int(len(all_images) * split_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Found {len(all_images)} total images.")
    print(f"Allocating {len(train_images)} to Training and {len(val_images)} to Validation...")

    # 4. Helper function to copy files and ensure labels exist
    def copy_files(image_list, split_type):
        valid_count = 0
        for img_name in image_list:
            # Swap the .jpg extension for .txt to find the matching label
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            
            src_img = os.path.join(images_dir, img_name)
            src_txt = os.path.join(labels_dir, txt_name)
            
            # Only copy if the image actually has a corresponding text file!
            if os.path.exists(src_txt):
                shutil.copy(src_img, os.path.join(output_dir, 'images', split_type, img_name))
                shutil.copy(src_txt, os.path.join(output_dir, 'labels', split_type, txt_name))
                valid_count += 1
            else:
                print(f"Warning: No label found for {img_name}. Skipping.")
        return valid_count

    train_count = copy_files(train_images, 'train')
    val_count = copy_files(val_images, 'val')

    # 5. Automatically generate the data.yaml file
    # We use absolute paths to prevent YOLO from getting confused about where the data is
    abs_output_dir = Path(output_dir).resolve().as_posix() 
    
    yaml_content = f"""path: {abs_output_dir}
train: images/train
val: images/val

# Classes
names:
  0: player
  1: ball
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Dataset ready! Copied {train_count} train pairs and {val_count} val pairs.")
    print(f"Generated YAML at: {yaml_path}\n")
    
    return yaml_path


def train_yolo_model(yaml_path, epochs=50, imgsz=1920):
    """
    Initializes a pre-trained YOLO model and fine-tunes it on the new dataset.
    """
    print("--- Starting YOLO Training ---")
    
    # Load the lightweight, pre-trained "Small" model
    model = YOLO("yolo26x.pt")
    project_path = os.path.abspath("volleyball_model")

    # os.makedirs(project_path, exist_ok=True)
    # os.makedirs(os.path.join(project_path, "custom_run"), exist_ok=True)
    # Start training
    results = model.train(
        data=os.path.abspath(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=1,
        workers=64,
        # device=[0, 1, 2, 3],
        device=0,
        project=project_path,
        # name="custom_run",
        # exist_ok=True
    )
    
    print("\n--- Training Complete! ---")
    print("Your new weights are saved in: volleyball_model/custom_run/weights/best.pt")


def main():
    # Define your raw input folders
    raw_images_folder = "data/yolo_people_and_ball/one_play_images"
    raw_labels_folder = "data/yolo_people_and_ball/obj_train_data"
    
    # Define where the formatted dataset should be built
    ready_dataset_folder = "data/yolo_people_and_ball/dataset"

    # Step 1: Prepare the data
    yaml_file_path = prepare_yolo_dataset(
        images_dir=raw_images_folder,
        labels_dir=raw_labels_folder,
        output_dir=ready_dataset_folder,
        split_ratio=0.8 # 80% for training, 20% for validation
    )

    # Step 2: Train the model
    train_yolo_model(yaml_path=yaml_file_path, epochs=50, imgsz=1920)


if __name__ == "__main__":
    main()
