#!/usr/bin/env python3
"""I just asked Chat to do this."""
#!/usr/bin/env python3

import os
import shutil
import random
import re
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. THE UNIFIED MASTER CLASS LIST
# ==========================================
MASTER_CLASSES = {
    0: "ball",
    1: "player",
    2: "block",
    3: "dig",
    4: "serve",
    5: "set",
    6: "spike"
}

# ==========================================
# 2. DATASET CONFIGURATIONS
# ==========================================
# Add as many datasets here as you want. The script will merge them all!
DATASETS = [
    {
        "prefix": "custom", # Prefix added to filenames to prevent overwriting
        "images_dir": "data/yolo_people_and_ball/one_play_images",
        "labels_dir": "data/yolo_people_and_ball/obj_train_data",
        "has_cvat_offset": True, # Applies your -1 math fix
        # How do the original classes map to the MASTER_CLASSES?
        # Your old dataset: 0 was person, 1 was ball.
        "class_map": {
            0: 1, # map original 0 to Master 1 (player)
            1: 0  # map original 1 to Master 0 (ball)
        }
    },
    {
        "prefix": "volleyvision",
        "images_dir": "data/volleyvision_actions/images", # Update with your real path
        "labels_dir": "data/volleyvision_actions/labels", # Update with your real path
        "has_cvat_offset": False, # VolleyVision doesn't need the CVAT fix
        # Assuming VolleyVision used: 0:block, 1:dig, 2:serve, 3:set, 4:spike
        "class_map": {
            0: 2, 
            1: 3, 
            2: 4, 
            3: 5, 
            4: 6  
        }
    }
]

def prepare_unified_dataset(datasets_config, output_dir, split_ratio=0.8):
    """
    Merges multiple YOLO datasets into one, remaps class IDs, prefixes filenames 
    to prevent collisions, and generates a unified data.yaml.
    """
    print(f"--- Building Unified Dataset in '{output_dir}' ---")
    
    # 1. Create the strict YOLO folder hierarchy
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    total_train = 0
    total_val = 0

    # 2. Process each dataset according to its config
    for config in datasets_config:
        prefix = config["prefix"]
        img_dir = config["images_dir"]
        lbl_dir = config["labels_dir"]
        class_map = config["class_map"]
        cvat_offset = config["has_cvat_offset"]

        print(f"\nProcessing dataset: [{prefix}]")
        if not os.path.exists(img_dir):
            print(f"  -> WARNING: Path {img_dir} not found! Skipping.")
            continue

        all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_images)

        split_idx = int(len(all_images) * split_ratio)
        splits = {
            'train': all_images[:split_idx],
            'val': all_images[split_idx:]
        }

        for split_type, images in splits.items():
            valid_count = 0
            for img_name in images:
                # Find the corresponding label file
                if cvat_offset:
                    match = re.search(r'frame_(\d+)', img_name)
                    if match:
                        txt_num = int(match.group(1)) - 1
                        src_txt_name = f"frame_{txt_num:06d}.txt"
                    else:
                        src_txt_name = os.path.splitext(img_name)[0] + '.txt'
                else:
                    src_txt_name = os.path.splitext(img_name)[0] + '.txt'
                
                src_img = os.path.join(img_dir, img_name)
                src_txt = os.path.join(lbl_dir, src_txt_name)
                
                if os.path.exists(src_txt):
                    # Define new names with prefix to prevent collisions
                    dest_img_name = f"{prefix}_{img_name}"
                    dest_txt_name = f"{prefix}_{os.path.splitext(img_name)[0]}.txt"
                    
                    dest_img = os.path.join(output_dir, 'images', split_type, dest_img_name)
                    dest_txt = os.path.join(output_dir, 'labels', split_type, dest_txt_name)
                    
                    # Copy Image
                    shutil.copy(src_img, dest_img)
                    
                    # REMAP AND COPY LABELS
                    with open(src_txt, 'r') as f_in, open(dest_txt, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                original_class = int(parts[0])
                                # Only keep it if it exists in our map (filters out garbage classes)
                                if original_class in class_map:
                                    new_class = class_map[original_class]
                                    parts[0] = str(new_class)
                                    f_out.write(" ".join(parts) + "\n")
                    
                    valid_count += 1
            
            print(f"  -> {split_type.capitalize()}: {valid_count} files remapped and copied.")
            if split_type == 'train': total_train += valid_count
            else: total_val += valid_count

    # 3. Automatically generate the Unified data.yaml file
    abs_output_dir = Path(output_dir).resolve().as_posix() 
    
    yaml_lines = [
        f"path: {abs_output_dir}",
        "train: images/train",
        "val: images/val",
        "",
        "# Unified Classes",
        "names:"
    ]
    for cid, cname in sorted(MASTER_CLASSES.items()):
        yaml_lines.append(f"  {cid}: {cname}")
        
    yaml_content = "\n".join(yaml_lines)
    yaml_path = os.path.join(output_dir, 'data.yaml')
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"\nUnified Dataset ready! Total Train: {total_train} | Total Val: {total_val}")
    print(f"Generated YAML at: {yaml_path}\n")
    
    return yaml_path


def train_yolo_model(yaml_path, epochs=50, imgsz=1920):
    """
    Initializes a pre-trained YOLO model and fine-tunes it on the new dataset.
    """
    print("--- Starting YOLO Training ---")
    model = YOLO("yolo26x.pt")
    project_path = os.path.abspath("volleyball_model")

    results = model.train(
        data=os.path.abspath(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=2,
        device=0, # Assuming you meant GPU 0, adjust if you actually have 8 GPUs (device=7)
        project=project_path,
        box=10.0,
    )
    
    print("\n--- Training Complete! ---")
    print("Your new weights are saved in: volleyball_model/train/weights/best.pt")


def main():
    ready_dataset_folder = "data/yolo_unified_dataset"

    # Step 1: Prepare the unified data
    yaml_file_path = prepare_unified_dataset(
        datasets_config=DATASETS,
        output_dir=ready_dataset_folder,
        split_ratio=0.8 
    )

    # Step 2: Train the model
    # Note: If VolleyVision images are smaller, you might want to drop imgsz to 1280 or 640
    train_yolo_model(yaml_path=yaml_file_path, epochs=50, imgsz=1920)


if __name__ == "__main__":
    main()
