#!/usr/bin/env python3
"""I also did this with Chat."""

#!/usr/bin/env python3

#!/usr/bin/env python3
"""ReID + Pose + Motion Feature Extractor for BoxMOT/StrongSORT."""

import os
import cv2
import random
import numpy as np
from pathlib import Path
import torch
import torchreid
from torchreid.utils import FeatureExtractor

# ----------------------------
# Pose Utilities
# ----------------------------
def pose_to_feature(keypoints):
    """
    Normalize keypoints (17 joints) to a fixed-size vector for similarity.
    keypoints: np.array of shape (17, 2) or (17,3)
    """
    kp = keypoints[:, :2]
    center = kp.mean(axis=0)
    kp -= center
    scale = np.linalg.norm(kp[5] - kp[6]) + 1e-6  # shoulder width
    kp /= scale
    return kp.flatten()  # 34-dim vector

def pose_similarity(p1, p2):
    """Cosine similarity between two pose vectors"""
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-6)

# ----------------------------
# Motion Utilities
# ----------------------------
def motion_similarity(det_box, pred_box):
    """IoU-based motion similarity"""
    xA = max(det_box[0], pred_box[0])
    yA = max(det_box[1], pred_box[1])
    xB = min(det_box[2], pred_box[2])
    yB = min(det_box[3], pred_box[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
    boxB_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    iou = inter_area / (boxA_area + boxB_area - inter_area + 1e-6)
    return iou

# ----------------------------
# Dataset Preparation
# ----------------------------
def prepare_market1501_dataset(mot_txt_path, images_dir, output_dir):
    """
    Convert MOT annotations into Market-1501 style dataset for TorchReID.
    """
    print(f"--- Preparing ReID Dataset ---")
    
    market_dir = os.path.join(os.path.abspath(output_dir), "market1501")
    train_dir = os.path.join(market_dir, "bounding_box_train")
    query_dir = os.path.join(market_dir, "query")
    gallery_dir = os.path.join(market_dir, "bounding_box_test")
    
    for folder in [train_dir, query_dir, gallery_dir]:
        os.makedirs(folder, exist_ok=True)

    # Parse MOT gt.txt
    frame_data = {}
    with open(mot_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            if frame_id not in frame_data:
                frame_data[frame_id] = []
            frame_data[frame_id].append((track_id, int(x), int(y), int(w), int(h)))

    print(f"Found annotations for {len(frame_data)} frames. Cropping players...")

    crop_count = 0
    for frame_id, boxes in frame_data.items():
        img_name = f"frame_{frame_id:06d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        for (track_id, x, y, w, h) in boxes:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            if x2 <= x1 or y2 <= y1: continue
            crop = cv2.resize(img[y1:y2, x1:x2], (128, 256))

            # Fake train/val split
            rand_val = random.random()
            if rand_val < 0.80:
                save_dir = train_dir
                cam = 1
            elif rand_val < 0.90:
                save_dir = query_dir
                cam = 2
            else:
                save_dir = gallery_dir
                cam = 3

            filename = f"{track_id:04d}_c{cam}s1_{frame_id:06d}_00.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), crop)
            crop_count += 1

    print(f"Generated {crop_count} ReID crops!")
    return market_dir

# ----------------------------
# Train ReID Model
# ----------------------------
def train_reid_model(output_dir, epochs=60, gpus=[0]):
    print("\n--- Training ReID Model ---")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    datamanager = torchreid.data.ImageDataManager(
        root=os.path.abspath(output_dir),
        sources='market1501',
        height=256,
        width=128,
        batch_size_train=64,
        batch_size_test=64,
        transforms=['random_flip', 'random_erase']
    )

    model = torchreid.models.build_model(
        name='osnet_ibn_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='step', stepsize=20)

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    save_dir = os.path.abspath("volleyball_reid_model")
    engine.run(
        save_dir=save_dir,
        max_epoch=epochs,
        eval_freq=10,
        print_freq=20,
        test_only=False
    )

    print(f"\n--- Training Complete ---\nWeights saved to {save_dir}/model.pth.tar-{epochs}")
    return model

# ----------------------------
# Fuse Appearance + Pose + Motion
# ----------------------------
def fuse_features(appearance_feat, pose_feat, motion_score, weights=(0.5, 0.3, 0.2)):
    """
    Combine appearance, pose, and motion features into a single vector.
    motion_score is scalar IoU or distance metric.
    """
    a, p, m = weights
    # normalize appearance and pose
    appearance_feat = appearance_feat / (np.linalg.norm(appearance_feat) + 1e-6)
    pose_feat = pose_feat / (np.linalg.norm(pose_feat) + 1e-6)
    fused = np.concatenate([a * appearance_feat, p * pose_feat, [m * motion_score]])
    return fused

# ----------------------------
# Main
# ----------------------------
def main():
    gpus = [0]  # adjust
    mot_txt_file = "data/mot_tracked_people/gt/gt.txt"
    images_folder = "data/mot_tracked_people/one_play_images"
    dataset_output = "data/mot_tracked_people/reid_ready_dataset"

    # Step 1: Prepare dataset
    market_dir = prepare_market1501_dataset(
        mot_txt_path=mot_txt_file,
        images_dir=images_folder,
        output_dir=dataset_output
    )

    # Step 2: Train ReID model
    model = train_reid_model(output_dir=market_dir, epochs=60, gpus=gpus)

    print("\n--- ReID + Pose + Motion Extractor Ready ---")

if __name__ == "__main__":
    main()
