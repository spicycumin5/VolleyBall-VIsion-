#!/usr/bin/env python3

import argparse
import os
import cv2
import glob
from ultralytics import YOLO

parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")

args = parser.parse_args()
input_folder = args.input
output_folder = args.output

model = YOLO('yolo26x.pt')

video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

for video_path in video_files:
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"Processing {video_name}...")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # Run tracking on the video
    results = model.track(
        source=video_path,
        tracker="botsort.yaml",
        stream=True,
        classes=[0],
        persist=True,
        verbose=False,
        imgsz=args.imgsz,
    )
    
    for r in results:
        frame = r.orig_img
        frame_count += 1
        
        # Skip if no objects detected or no IDs assigned yet
        if r.boxes is None or r.boxes.id is None:
            continue
            
        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Prevent out-of-bounds cropping
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            
            # Skip tiny or invalid crops
            if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 20:
                continue
                
            # Create a folder for this specific Track ID from this specific video
            track_dir = os.path.join(output_folder, f"{video_name}_ID_{track_id}")
            os.makedirs(track_dir, exist_ok=True)
            
            # Save the crop (saving every 5th frame to avoid millions of identical images)
            if frame_count % 5 == 0:
                save_path = os.path.join(track_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(save_path, crop)

    cap.release()
print("Extraction complete!")
