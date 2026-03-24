#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")
args = parser.parse_args()

def convert_to_market1501(source_dir, dest_dir):
    """
    Converts folders of player crops into FastReID Market1501 format.
    source_dir: Directory containing folders like 'Player_1', 'Player_2'
    dest_dir: Where to save the formatted dataset
    """
    # Create the required Market1501 directory structure
    train_dir = os.path.join(dest_dir, 'bounding_box_train')
    query_dir = os.path.join(dest_dir, 'query')
    test_dir = os.path.join(dest_dir, 'bounding_box_test')

    for d in [train_dir, query_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    source_path = Path(source_dir)
    player_folders = [f for f in source_path.iterdir() if f.is_dir()]

    for pid, folder in enumerate(player_folders, start=1):
        # Format the ID to be 4 digits (e.g., 0001)
        str_pid = f"{pid:04d}"

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        for frame_num, img_path in enumerate(images, start=1):
            # Market1501 Naming Convention: [ID]_c[Camera]s[Sequence]_[Frame].jpg
            # We assume 1 camera (c1) and 1 sequence (s1) for simplicity
            new_filename = f"{str_pid}_c1s1_{frame_num:06d}.jpg"

            # Put 80% of data in train, 20% in test (Standard split)
            if frame_num % 5 == 0:
                dest_path = os.path.join(test_dir, new_filename)
                # We also need to add a few to 'query' for validation to work
                if frame_num % 10 == 0:
                     shutil.copy(img_path, os.path.join(query_dir, new_filename))
            else:
                dest_path = os.path.join(train_dir, new_filename)

            shutil.copy(img_path, dest_path)

    print(f"Dataset successfully formatted at: {dest_dir}")


convert_to_market1501(args.input, args.output)
