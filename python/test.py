import argparse
import os
import cv2
from ultralytics import YOLO


parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True, help="Model to load")
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")

args = parser.parse_args()

def main():
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FOS)
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*.'XVID')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(
            frame,
            classes=[0, 32], # [human, sportsball]
        )

        annotated_frame = frame.copy()
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
