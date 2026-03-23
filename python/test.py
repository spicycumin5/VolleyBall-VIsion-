import argparse
import os
import cv2
from ultralytics import YOLO


parser = argparse.ArgumentParser()

parser.add_argument("--model", help="Model to load. If none, defaults to 'yolo26x.pt'",
                    default='yolo216x.pt')
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", required=True, help="Output video path")
parser.add_argument('--show_conf', default=False, action='store_true',
                    help='Whether to show the confidence scores')
parser.add_argument('--show_labels', default=False,
                    action='store_true', help='Whether to show the labels')
parser.add_argument('--conf', type=float, default=0.5,
                    help='Object confidence threshold for detection')
parser.add_argument('--classes', nargs='+', default=None,
                    help='List of classes to detect')
parser.add_argument('--line_width', type=int, default=3,
                    help='Line width for bounding box visualization')
parser.add_argument('--font_size', type=float, default=3,
                    help='Font size for label visualization')

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
        results = model.track(
            frame,
            conf=args.conf,
            classes=[0, 32], # [human, sportsball]
            tracker='botsort.yaml',
            with_reid=True,
            persist=True,
            verbose=False,
        )

        annotated_frame = frame.copy()
        annotated_frame = results[0].plot(
            conf=args.show_conf,
            labels=args.show_labels,
            line_width=args.line_width,
            font_size=args.font_size,
        )

        out.write(annotated_frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
