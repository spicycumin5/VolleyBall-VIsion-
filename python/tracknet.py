import cv2
import torch
import numpy as np
from collections import deque

from models.tracket_v4 import TrackNet as TrackNetV4

class PyTorchTrackNetTracker:
    def __init__(self, weights_path, input_size=(512, 288), threshold=0.5):
        self.input_width, self.input_height = input_size
        self.threshold = threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = TrackNetV4()
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 3-frame buffer
        self.frame_buffer = deque(maxlen=3)

        self.current_heatmap = None

    def predict(self, frame):
        """Return (x, y) coordinates of the ball in the current frame, or None."""
        orig_h, orig_w = frame.shape[:2]

        # 1. Resize, convert BGR to RGB, and normalize to [0, 1]
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized = resized.astype(np.float32) / 255.0

        self.frame_buffer.append(resized)

        if len(self.frame_buffer) < 3:
            return None

        # 2. Prepare native PyTorch Tensor: (C, H, W)
        # Convert each (H, W, 3) frame to (3, H, W)
        frames_transposed = [np.transpose(f, (2, 0, 1)) for f in self.frame_buffer]

        # Concatenate along the channel axis to get a single (9, H, W) array
        combined_frames = np.concatenate(frames_transposed, axis=0)

        # Add the batch dimension to get (1, 9, H, W)
        tensor_input = torch.from_numpy(combined_frames).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            heatmap = self.model(tensor_input)

            # The model outputs a tensor of shape (1, 3, H, W). 
            # Index 2 is the heatmap corresponding to the current frame 't'.
            current_heatmap = heatmap[0, 2, :, :].cpu().numpy()

        # 4. Extract Coordinates
        max_val = np.max(current_heatmap)
        if max_val < self.threshold:
            return None # Confidence too low

        self.current_heatmap = current_heatmap

        # Find the (y, x) of the highest peak in the heatmap
        y_pred, x_pred = np.unravel_index(np.argmax(current_heatmap), current_heatmap.shape)

        # 5. Scale coordinates back to original video resolution
        x_orig = int(x_pred * (orig_w / self.input_width))
        y_orig = int(y_pred * (orig_h / self.input_height))

        return (x_orig, y_orig)


def draw_heatmap_overlay(annotated_frame, heatmap_2d, alpha=0.8):
    """
    Overlays a TrackNet heatmap onto the main video frame.
    """
    if heatmap_2d is None:
        return annotated_frame

    # 1. Resize the 512x288 heatmap to match the main video frame dimensions
    h, w = annotated_frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap_2d, (w, h))

    # 2. Normalize the math tensor into a standard 0-255 image
    heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # 3. Apply the JET colormap (Red = High Confidence, Blue = Low Confidence)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 4. Alpha blend the heatmap with the original frame
    blended = cv2.addWeighted(annotated_frame, 1 - alpha, heatmap_color, alpha, 0)

    # 5. MASKING (Crucial step!)
    # We don't want to tint the entire gym blue. We only apply the blended 
    # overlay to pixels where the neural network actually detects *something*.
    mask = heatmap_uint8 > 15
    annotated_frame[mask] = blended[mask]

    return annotated_frame
