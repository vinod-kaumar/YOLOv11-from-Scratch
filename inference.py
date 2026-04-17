import cv2
import torch
import numpy as np
import os
import time
from pathlib import Path

# Import custom architecture and NMS helper
from utils import YOLOv11, apply_nms

# You can change these values to suit your environment
MODEL_NAME   = "yolo11n"
NC           = 1
IMG_SIZE     = 640
CHECKPOINT   = os.path.join("checkpoints", "train", "best.pt")

CONF_THRESH  = 0.25
IOU_THRESH   = 0.45

# Colors (BGR format for OpenCV)
BOX_COLOR    = (0, 0, 255)    # Red
TEXT_COLOR   = (0, 0, 0)      # Black
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.6
FONT_THICK   = 1
BOX_THICK    = 2

CLASS_NAMES  = ["polyp"]

def letterbox_frame(frame, target=640, color=(114, 114, 114)):
    """Resize frame to target with padding, returns (padded_image, lb_params)."""
    h, w = frame.shape[:2]
    r = min(target / h, target / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    pad_w, pad_h = target - nw, target - nh
    left, top = pad_w // 2, pad_h // 2
    
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, (r, left, top)

def preprocess(frame, device):
    """Convert frame to model input tensor."""
    padded, lb_params = letterbox_frame(frame, target=IMG_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, lb_params

def postprocess(output, lb_params, orig_shape):
    """Convert model output to original pixel space detections."""
    r, pad_left, pad_top = lb_params
    orig_h, orig_w = orig_shape
    
    # 1. Apply NMS (handles confidence thresholding and xywh -> xyxy)
    output_permuted = output.permute(0, 2, 1)
    batch_preds = apply_nms(output_permuted, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH)
    
    pred = batch_preds[0] # Single image batch
    if pred.shape[0] == 0:
        return []
    
    detections = []
    for px1, py1, px2, py2, score, _ in pred:
        # 2. Reverse letterbox (undo padding and scaling)
        ox1 = int(np.clip((px1 - pad_left) / r, 0, orig_w))
        oy1 = int(np.clip((py1 - pad_top) / r, 0, orig_h))
        ox2 = int(np.clip((px2 - pad_left) / r, 0, orig_w))
        oy2 = int(np.clip((py2 - pad_top) / r, 0, orig_h))
        
        detections.append({
            'x1': ox1, 'y1': oy1, 'x2': ox2, 'y2': oy2,
            'score': float(score),
            'label': CLASS_NAMES[0]
        })
    return detections

def draw_detections(frame, detections):
    """Draw annotations on the original frame."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        score, label = det['score'], det['label']
        
        # Draw Box
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)
        
        # Draw Label
        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        label_y = max(y1, th + baseline + 5)
        cv2.rectangle(out, (x1, label_y - th - baseline - 5), (x1 + tw + 5, label_y), BOX_COLOR, -1)
        cv2.putText(out, text, (x1 + 2, label_y - baseline - 2), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)
    
    return out

def run_inference(source_path):
    """Main runner for image or video inference."""
    device = torch.device('cpu') # Force CPU as requested
    
    # Resolve script root to handle relative internal paths correctly
    ROOT = os.path.dirname(os.path.abspath(__file__))

    try:
        # 1. Load Model
        model = YOLOv11(model_name=MODEL_NAME, nc=NC).to(device)
        checkpoint_path = os.path.join(ROOT, CHECKPOINT)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 2. Identify Source Type
    ext = os.path.splitext(source_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    if is_video:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {source_path}")
            return
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        window_name = f"Inference: {os.path.basename(source_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference pipeline
                tensor, lb_params = preprocess(frame, device)
                with torch.no_grad():
                    output = model(tensor)
                
                detections = postprocess(output, lb_params, (orig_h, orig_w))
                annotated = draw_detections(frame, detections)
                
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == 27: # ESC to stop
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    else:
        # Image inference
        try:
            frame = cv2.imread(source_path)
            if frame is None:
                print(f"[ERROR] Could not read image: {source_path}")
                return
            
            orig_h, orig_w = frame.shape[:2]
            tensor, lb_params = preprocess(frame, device)
            with torch.no_grad():
                output = model(tensor)
            
            detections = postprocess(output, lb_params, (orig_h, orig_w))
            annotated = draw_detections(frame, detections)
            
            window_name = f"Inference: {os.path.basename(source_path)}"
            cv2.imshow(window_name, annotated)
            print(f"Detections found: {len(detections)}. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[ERROR] Image inference failed: {e}")

if __name__ == "__main__":
    # Resolve root for source path relative to script
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # Change this path to test images or videos
    source = os.path.join(ROOT, "dataset", "test", "images", "Horizontal-CVC-557_png.rf.54f642eb47c7918a41d01c979e0d77ca.jpg")
    
    # source = os.path.join(ROOT, "dataset", "test", "video", "demo_video.mp4")
    
    run_inference(source)
