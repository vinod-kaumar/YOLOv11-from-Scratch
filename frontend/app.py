"""
FastAPI backend for Polyp Detection Web Application.
Loads the YOLOv11 model once at startup and serves predictions
for both images and videos.
"""

import sys
import os
import io
import time
import tempfile
import uuid

import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

# ── Add parent directory to sys.path so we can import utils ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from utils import YOLOv11, apply_nms

# ── Configuration ──────────────────────────────────────────────
MODEL_NAME  = "yolo11n"
NC          = 1
IMG_SIZE    = 640
CHECKPOINT  = os.path.join(PARENT_DIR, "checkpoints", "train", "best.pt")
IOU_THRESH  = 0.45
CLASS_NAMES = ["polyp"]

# Drawing constants (BGR for OpenCV)
BOX_COLOR   = (0, 200, 100)    # Teal-green
TEXT_COLOR  = (255, 255, 255)   # White
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.7
FONT_THICK  = 2
BOX_THICK   = 2

# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(title="PolypVision AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="static")

# ── Model Loading (once at startup) ───────────────────────────
device = torch.device("cpu")
model  = None

@app.on_event("startup")
def load_model():
    global model
    print(f"[INFO] Loading model from: {CHECKPOINT}")
    model = YOLOv11(model_name=MODEL_NAME, nc=NC).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("[INFO] Model loaded successfully!")

# ── Helper Functions ──────────────────────────────────────────

def letterbox_frame(frame, target=640, color=(114, 114, 114)):
    """Resize frame to target with padding."""
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


def preprocess(frame):
    """Convert frame to model input tensor."""
    padded, lb_params = letterbox_frame(frame, target=IMG_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    return tensor, lb_params


def postprocess(output, lb_params, orig_shape, conf_thresh=0.25):
    """Convert model output to original pixel space detections."""
    r, pad_left, pad_top = lb_params
    orig_h, orig_w = orig_shape

    output_permuted = output.permute(0, 2, 1)
    batch_preds = apply_nms(output_permuted, conf_thresh=conf_thresh, iou_thresh=IOU_THRESH)

    pred = batch_preds[0]
    if pred.shape[0] == 0:
        return []

    detections = []
    for px1, py1, px2, py2, score, _ in pred:
        ox1 = int(np.clip((px1 - pad_left) / r, 0, orig_w))
        oy1 = int(np.clip((py1 - pad_top) / r, 0, orig_h))
        ox2 = int(np.clip((px2 - pad_left) / r, 0, orig_w))
        oy2 = int(np.clip((py2 - pad_top) / r, 0, orig_h))
        detections.append({
            "x1": ox1, "y1": oy1, "x2": ox2, "y2": oy2,
            "score": round(float(score), 4),
            "label": CLASS_NAMES[0],
        })
    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes on the frame."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        score, label = det["score"], det["label"]

        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)

        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        label_y = max(y1, th + baseline + 5)
        cv2.rectangle(out, (x1, label_y - th - baseline - 5), (x1 + tw + 5, label_y), BOX_COLOR, -1)
        cv2.putText(out, text, (x1 + 2, label_y - baseline - 2), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)

    return out


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(SCRIPT_DIR, "static", "index.html"))


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    conf_thresh: float = Query(0.25, ge=0.01, le=1.0),
):
    """
    Accept an image file, run inference, return annotated image as JPEG
    along with detection metadata in headers.
    """
    start = time.time()

    # Read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    orig_h, orig_w = frame.shape[:2]

    # Inference
    tensor, lb_params = preprocess(frame)
    with torch.no_grad():
        output = model(tensor)

    detections = postprocess(output, lb_params, (orig_h, orig_w), conf_thresh)
    annotated = draw_detections(frame, detections)

    elapsed = round((time.time() - start) * 1000, 1)

    # Encode annotated image to JPEG bytes
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Return image with metadata headers
    headers = {
        "X-Detections": str(len(detections)),
        "X-Processing-Time": f"{elapsed}ms",
        "X-Detections-JSON": str(detections).replace("'", '"'),
        "Access-Control-Expose-Headers": "X-Detections, X-Processing-Time, X-Detections-JSON",
    }

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers=headers,
    )


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    conf_thresh: float = Query(0.25, ge=0.01, le=1.0),
):
    """
    Accept a video file, process each frame, return annotated video as MP4.
    """
    start = time.time()

    # Save uploaded video to a temp file
    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    tmp_in = os.path.join(tempfile.gettempdir(), f"polyp_in_{uuid.uuid4().hex}{suffix}")
    tmp_out = os.path.join(tempfile.gettempdir(), f"polyp_out_{uuid.uuid4().hex}.mp4")

    with open(tmp_in, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(tmp_in)
    if not cap.isOpened():
        return JSONResponse(status_code=400, content={"error": "Could not open video"})

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (orig_w, orig_h))

    total_detections = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor, lb_params = preprocess(frame)
        with torch.no_grad():
            output = model(tensor)

        detections = postprocess(output, lb_params, (orig_h, orig_w), conf_thresh)
        total_detections += len(detections)
        annotated = draw_detections(frame, detections)
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()

    elapsed = round((time.time() - start) * 1000, 1)

    # Read the output video and return it
    def iterfile():
        with open(tmp_out, "rb") as f:
            yield from f
        # Clean up temp files
        try:
            os.remove(tmp_in)
            os.remove(tmp_out)
        except OSError:
            pass

    headers = {
        "X-Detections": str(total_detections),
        "X-Frames": str(frame_count),
        "X-Processing-Time": f"{elapsed}ms",
        "Access-Control-Expose-Headers": "X-Detections, X-Frames, X-Processing-Time",
    }

    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers=headers,
    )
