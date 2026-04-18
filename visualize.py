import cv2
import numpy as np
import os


def draw_yolo_boxes(image_path, label_path, save=False):
    """
    Reads an image and its YOLO-format label file, draws bounding boxes,
    and displays the result using OpenCV.
    """
    # --- Load Image ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    MAX_DISPLAY_DIM = 900

    # --- Parse Labels ---
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Convert normalized center format -> pixel corner format
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((cls_id, x1, y1, x2, y2))
    else:
        print(f"[WARNING] Label file not found: {label_path}")

    # --- Draw Boxes ---
    CLASS_NAME = {0: "polyp"}
    BOX_COLOR = (0, 255, 0)       # green
    TEXT_COLOR = (255, 255, 255)   # white
    THICKNESS = 2

    for cls_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)

        label_text = CLASS_NAME.get(cls_id, f"cls_{cls_id}")
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(img, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    # --- Saving Logic ---
    if save:
        # Create input directory relative to the image (assumes dataset/test1/images/...)
        input_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), "input")
        os.makedirs(input_dir, exist_ok=True)
        
        save_path = os.path.join(input_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, img)
        print(f"[INFO] Saved ground truth visualization to: {save_path}")

    # --- Display ---
    display_img = img.copy()
    if max(h, w) > MAX_DISPLAY_DIM:
        scale = MAX_DISPLAY_DIM / max(h, w)
        display_img = cv2.resize(img, (int(w * scale), int(h * scale)))

    window_name = f"Ground Truth: {os.path.basename(image_path)}"
    cv2.imshow(window_name, display_img)
    print(f"Showing {len(boxes)} box(es). Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    label_path = os.path.join(SCRIPT_DIR, "test", "labels", "1.txt")
    image_path = os.path.join(SCRIPT_DIR, "test", "images", "1.jpg")
    
    print(f"Processing: {image_path}")
    draw_yolo_boxes(image_path, label_path, save=True)
