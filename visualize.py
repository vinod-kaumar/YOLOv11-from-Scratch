import cv2
import numpy as np
import os


def draw_yolo_boxes(image_path, label_path):
    """
    Reads an image and its YOLO-format label file, draws bounding boxes,
    and displays the result using OpenCV.
    
    Label format (per line): class_id  cx  cy  w  h   (all normalized 0-1)
    """
    # --- Load Image ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

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
        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(img, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    # --- Display ---
    window_name = os.path.basename(image_path)
    cv2.imshow(window_name, img)
    print(f"Showing {len(boxes)} box(es). Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    label_path = os.path.join(SCRIPT_DIR, "dataset", "test", "labels",
                              "Horizontal-CVC-557_png.rf.54f642eb47c7918a41d01c979e0d77ca.txt")
    image_path = os.path.join(SCRIPT_DIR, "dataset", "test", "images",
                              "Horizontal-CVC-557_png.rf.54f642eb47c7918a41d01c979e0d77ca.jpg")
    print(image_path)
    draw_yolo_boxes(image_path, label_path)
