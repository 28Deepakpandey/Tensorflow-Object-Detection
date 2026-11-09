"""
Run a trained TFOD model on an image, a video file or a webcam.

    python detect.py --source test.jpg
    python detect.py --source traffic.mp4
    python detect.py --source 0            # webcam, press q to quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from utils import draw_detections, load_label_map

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict(detect_fn, frame_bgr):
    """Run the model on one BGR frame. Returns boxes, scores, class ids."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    batch = tf.convert_to_tensor(rgb[np.newaxis, ...], dtype=tf.uint8)
    out = detect_fn(batch)
    return (
        out["detection_boxes"][0].numpy(),
        out["detection_scores"][0].numpy(),
        out["detection_classes"][0].numpy().astype(int),
    )


def run_image(detect_fn, labels, path, threshold):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    boxes, scores, classes = predict(detect_fn, frame)
    result = draw_detections(frame, boxes, scores, classes, labels, threshold)

    out_path = Path("outputs") / f"{Path(path).stem}_detected.jpg"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), result)
    print(f"Saved -> {out_path}")


def run_stream(detect_fn, labels, source, threshold):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.time()
        boxes, scores, classes = predict(detect_fn, frame)
        fps = 1 / (time.time() - start)

        frame = draw_detections(frame, boxes, scores, classes, labels, threshold)
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Object detection (q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Object detection inference")
    parser.add_argument("--source", default="0", help="image / video path, or camera index")
    parser.add_argument("--model", default="exports/saved_model", help="exported saved_model folder")
    parser.add_argument("--labels", default="data/label_map.pbtxt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    detect_fn = tf.saved_model.load(args.model)
    labels = load_label_map(args.labels)

    if args.source.isdigit():
        run_stream(detect_fn, labels, int(args.source), args.threshold)
    elif Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(detect_fn, labels, args.source, args.threshold)
    else:
        run_stream(detect_fn, labels, args.source, args.threshold)


if __name__ == "__main__":
    main()
