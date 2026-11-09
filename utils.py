"""Small helpers shared by the notebook and the inference script."""

import re
from pathlib import Path

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def write_label_map(classes, path):
    """Write a label_map.pbtxt. Ids start at 1 (0 is reserved for background)."""
    with open(path, "w") as f:
        for i, name in enumerate(classes, start=1):
            f.write(f'item {{\n    name: "{name}"\n    id: {i}\n}}\n\n')


def load_label_map(path):
    """Read a label_map.pbtxt and return {id: name}."""
    text = Path(path).read_text()
    pairs = re.findall(r"name:\s*[\"']([^\"']+)[\"']\s*id:\s*(\d+)", text)
    return {int(i): name for name, i in pairs}


def _color(class_id):
    # same class -> same color on every frame
    return (37 * class_id % 255, 91 * class_id % 255, 173 * class_id % 255)


def draw_detections(image, boxes, scores, class_ids, labels, threshold=0.5):
    """Draw detections on a BGR image.

    boxes are the normalized [ymin, xmin, ymax, xmax] values the model returns.
    """
    h, w = image.shape[:2]
    out = image.copy()

    for box, score, cid in zip(boxes, scores, class_ids):
        if score < threshold:
            continue

        ymin, xmin, ymax, xmax = box
        p1 = (int(xmin * w), int(ymin * h))
        p2 = (int(xmax * w), int(ymax * h))
        color = _color(int(cid))
        text = f"{labels.get(int(cid), int(cid))} {score:.0%}"

        cv2.rectangle(out, p1, p2, color, 2)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        top = max(p1[1] - th - 8, 0)
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw + 6, top + th + 8), color, -1)
        cv2.putText(out, text, (p1[0] + 3, top + th + 2), FONT, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return out
