"""
Convert labelImg (Pascal VOC) XML annotations into a TFRecord file.

    python generate_tfrecord.py -x data/images/train -l data/label_map.pbtxt -o data/train.record
    python generate_tfrecord.py -x data/images/test  -l data/label_map.pbtxt -o data/test.record
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import tensorflow as tf

from utils import load_label_map

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int_list(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def find_image(xml_path):
    # labelImg stores the filename inside the XML, but the extension case
    # sometimes drifts (.JPG vs .jpg), so look the image up by stem instead
    for ext in IMAGE_EXTS:
        for candidate in (ext, ext.upper()):
            path = xml_path.with_suffix(candidate)
            if path.exists():
                return path
    raise FileNotFoundError(f"No image found next to {xml_path}")


def build_example(xml_path, label_ids):
    root = ET.parse(xml_path).getroot()
    image_path = find_image(xml_path)

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    xmins, xmaxs, ymins, ymaxs, names, ids = [], [], [], [], [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        box = obj.find("bndbox")
        xmins.append(float(box.find("xmin").text) / width)
        xmaxs.append(float(box.find("xmax").text) / width)
        ymins.append(float(box.find("ymin").text) / height)
        ymaxs.append(float(box.find("ymax").text) / height)
        names.append(name.encode())
        ids.append(label_ids[name])

    filename = image_path.name.encode()
    return tf.train.Example(features=tf.train.Features(feature={
        "image/height": _int(height),
        "image/width": _int(width),
        "image/filename": _bytes(filename),
        "image/source_id": _bytes(filename),
        "image/encoded": _bytes(image_path.read_bytes()),
        "image/format": _bytes(image_path.suffix.lstrip(".").lower().encode()),
        "image/object/bbox/xmin": _float_list(xmins),
        "image/object/bbox/xmax": _float_list(xmaxs),
        "image/object/bbox/ymin": _float_list(ymins),
        "image/object/bbox/ymax": _float_list(ymaxs),
        "image/object/class/text": _bytes_list(names),
        "image/object/class/label": _int_list(ids),
    }))


def main():
    parser = argparse.ArgumentParser(description="labelImg XML -> TFRecord")
    parser.add_argument("-x", "--xml_dir", required=True, help="folder with images + XML files")
    parser.add_argument("-l", "--labels_path", required=True, help="path to label_map.pbtxt")
    parser.add_argument("-o", "--output_path", required=True, help="where to write the .record file")
    args = parser.parse_args()

    label_ids = {name: i for i, name in load_label_map(args.labels_path).items()}
    xml_files = sorted(Path(args.xml_dir).glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found in {args.xml_dir}")

    with tf.io.TFRecordWriter(args.output_path) as writer:
        for xml_path in xml_files:
            writer.write(build_example(xml_path, label_ids).SerializeToString())

    print(f"Wrote {len(xml_files)} examples -> {args.output_path}")


if __name__ == "__main__":
    main()
