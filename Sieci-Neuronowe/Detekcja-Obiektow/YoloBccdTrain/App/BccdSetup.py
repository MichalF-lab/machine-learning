"""
BCCD dataset setup: download Shenggan/BCCD_Dataset (Pascal VOC format)
and convert to YOLO format with standard train/val/test split.
"""
import os
import shutil
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

BCCD_ZIP_URL = 'https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip'

# Class ordering matches alphabetical convention used by most BCCD-in-YOLO mirrors
BCCD_CLASSES = ['Platelets', 'RBC', 'WBC']


def voc_xml_to_yolo(xml_path: Path, txt_path: Path, classes: list[str]) -> int:
    """Convert one Pascal VOC XML annotation to YOLO format. Returns # of boxes written."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)

        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)

        # YOLO format: class_id cx cy w h (all normalized to [0,1])
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

    txt_path.write_text('\n'.join(lines) + ('\n' if lines else ''))
    return len(lines)


def setup_bccd(dataset_root: Path) -> Path:
    """
    Ensure BCCD is present at dataset_root in YOLO format. Returns dataset_root.
    Layout produced:
        dataset_root/
            images/{train,val,test}/*.jpg
            labels/{train,val,test}/*.txt
            classes.txt
    """
    train_marker = dataset_root / 'images' / 'train'
    if train_marker.exists() and any(train_marker.iterdir()):
        print(f'[INFO] BCCD already prepared at {dataset_root}')
        return dataset_root

    dataset_root.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ────────────────────────────────────────────────────────
    download_dir = dataset_root.parent / '_bccd_download'
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / 'bccd.zip'

    if not zip_path.exists():
        print(f'[INFO] Downloading BCCD from {BCCD_ZIP_URL}')
        urllib.request.urlretrieve(BCCD_ZIP_URL, zip_path)
        print(f'[INFO] Downloaded {zip_path.stat().st_size / 1e6:.1f} MB')

    # ── 2. Extract ────────────────────────────────────────────────────────
    print('[INFO] Extracting BCCD archive')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(download_dir)

    extracted = download_dir / 'BCCD_Dataset-master' / 'BCCD'
    if not extracted.exists():
        raise RuntimeError(f'Expected extracted folder not found: {extracted}')

    # ── 3. Read split files ────────────────────────────────────────────────
    splits = {}
    for split in ['train', 'val', 'test']:
        split_file = extracted / 'ImageSets' / 'Main' / f'{split}.txt'
        if not split_file.exists():
            raise RuntimeError(f'Missing split file: {split_file}')
        ids = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
        splits[split] = ids
        print(f'[INFO]   split {split}: {len(ids)} images')

    # ── 4. Create output structure & convert ───────────────────────────────
    for split in splits:
        (dataset_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    skipped = 0
    for split, ids in splits.items():
        for img_id in ids:
            src_jpg = extracted / 'JPEGImages' / f'{img_id}.jpg'
            src_xml = extracted / 'Annotations' / f'{img_id}.xml'
            if not src_jpg.exists() or not src_xml.exists():
                skipped += 1
                continue
            dst_jpg = dataset_root / 'images' / split / f'{img_id}.jpg'
            dst_txt = dataset_root / 'labels' / split / f'{img_id}.txt'
            shutil.copy(src_jpg, dst_jpg)
            total_boxes += voc_xml_to_yolo(src_xml, dst_txt, BCCD_CLASSES)

    print(f'[INFO] Converted {sum(len(v) for v in splits.values()) - skipped} images, '
          f'{total_boxes} boxes, skipped {skipped}')

    # ── 5. Write classes.txt for human reference ───────────────────────────
    (dataset_root / 'classes.txt').write_text('\n'.join(BCCD_CLASSES) + '\n')

    # ── 6. Cleanup download temp ───────────────────────────────────────────
    shutil.rmtree(download_dir, ignore_errors=True)

    print(f'[INFO] BCCD ready at {dataset_root}')
    return dataset_root


def write_yaml(dataset_root: Path, yaml_path: Path) -> Path:
    """Write a YOLO-format dataset YAML for the prepared BCCD."""
    content = (
        f"path: {dataset_root}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"names:\n"
    )
    for i, name in enumerate(BCCD_CLASSES):
        content += f"  {i}: {name}\n"
    yaml_path.write_text(content)
    print(f'[INFO] Wrote dataset yaml: {yaml_path}')
    return yaml_path
