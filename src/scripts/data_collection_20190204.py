import shutil
from pathlib import Path
import csv

PROPOSAL_DIR = Path('/data/visitors/biomax/20170251/20190204/raw')


def get_images(src: Path):
    return list(map(lambda f: f.absolute(), src.rglob('*.jpeg')))


def get_rows(images: list):
    rows = []
    for img in images:
        rows.append([str(img), '0'])
    return rows


if __name__ == '__main__':
    imgs = get_images(PROPOSAL_DIR)
    print(len(imgs))
    rows = get_rows(imgs)
    with open('/data/staff/common/ML-crystals/csv/classification.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
