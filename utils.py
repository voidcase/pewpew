from pathlib import Path
from datetime import datetime
from uuid import uuid4
import json


def read_meta(filename) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


def get_meta_files(snapshot_dir):
    d = Path(snapshot_dir)
    if not d.exists():
        raise Exception(f"Directory does not exists: {d.absolute()}")
    return Path(snapshot_dir).glob("*.meta.txt")


def fts(ts):
    return datetime.fromtimestamp(ts).strftime("%H-%M-%S.%f")


def create_tmp_dir(base_dir: Path) -> Path:
    tmp = base_dir / str(uuid4())
    tmp.mkdir()
    return tmp
