from multiprocessing import pool
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import subprocess
import re
from tqdm import tqdm

tqdm.pandas()


def run(cmds):
    with pool.Pool(processes=None) as p:
        return_codes = p.map(subprocess.call, cmds)
        failed_idx = np.argwhere(np.array(return_codes) > 0)
        if len(failed_idx) > 0:
            failed_files = [cmd[1] for i, cmd in enumerate(cmds) if np.any(np.isin(failed_idx, i))]
            raise Exception(f"ERROR: {failed_files}")


def read_meta(*args) -> dict:
    for filename in args:
        with open(filename, "r") as f:
            yield json.load(f)


def get_meta_files(snapshot_dir):
    d = Path(snapshot_dir)
    if not d.exists():
        raise Exception(f"Directory does not exists: {d.absolute()}")
    return read_meta(Path(snapshot_dir).glob("*.meta.txt"))


def fts(ts):
    return datetime.fromtimestamp(ts).strftime("%H-%M-%S.%f")


def get_sample(s: Path):
    return str(re.search('Sample-([0-9]+-[0-9]+)', str(s)).group(1))


def get_scan(s: Path):
    return str(re.search('local-user_([0-9]+)', str(s)).group(1))
