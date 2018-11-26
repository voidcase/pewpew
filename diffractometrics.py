from pathlib import Path
from multiprocessing import pool
import numpy as np
import shutil
import subprocess
import re
import json

import utils
import dataset
import config as cfg


class QueueEntry:
    SNAPSHOT_DIR = "timed_snapshots"
    TIME_OFFSET = 101.724_564_075_469_97

    def __init__(self, meta, sample_dir):
        self.meta = meta
        self.sample_dir = Path(sample_dir)
        self.master_file = Path(meta["fileinfo"]["filename"])
        self.nbr_images = meta["oscillation_sequence"][0]["number_of_images"]
        self.zoom = meta.get("zoom1", None)
        self.frontlight = meta.get("backlight", None)
        self.prefix = f"{meta['fileinfo']['prefix']}_{meta['fileinfo']['run_number']}"
        self.pairs = {}

    def write_data_pairs(self):
        if not len(self.pairs):
            self.__find_closest_images()
        name = f"{self.sample_dir.name}_{self.prefix}"
        with open(cfg.DATA_DIR / name, "w") as f:
            json.dump(self.pairs, f, indent=4)

    def __find_closest_images(self, THRESHOLD=0.5):
        data_pairs = {}
        timestamps = dataset.get_timestamps(self.__local_master_file(), self.nbr_images)
        candidates = sorted((self.sample_dir / cfg.SNAPSHOT_DIR).glob(f"{self.prefix}*.jpeg"), key=lambda p: p.name)

        ts_i = 0
        i = 0
        while i < len(candidates) - 1:
            # We found matches for all timestamps
            if ts_i == len(timestamps):
                break
            ts = timestamps[ts_i]
            img_ts = (
                np.array(
                    [
                        float(candidates[i].stem[len(self.prefix) + 1 :]),
                        float(candidates[i + 1].stem[len(self.prefix) + 1 :]),
                    ]
                )
                - self.TIME_OFFSET
            )

            # Skip images taken before collection
            if ts > img_ts[1]:
                i += 2
                continue

            diff = np.abs(img_ts - ts)
            closest_idx = np.argmin(diff)

            # Only add images if they are a taken close enough
            if diff[closest_idx] < THRESHOLD:
                data_pairs[candidates[i + closest_idx].name] = ts_i + 1
                ts_i += 1
                i += 2
            else:  # Skip frame if there is not image taken close enough
                ts_i += 1
                continue

        return data_pairs

    def __local_master_file(self):
        return self.sample_dir / self.master_file.name

    def __str__(self):
        return str(Path(self.master_file))

    def __repr__(self):
        return self.__str__()


def number_frames(master: Path):
    res = subprocess.run([cfg.EIGER_2_CBF, master], stdout=subprocess.PIPE)
    return int(res.stdout)


# Not sure if working :)
def run(cmds):
    with pool.Pool(processes=None) as p:
        return_codes = p.map(subprocess.call, cmds)
        failed_idx = np.argwhere(np.array(return_codes) > 0)
        if len(failed_idx) > 0:
            failed_files = [cmd[1] for i, cmd in enumerate(cmds) if np.any(np.isin(failed_idx, i))]
            raise Exception(f"ERROR: {failed_files}")


def eiger2cbf_command(out: Path, master: Path, n: int, m: int = None):
    """ Run eiger2cbg on all files in masters. n and m is same for all files """
    cmd = [cfg.EIGER_2_CBF, master, n]
    sub_dir = out / master.name[:-3]  # Remove .h5
    if not sub_dir.is_dir():
        sub_dir.mkdir()
    out_file = None
    if m is None:
        out_file = f'{sub_dir}/out_{n:06}.cbf'
    else:
        cmd[2] = f'{n}:{m}'
        out_file = f'{sub_dir}/out'
    cmd.append(out_file)
    return list(map(str, cmd))


def signal_strength(cbf: Path):
    proc = subprocess.Popen(
        f"source {cfg.DIALS_ENV} && {cfg.SIGNAL_STRENGTH} {cbf}", shell=True, stdout=subprocess.PIPE
    )
    res, err = proc.communicate()
    if err is not None:
        print(err)
    m = re.search(r"^\s*Spot\sTotal\s*:\s*(\d+)", res.decode(), flags=re.MULTILINE)
    if m is None:
        raise Exception("Could not determine number of spots")
    return int(m.group(1))


# WIP
def extract_y(clean=False):
    tmp = utils.create_tmp_dir(cfg.BASE_DIR)
    masters = list((cfg.BASE_DIR / "h5").glob("*master.h5"))[:4]
    cmds = eiger2cbf_commands(tmp, masters, 1)
    run(cmds)
    cbfs = [c[3] for c in cmds]
    if clean:
        shutil.rmtree(tmp)

    return tmp, cmds


def clean(tmp):
    shutil.rmtree(tmp)
