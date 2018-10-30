from pathlib import Path
from uuid import uuid4
from multiprocessing import pool
import numpy as np
import shutil
import subprocess
import re


EIGER_2_CBF = 'eiger2cbf'
SIGNAL_STRENGTH = 'distl.signal_strength'
BASE_DIR = Path('/mnt/maxiv/common/ML-crystals')
DIALS_ENV = '/usr/local/dials-v1-11-4/dials_env.sh'
master = BASE_DIR / 'h5/1149_2-Lysozyme_16_master.h5'


def create_tmp_dir(base_dir: Path) -> Path:
    tmp = base_dir / str(uuid4())
    tmp.mkdir()
    return tmp


def number_frames(master: Path):
    res = subprocess.run([EIGER_2_CBF, master], stdout=subprocess.PIPE)
    return int(res.stdout)


# Not sure if working :)
def run(cmds):
    with pool.Pool(processes=None) as p:
        return_codes = p.map(subprocess.call, cmds)
        failed_idx = np.argwhere(np.array(return_codes) > 0)
        if len(failed_idx) > 0:
            failed_files = [
                cmd[1] for i, cmd in enumerate(cmds) if np.any(np.isin(failed_idx, i))
            ]
            raise Exception(f"ERROR: {failed_files}")


def eiger2cbf_commands(out: Path, masters: list, n: int, m: int = None):
    """ Run eiger2cbg on all files in masters. n and m is same for all files """
    cmds = []
    dirs = []
    for master in masters:
        cmd = [EIGER_2_CBF, master, n]
        sub_dir = out / master.name[:-3]  # Remove .h5
        dirs.append(sub_dir)
        sub_dir.mkdir()
        out_file = None
        if m is None:
            out_file = f'{sub_dir}/out_{n:06}.cbf'
        else:
            cmd[2] = f'{n}:{m}'
            out_file = f'{sub_dir}/out'
        cmd.append(out_file)
        cmds.append(list(map(str, cmd)))
    return cmds


def signal_strength(cbf: Path):
    proc = subprocess.Popen(
        f'source {DIALS_ENV} && {SIGNAL_STRENGTH} {cbf}',
        shell=True,
        stdout=subprocess.PIPE,
    )
    res, err = proc.communicate()
    if err is not None:
        print(err)
    m = re.search(r'^\s*Spot\sTotal\s*:\s*(\d+)', res.decode(), flags=re.MULTILINE)
    if m is None:
        raise Exception('Could not determine number of spots')
    return int(m.group(1))


# WIP
def extract_y(clean=False):
    tmp = create_tmp_dir(BASE_DIR)
    masters = list((BASE_DIR / 'h5').glob('*master.h5'))[:4]
    cmds = eiger2cbf_commands(tmp, masters, 1)
    run(cmds)
    cbfs = [c[3] for c in cmds]
    if clean:
        shutil.rmtree(tmp)

    return tmp, cmds


def clean(tmp):
    shutil.rmtree(tmp)
