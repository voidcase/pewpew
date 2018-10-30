from pathlib import Path
from uuid import uuid4
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


def process_master(out: Path, masters: list, n: int, m: int = None):
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
    procs = [subprocess.Popen(cmd) for cmd in cmds]
    for proc in procs:
        proc.wait()
        if proc.returncode != 0:
            raise Exception(f"ERROR: error code {proc.returncode}")
    return dirs


def signal_strength(cbf: Path):
    proc = subprocess.Popen(
            f'source {DIALS_ENV} && {SIGNAL_STRENGTH} {cbf}',
            shell=True,
            stdout=subprocess.PIPE
            )
    res, err = proc.communicate()
    if err is not None:
        print(err)
    m = re.search(r'^\s*Spot\sTotal\s*:\s*(\d+)', res.decode(), flags=re.MULTILINE)
    if m is None:
        raise Exception('Could not determine number of spots')
    return int(m.group(1))


if __name__ == '__main__':
    pass
    # tmp = create_tmp_dir(BASE_DIR)
    # out = process_master(tmp, master, 3, 4)
    # ys = [signal_strength(f) for f in out]
    # print(ys)
