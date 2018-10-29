from pathlib import Path
from uuid import uuid4
import subprocess
import re


EIGER_2_CBF = 'eiger2cbf'
SIGNAL_STRENGTH = 'distl.signal_strength'
BASE_DIR = Path('/mnt/maxiv/common/ML-crystals')
master = BASE_DIR / 'h5/1149_2-Lysozyme_16_master.h5'


def create_tmp_dir(base_dir: Path) -> Path:
    tmp = base_dir / str(uuid4())
    tmp.mkdir()
    return tmp


def number_frames(master: Path):
    res = subprocess.run([EIGER_2_CBF, master], stdout=subprocess.PIPE)
    return int(res.stdout)


def process_master(out: Path, master: Path, n: int, m: int = None):
    cmd = [EIGER_2_CBF, master, n]
    out_file = None
    if m is None:
        out_file = f'{str(out)}/out_{n:06}.cbf'
    else:
        cmd[2] = f'{n}:{m}'
        out_file = f'{str(out)}/out'
    cmd.append(out_file)
    cmd = list(map(str, cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise Exception(f"ERROR: error code {res.returncode}")
    if m is None:
        return Path(out_file)
    return [out / ('out' + str(i).zfill(6) + '.cbf') for i in range(n, m+1)]


def signal_strength(cbf: Path):
    proc = subprocess.Popen(SIGNAL_STRENGTH + ' ' + str(cbf), shell=True, stdout=subprocess.PIPE)
    res, _ = proc.communicate()
    m = re.search(r'^\s*Spot\sTotal\s*:\s*(\d+)', res.decode(), flags=re.MULTILINE)
    if m is None:
        raise Exception('Could not determine number of spots')
    return int(m.group(1))


if __name__ == '__main__':
    tmp = create_tmp_dir(BASE_DIR)
    out = process_master(tmp, master, 3, 4)
    ys = [signal_strength(f) for f in out]
    print(ys)
