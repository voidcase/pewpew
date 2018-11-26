from pathlib import Path
import shutil
from os import system

ROOT = Path('/mnt/maxiv-staff/common/ML-crystals/real_cbf')
for sample in ROOT.iterdir():
    for outer in sample.iterdir():
        inner = outer / outer.stem
        if inner.exists():
            status = system(f'mv {inner.absolute()}/* {outer.absolute()}')
            shutil.rmtree(inner.absolute())
