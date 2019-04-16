import sys
from pathlib import Path

sys.path.append(str(Path().parent.absolute()))
from src.dataset.prepare import gen_all_data_pairs, gen_all_cbf
import src.config as cfg

COLLECT_DATES = ['20181119', '20181214']

if __name__ == '__main__':
    res = input('Generate all data_pairs? y/N ')
    if res == 'y':
        for date in COLLECT_DATES:
            gen_all_data_pairs(src_dir=cfg.PROPOSAL_DIR / date / 'raw')

    res = input('Generate all cbfs? y/N ')
    if res == 'y':
        for date in COLLECT_DATES:
            gen_all_cbf(src_dir=cfg.PROPOSAL_DIR / date / 'raw', dst_dir=cfg.CBF_DIR / date)
