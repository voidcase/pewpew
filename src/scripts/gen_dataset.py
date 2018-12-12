from pathlib import Path

from dataset.prepare import gen_all_data_pairs, gen_all_cbf
import config as cfg

if __name__ == '__main__':
    res = input('Generate all data_pairs? y/N ')
    if res == 'y':
        gen_all_data_pairs(src_dir=Path(cfg.PROPOSAL_DIR))

    res = input('Generate all cbfs? y/N ')
    if res == 'y':
        gen_all_cbf(
            src_dir=Path(cfg.PROPOSAL_DIR),
            dst_dir=Path(cfg.CBF_DIR),
        )
