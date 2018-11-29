import logging as log
log.basicConfig(
    level=log.INFO,
    filename='./logs/pairgen.log',
    )

from pathlib import Path
from dataset import gen_all_data_pairs

log.info('====================NEW=RUN====================')

gen_all_data_pairs(
    src_dir=Path('/mnt/maxiv-visitors/biomax/20180479/20181119/raw/'),
    )

log.info('ALL DONE!')
