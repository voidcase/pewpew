from pathlib import Path
from dataset import gen_all_cbf
import logging as log

log.basicConfig(filename='logs/cbfgen.log', level=log.INFO)

log.info('========================================================')

gen_all_cbf(
        src_dir=Path('/mnt/maxiv-visitors/biomax/20180479/20181119/raw'),
        dst_dir=Path('/mnt/maxiv-staff/common/ML-crystals/real_cbf'))
