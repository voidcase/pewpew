from pathlib import Path
from diffractometrics import QueueEntry
import config as cfg
import logging as log
import re
import json
import subprocess

log.basicConfig(filename=f'logs/{Path(__file__).stem}.log', level=log.INFO)


def write_csv():
    import csv

    rows = []
    for json_path in cfg.DATA_DIR.iterdir():
        json_file = json_path.name
        print(json_file)
        split = json_file.split('_')
        sample, local_user, threshold = split[0], '_'.join(split[1:3]), split[3][:-5]
        snapshot_dir = cfg.PROPOSAL_DIR / sample / 'timed_snapshots'
        with open(json_path, 'r') as f:
            pairs = json.load(f)
        for img, frame_nbr in pairs.items():
            dials_out = (
                cfg.PATH_DIR_PROJECT
                / 'sig_str_out'
                / '__data__staff__common__ML-crystals__real_cbf__{sample}__{local_user}_master__out{frame_nbr}'.format(
                    sample=sample, local_user=local_user, frame_nbr=str(frame_nbr).zfill(6)
                )
            )
            y = parse_sigstr(dials_out)
            if y is not None:
                rows.append([str(snapshot_dir / img), y])

    with open(cfg.PATH_DIR_PROJECT / 'csv' / f'data_{threshold}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'y'])
        writer.writerows(rows)


def parse_sigstr(fp: Path):
    if not fp.exists():
        return None

    with open(fp) as f:
        match = re.search(r'Spot Total :\s*(\d+)', f.read())
        if not match:
            return None
        return int(match.group(1))


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


def number_frames(master: Path):
    res = subprocess.run([cfg.EIGER_2_CBF, master], stdout=subprocess.PIPE)
    return int(res.stdout)


def gen_cbf(sample_dir, dst_dir):
    from utils import run

    log.info(f'processing {sample_dir}')
    master_paths = list(sample_dir.glob('*_master.h5'))  # file access
    cmds = []
    for master in master_paths:
        cbf_dir = dst_dir / (sample_dir.stem)
        if not cbf_dir.is_dir():
            log.debug(f'making cbf dir for {sample_dir}')
            cbf_dir.mkdir(parents=True)
        num_frames = number_frames(master)  # file access
        if num_frames != 100:
            log.warning(f'abnormal number of frames: {num_frames}')
        master_dst = cbf_dir / master.stem

        # how many frames are already generated?
        num_done_frames = 0
        if master_dst.exists():
            num_done_frames = len(list(master_dst.iterdir()))

        if num_done_frames == num_frames:
            log.info(f'all cbfs already generated. continuing.')
            continue
        elif num_done_frames > 0:
            log.info(f'found {num_done_frames} already existing cbfs.')
        cmds.append(eiger2cbf_command(out=cbf_dir, master=master, n=1 + num_done_frames, m=num_frames))
    log.info(f'running {len(cmds)}/{len(master_paths)} commands...')
    try:
        run(cmds)  # file access
    except Exception as e:
        log.error(f'commands failed: {e.args[0]}')
    log.info('done!')


def gen_all_cbf(src_dir: Path, dst_dir: Path):
    from time import sleep

    log.info('========================================================')
    log.info('generating cfb, this will be mega slow, so buckle up mofo.')
    for sample_dir in src_dir.glob('Sample-*-*'):
        done = False
        while not done:
            try:
                gen_cbf(sample_dir, dst_dir)
                done = True
            except OSError:
                pause = 60  # seconds
                log.error(f'temporarily lost connection to samba, sleeping for {pause} seconds...')
                sleep(pause)
                log.info('yawn! waking up')

    log.info('all done!')


def gen_all_data_pairs(src_dir: Path):
    log.info('====================NEW=RUN====================')
    log.info('ALL DONE!')

    meta_files = src_dir.rglob('*.meta.txt')
    for meta_file in meta_files:
        meta = json.load(open(meta_file.absolute(), 'r'))
        qe = QueueEntry(meta)
        if qe.master_file.exists():
            qe.write_data_pairs()
            log.info(f'pairs of {qe.sample_dir.name} written')
        else:
            log.error(f'missing h5 file: {qe.master_file}')


if __name__ == '__main__':
    res = input('Generate all data_pairs? y/N ')
    if res == 'y':
        gen_all_data_pairs(src_dir=Path('/data/visitors/biomax/20180479/20181119/raw/'))

    res = input('Generate all cbfs? y/N ')
    if res == 'y':
        gen_all_cbf(
            src_dir=Path('/data/visitors/biomax/20180479/20181119/raw'),
            dst_dir=Path('/data/staff/common/ML-crystals/real_cbf'),
        )
