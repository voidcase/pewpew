import h5py as h5
import sys
from pathlib import Path
from datetime import datetime
import logging as log
import config as cfg
import json
import re
from diffractometrics import QueueEntry


def get_timestamps_log(entry: QueueEntry):
    fmt = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    log = Path('/data/staff/common/ML-crystals/eiger_logs/timestamps.log')
    sample, scan = '/'.join(Path(entry.master_file).parts[-2:])[:-10].split('/')
    start_re = re.compile(r'^Starting.*{}\/{}.*at\s(\d.+)$'.format(sample, scan))
    finish_re = re.compile(r'^Finished.*{}\/{}.*at\s(\d.+)$'.format(sample, scan))
    matches = []
    with open(log, 'r') as f:
        for line in f:
            # No point in matching second match if we haven't found first
            match = start_re.match(line) if len(matches) == 0 else finish_re.match(line)
            if match is not None:
                matches.append(fmt(match.group(1)))
    if len(matches) != 2:
        raise Exception('Could not find start and/or finish time for {master_path}')

    duration = entry.nbr_frames * entry.exposure_time
    diff = matches[1] - matches[0]
    start_time = matches[0] + (diff - duration)
    return [start_time + (frame_nbr * entry.exposure_time) for frame_nbr in range(entry.nbr_frames)]


def get_timestamps(entry: QueueEntry) -> list:
    DATEPATH = 'entry/instrument/detector/detectorSpecific/data_collection_date'
    f = h5.File(str(entry.master_file))
    try:
        datestr = f[DATEPATH].value
    except KeyError:
        print(f'file "{entry.master_file}" does not have dataset "{DATEPATH}, exiting."', file=sys.stderr)
        sys.exit(1)
    start_time = datetime.strptime(datestr.decode(), '%Y-%m-%dT%H:%M:%S.%f')
    exposure_time = f['entry/instrument/detector/frame_time'].value
    readout_time = f['entry/instrument/detector/detector_readout_time'].value
    frame_time = exposure_time + readout_time
    return [start_time.timestamp() + frame_time * i for i in range(entry.nbr_frames)]


def gen_cbf(sample_dir, dst_dir):
    from diffractometrics import run, eiger2cbf_command, number_frames

    log.info(f'processing {sample_dir}')
    master_paths = list((rootdir or Path).rglob('*_master.h5'))  # file access
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
    import json
    from diffractometrics import QueueEntry

    meta_files = src_dir.rglob('*.meta.txt')
    for meta_file in meta_files:
        meta = json.load(open(meta_file.absolute(), 'r'))
        qe = QueueEntry(meta, meta['fileinfo']['directory'])
        if qe.master_file.exists():
            qe.write_data_pairs()
            log.info(f'pairs of {qe.sample_dir.name} written')
        else:
            log.error(f'missing h5 file: {qe.master_file}')


def compile_dataset():
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


def test_parse_sigstr():
    path = (
        cfg.PATH_DIR_PROJECT
        / 'sig_str_out/__data__staff__common__ML-crystals__real_cbf__Sample-4-16__local-user_2_master__out000069'
    )
    res = parse_sigstr(path)
    assert res == 0
