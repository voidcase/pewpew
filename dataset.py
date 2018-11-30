import h5py as h5
import sys
from pathlib import Path
import logging as log


def get_timestamps(master_path: Path, num_frames) -> list:
    from datetime import datetime

    DATEPATH = 'entry/instrument/detector/detectorSpecific/data_collection_date'
    f = h5.File(str(master_path))
    try:
        datestr = f[DATEPATH].value
    except KeyError:
        print(f'file "{master_path}" does not have dataset "{DATEPATH}, exiting."', file=sys.stderr)
        sys.exit(1)
    start_time = datetime.strptime(datestr.decode(), '%Y-%m-%dT%H:%M:%S.%f')
    exposure_time = f['entry/instrument/detector/frame_time'].value
    readout_time = f['entry/instrument/detector/detector_readout_time'].value
    frame_time = exposure_time + readout_time
    return [start_time.timestamp() + frame_time * i for i in range(num_frames)]


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
        cmds.append(eiger2cbf_command(
            out=cbf_dir,
            master=master,
            n=1+num_done_frames,
            m=num_frames,
            ))
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


def compile_dataset(mpath: Path):
    '''h5 -> dataset in database, assumes cbfs are made'''
    from diffractometrics import number_frames

    num_frames = number_frames(mpath)
    timestamps = get_timestamps(str(mpath), num_frames)
    ys = extract_ys(mpath)
    uuids = [closest_img(t) for t in timestamps]
    save_dataset(list(zip(uuids, ys)))
