import h5py as h5
import sys
import sqlite3 as sql
import config
from pathlib import Path


cfg = config.TestConfig()


def get_timestamps(master_path: Path, num_frames) -> list:
    from datetime import datetime

    DATEPATH = 'entry/instrument/detector/detectorSpecific/data_collection_date'
    f = h5.File(str(master_path))
    try:
        datestr = f[DATEPATH].value
    except KeyError:
        print(
            f'file "{master_path}" does not have dataset "{DATEPATH}, exiting."',
            file=sys.stderr,
        )
        sys.exit(1)
    start_time = datetime.strptime(datestr.decode(), '%Y-%m-%dT%H:%M:%S.%f')
    exposure_time = f['entry/instrument/detector/frame_time'].value
    readout_time = f['entry/instrument/detector/detector_readout_time'].value
    frame_time = exposure_time + readout_time
    return [start_time.timestamp() + frame_time * i for i in range(num_frames)]


def extract_ys(masterpath: Path, generate=False):
    from diffractometrics import process_master, signal_strength, number_frames
    cbf_dir = cfg.PATH_DIR_CBF / masterpath.name
    if cbf_dir.is_dir():
        print('found existing cbfs, using them')
        cbf_paths = list(cbf_dir.iterdir())
    elif generate:
        print('found no cbfs, generating...')
        cbf_dir.mkdir()
        num_frames = number_frames(masterpath)
        cbf_paths = process_master(out=cbf_dir, master=masterpath, n=1, m=num_frames)
    else:
        raise RuntimeError('No cbfs found, call with generate=True to generate them as needed')
    return [signal_strength(cbf) for cbf in cbf_paths]


def save_dataset(xy: list):
    '''xy is a list of (image id, y) tuples'''
    conn = database()
    query = 'INSERT INTO dataset (uuid, y) VALUES '
    query += ','.join([f'("{uuid}", {y})' for uuid, y in xy]) + ';'
    conn.execute(query)


def get_all_masters():
    from os import listdir
    return [
        Path(f'{cfg.PATH_DIR_H5}/{fname}')
        for fname in listdir(cfg.PATH_DIR_H5)
        if fname.endswith('_master.h5')
        ]


def compile_dataset(mpath: Path):
    '''h5 -> dataset in database, assumes cbfs are made'''
    from diffractometrics import number_frames
    num_frames = number_frames(mpath)
    timestamps = get_timestamps(str(mpath), num_frames)
    ys = extract_ys(mpath)
    uuids = [closest_img(t) for t in timestamps]
    save_dataset(list(zip(uuids, ys)))


def database():
    conn = sql.connect(str(cfg.PATH_DB))
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS images (
            uuid VARCHAR(100) UNIQUE NOT NULL,
            timestamp REAL NOT NULL
        );
        '''
    )
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS dataset (
            uuid VARCHAR(100) UNIQUE NOT NULL,
            y INTEGER NOT NULL
        );
        '''
    )
    return conn


def save_buffer(buf):
    from uuid import uuid4
    from time import time
    import cv2
    import numpy as np

    print('saving buffer to disk at {}'.format(time()))
    db = database()
    if len(buf) == 0:
        return
    query = 'insert into images (uuid, timestamp) values '
    insert_items = []
    for jpg, timestamp in buf:
        image_id = uuid4()
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f'{cfg.PATH_DIR_CAM}/{image_id}.jpg', i)
        insert_items.append(f'("{image_id}", "{timestamp}")')
    query += ','.join(insert_items) + ';'
    db.execute(query)
    db.commit()
    db.close()


def closest_img(frame_time):
    conn = database()
    query = '''
        SELECT uuid, timestamp
        FROM images
        ORDER BY abs(timestamp - ?)
        LIMIT 1;
        '''
    res = conn.execute(query, frame_time)
    if len(res) < 1:
        print('db is empty, get some images first.', file=sys.stderr)
        sys.exit(1)
    uuid, timestamp = list(res)[0]
    return uuid, timestamp
