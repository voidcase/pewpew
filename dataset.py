import h5py as h5
import sys
import sqlite3 as sql


def get_timestamps(h5filepath: str, num_frames) -> list:
    from datetime import datetime
    DATEPATH = 'entry/instrument/detector/detectorSpecific/data_collection_date'
    f = h5.File(h5filepath)
    try:
        datestr = f[DATEPATH].value
    except KeyError:
        print(f'file "{h5filepath}" does not have dataset "{DATEPATH}, exiting."', file=sys.stderr)
        sys.exit(1)
    start_time = datetime.strptime(datestr.decode(), '%Y-%m-%dT%H:%M:%S.%f')
    exposure_time = f['entry/instrument/detector/frame_time'].value
    readout_time = f['entry/instrument/detector/detector_readout_time'].value
    frame_time = exposure_time + readout_time
    return [start_time.timestamp() + frame_time*i for i in range(num_frames)]


def to_cbfs(path: str, h5filename: str) -> int:
    """returns number of frames"""
    # TODO
    pass


def extract_ys(h5file: str):
    # TODO
    # NOTE: don't create cbfs if they already exist. because they will.
    pass


def compile_dataset(path):
    from os import listdir
    data = list()
    for h5file in listdir(f'{path}/h5/'):
        num_frames = to_cbfs(path, h5file)
        timestamps = get_timestamps(f'{path}/h5/{h5file}', num_frames)
        ys = extract_ys(h5file)
        uuids = [closest_img(t) for t in timestamps]
        data += list(zip(uuids, ys))
    # save to db
    conn = database(path)
    query = 'INSERT INTO dataset (uuid, y) VALUES '
    query += ','.join([f'("{uuid}", {y})' for uuid, y in data]) + ';'
    conn.execute(query)


def database(path):
    conn = sql.connect(f'{path}/meta.db')
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            uuid VARCHAR(100) UNIQUE NOT NULL,
            timestamp REAL NOT NULL
        );
        """)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset (
            uuid VARCHAR(100) UNIQUE NOT NULL,
            y INTEGER NOT NULL
        );
        """)
    return conn


def save_buffer(buf, path):
    from uuid import uuid4
    from time import time
    import cv2
    import np
    print('saving buffer to disk at {}'.format(time()))
    db = database(path)
    if len(buf) == 0:
        return
    query = 'insert into images (uuid, timestamp) values '
    for jpg, timestamp in buf:
        image_id = uuid4()
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f'{path}/images/{image_id}.jpg', i)
        query += f'("{image_id}", "{timestamp}"),'
    query = query[:-1]  # remove last ',' from query
    query += ';'
    db.execute(query)
    db.commit()
    db.close()


def closest_img(datapath, frame_time):
    conn = sql.connect(f'{datapath}/meta.db')
    query = """
        SELECT uuid, timestamp
        FROM images
        ORDER BY abs(timestamp - ?)
        LIMIT 1;
        """
    res = conn.execute(query, frame_time)
    if len(res) < 1:
        print('db is empty, get some images first.', file=sys.stderr)
        sys.exit(1)
    uuid, timestamp = list(res)[0]
    return uuid, timestamp
