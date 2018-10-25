import cv2
import requests
import numpy as np
import sys
import sqlite3 as sql
from time import time

REAL = True
# REAL = False

API_PATH = '/mxcube/api/v0.1'

mock_cookies = {
    'session': '4eb6807d-ed71-46d8-a7d5-5e1ea0c0ca12',
}

mock_headers = {
    'Host': 'http://w-v-kitslab-mxcube-1:8081',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Upgrade-Insecure-Requests': '1',
    'DNT': '1',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0',
}

real_cookies = {
    'session': '624f629e-b4e8-419f-9370-ab3dadce37c8',
}

real_headers = {
        'Host': 'http://b-v-biomax-web-0:8081',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
        }


APIURL = (real_headers if REAL else mock_headers)['Host'] + API_PATH


def reqstuff():
    if REAL:
        return dict(cookies=real_cookies, headers=real_headers, timeout=2)
    else:
        return dict(cookies=mock_cookies, headers=mock_headers, timeout=2)


def setblacklight(mode):
    requests.put(
            '{}/sampleview/backlight{}'.format(APIURL, 'on' if mode else 'off'),
            timeout=2,
            **reqstuff(),
            )


def get_stream() -> requests.Response:
    return requests.get(
            f'{APIURL}/sampleview/camera/subscribe',
            stream=True,
            **reqstuff(),
            )


def img_stream():
    stream = get_stream()
    if(stream.status_code == 200):
        img_buffer = bytes()
        for chunk in stream.iter_content(chunk_size=64):
            img_buffer += chunk
            start = img_buffer.find(b'\xff\xd8')
            end = img_buffer.find(b'\xff\xd9')
            if start != -1 and end != -1:
                jpg = img_buffer[start:end+2]
                img_buffer = img_buffer[end+2:]
                # i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                t = time()
                # yield (i, t)
                yield (jpg, t)
    else:
        print("Received unexpected status code {}".format(stream.status_code))


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
    impath = f'{datapath}/img/{uuid}.jpg'
    img = cv2.imread(impath)
    if not img:
        print(f'file "{impath}" could not be read for some reason.', file=sys.stderr)
        sys.exit(1)
    return img, timestamp


def create_db(path):
    conn = sql.connect(f'{path}/meta.db')
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            uuid VARCHAR(100) UNIQUE NOT NULL,
            timestamp INTEGER NOT NULL
        );
        """)
    return conn


def save_buffer(buf, path):
    from uuid import uuid4
    print('saving buffer to disk at {}'.format(time()))
    db = create_db(path)
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


def eval_fps(buf):
    delays = [b[1] - a[1] for a, b in zip(buf[:-1], buf[1:])]
    avg_delay = sum(delays)/len(delays)
    return 1/avg_delay


if __name__ == '__main__':
    PATH = 'data/cam'
    camera_buffer = []
    gen = img_stream()
    prev = time()
    print('watching camera stream...')
    last_save = time()
    while True:
        i, t = next(gen)
        camera_buffer.append((i, t))
        if not REAL:
            print(t)
        if t - last_save > 30:
            print('fps:{}'.format(eval_fps(camera_buffer)))
            save_buffer(camera_buffer, PATH)
            camera_buffer = []
            last_save = t
