import cv2
import requests
import numpy as np
from time import time
import sqlite3 as sql

cookies = {
    'session': '4eb6807d-ed71-46d8-a7d5-5e1ea0c0ca12',
}
headers = {
    'Host': 'w-v-kitslab-mxcube-1:8081',
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


def setblacklight(mode):
    requests.put(
            'http://b-v-biomax-web-0:8081/mxcube/api/v0.1/sampleview/backlight{}'.format('on' if mode else 'off'),
            cookies=real_cookies,
            headers=real_headers,
            timeout=2,
            )


def get_stream() -> requests.Response:
    return requests.get(
            'http://b-v-biomax-web-0:8081/mxcube/api/v0.1/sampleview/camera/subscribe',
            headers=real_headers,
            cookies=real_cookies,
            stream=True,
            timeout=2
            )


def img_stream():
    stream = get_stream()
    if(stream.status_code == 200):
        img_buffer = bytes()
        for chunk in stream.iter_content(chunk_size=256):
            img_buffer += chunk
            start = img_buffer.find(b'\xff\xd8')
            end = img_buffer.find(b'\xff\xd9')
            if start != -1 and end != -1:
                jpg = img_buffer[start:end+2]
                img_buffer = img_buffer[end+2:]
                i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                t = time()
                yield (i, t)
    else:
        print("Received unexpected status code {}".format(stream.status_code))


def find_frame(buf: list, timestamp: float) -> tuple:
    def residual(t):
        return abs(timestamp - t)
    closest_frame = None
    closest_time = None
    for frame, frame_time in buf:
        if not closest_frame or residual(frame_time) < residual(closest_time):
            closest_frame = frame
            closest_time = frame_time
    return (closest_frame, closest_time)


def test_find_frame():
    b = [
        ('a', 1.5),
        ('b', 2.7),
        ('c', 4.7),
    ]
    f, t = find_frame(b, 2.6)
    assert f == 'b'
    assert t == 2.7


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
    for img, timestamp in buf:
        image_id = uuid4()
        cv2.imwrite(f'{path}/images/{image_id}.jpg', img)
        query += f'("{image_id}", "{timestamp}"),'
    query = query[:-1]  # remove last ',' from query
    query += ';'
    db.execute(query)
    db.commit()
    db.close()


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
        if t - last_save > 60:
            save_buffer(camera_buffer, PATH)
            camera_buffer = []
            last_save = t
