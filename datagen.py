import cv2
import requests
import numpy as np
from time import time

def get_stream():
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

    return requests.get(
            'http://w-v-kitslab-mxcube-1:8081/mxcube/api/v0.1/sampleview/camera/subscribe',
            headers=headers,
            cookies=cookies,
            stream=True,
            timeout=2
            )


def grid(origin, distance, height, width):
    """
    origin: (x, y)
    """
    return [
        (origin[0] + distance * i, origin[1] + distance * j)
        for i in range(width)
        for j in range(height)
        ]

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
                yield (i, time())
    else:
        print("Received unexpected status code {}".format(stream.status_code))



if __name__ == '__main__':
    gen = img_stream()
    prev = time()
    while True:
        i, t = next(gen)
        print(t - prev)
        prev = t
