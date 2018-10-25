import requests
from time import time
from dataset import save_buffer

# REAL = True
REAL = False

URL = "http://b-v-biomax-web-0:8081" if REAL else "http://w-v-kitslab-mxcube-1:8081"
API_PATH = f"{URL}/mxcube/api/v0.1"

auth = dict(proposal="idtest000", password="")
# login = dict(proposal='', password='')


def setup_session() -> requests.Session:
    s = requests.Session()
    res = s.post(f"{API_PATH}/login", json=auth)
    if not res.ok:
        raise Exception(f"Error {res.status_code}: {res.reason}")
    return s


def setblacklight(mode: bool):
    res = s.put(f"{API_PATH}/sampleview/backlight{'on' if mode else 'off'}", timeout=2)
    if not res.ok:
        raise Exception(f"Error {res.status_code}: {res.reason}")


def get_stream() -> requests.Response:
    return s.get(f"{API_PATH}/sampleview/camera/subscribe", stream=True)


def img_stream():
    stream = get_stream()
    if stream.status_code == 200:
        img_buffer = bytes()
        for chunk in stream.iter_content(chunk_size=64):
            img_buffer += chunk
            start = img_buffer.find(b"\xff\xd8")
            end = img_buffer.find(b"\xff\xd9")
            if start != -1 and end != -1:
                jpg = img_buffer[start : end + 2]
                img_buffer = img_buffer[end + 2 :]
                # i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                t = time()
                # yield (i, t)
                yield (jpg, t)
    else:
        print("Received unexpected status code {}".format(stream.status_code))


def eval_fps(buf):
    delays = [b[1] - a[1] for a, b in zip(buf[:-1], buf[1:])]
    avg_delay = sum(delays) / len(delays)
    return 1 / avg_delay


if __name__ == "__main__":
    PATH = "data/cam"
    camera_buffer = []
    s = setup_session()
    gen = img_stream()
    prev = time()
    print("watching camera stream...")
    last_save = time()
    while True:
        i, t = next(gen)
        camera_buffer.append((i, t))
        if not REAL:
            print(t)
        if t - last_save > 30:
            print("fps:{}".format(eval_fps(camera_buffer)))
            save_buffer(camera_buffer, PATH)
            camera_buffer = []
            last_save = t
