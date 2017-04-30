import sys
import requests
import itertools
import time
from multiprocessing import Pool

def get_key():
    with open("API_KEY", "r") as f:
        key = f.readline().strip()
    return key


API_KEY = get_key()
ENDPOINT = "https://maps.googleapis.com/maps/api/staticmap"
SIZE = "512x512"
ZOOM = 16


def get(center, maptype, num):
    req = requests.get(ENDPOINT,
                       params={
                            'center': center,
                            'size': SIZE,
                            'zoom': ZOOM,
                            'maptype': maptype,
                            'key': API_KEY,
                            'style': 'element:labels|visibility:off',
                        },
                       stream=True)

    if req.status_code == requests.codes.ok:
        if maptype == 'roadmap':
            pre = "maps/roadmap/pit"
        else:
            pre = "maps/satellite/pit"
        output = pre + str(num) + ".png"
        with open(output, "wb") as fd:
            for chunk in req.iter_content(chunk_size=128):
                fd.write(chunk)
    else:
        print(req.status_code)
        print("failed to get %s -> %s" % (center, maptype))
        time.sleep(2)
        get(center, maptype, num)
    return req


def do(x):
    num = x[0]
    center, maptype = x[1]
    print(center, maptype)
    get(center, maptype, num)


def frange(start, stop, inc=1):
    if start < stop and inc > 0:
        while start < stop:
            yield start
            start += inc
    elif start > stop and inc < 0:
        while start > stop:
            yield start
            start += inc


def make_map(p1, p2, num_points=50):
    (x1, y1) = p1
    (x2, y2) = p2
    xs = list(frange(x1, x2, (x2 - x1) / num_points))
    ys = list(frange(y1, y2, (y2 - y1) / num_points))
    points = itertools.product(xs, ys)

    def mapper(x, y):
        return [(str(x) + "," + str(y), b) for b in ['roadmap', 'satellite']]

    instructions = [(i, x) for (i, p) in enumerate(points) for x in mapper(*p)]
    print(instructions)

    pool = Pool(12)
    pool.map(do, instructions)


p1 = (40.466423, -79.989500)
p2 = (40.443124, -79.947829)
make_map(p1, p2)
# r = get("0, 0", "roadmap", 1)
