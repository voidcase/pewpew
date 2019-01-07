from pathlib import Path
from datetime import datetime
import numpy as np
import h5py as h5
import config as cfg
import json
import re
import sys
import utils


class QueueEntry:
    def __init__(self, meta: dict):
        self.meta = meta
        self.master_file = Path(meta["fileinfo"]["filename"])
        self.sample_dir = self.master_file.parent
        self.nbr_frames = meta["oscillation_sequence"][0]["number_of_images"]
        self.exposure_time = meta["oscillation_sequence"][0]["exposure_time"]
        self.zoom = meta.get("zoom1", None)
        self.frontlight = meta.get("backlight", None)
        self.prefix = f"{meta['fileinfo']['prefix']}_{meta['fileinfo']['run_number']}"
        self.date = utils.get_date(self.sample_dir)

    def write_data_pairs(self):
        thres = 0.5
        pairs = self.find_closest_images(self.images(), self.get_timestamps_log(), threshold=thres)
        name = f"{self.sample_dir.name}_{self.prefix}_{thres}.json"
        date_dir = Path(cfg.DATA_DIR) / self.__get_date()
        if not date_dir.exists():
            date_dir.mkdir()
        with open(date_dir / name, "w") as f:
            json.dump(pairs, f, indent=4)

    def __get_date(self):
        return re.findall(r'\d{8}', str(self.master_file))[1]

    def images(self):
        return sorted((self.sample_dir / cfg.SNAPSHOT_DIR).glob(f"{self.prefix}*.jpeg"), key=lambda p: p.name)

    def images_to_timestamps(self, images: list):
        return [float(img.stem[len(self.prefix) + 1:]) for img in images]

    @staticmethod
    def match_closest_distance(li: list, lj: list, threshold: float):
        """[1,5,7,11], [3, 5, 12] -> [0, 1, None, 2]"""
        if threshold < 0:
            raise Exception('threshold can not be less than 0')
        if np.any(np.append(np.diff(li) < 0, np.diff(lj) < 0)):
            raise Exception('One or both lists are not sorted')

        if len(li) < len(lj):
            li, lj = lj, li

        i = 0
        j = 0
        matches = [None for _ in range(len(li))]
        while i < len(li) - 1 and j < len(lj):
            lis = np.array([li[i], li[i + 1]])
            if lj[j] > li[i + 1] and i + 2 < len(li):
                i += 1
                continue

            diff = np.abs(lis - lj[j])
            closest_idx = np.argmin(diff)

            if diff[closest_idx] > threshold:
                j += 1
                continue

            matches[i + closest_idx] = j
            j += 1
            i += (1 + closest_idx)

            # One element left
            if len(li[i:]) == 1 and j < len(lj):
                if np.abs(li[i] - lj[j]) > threshold:
                    continue
                matches[i] = j

        return matches

    def find_closest_images(self, images: list, timestamps: list, threshold: float = 0.5):
        print(f'matching pairs for {self.master_file}')
        images_ts = self.images_to_timestamps(images)
        time_diff = images_ts[0] - timestamps[0]
        images_ts = np.array(images_ts) - time_diff
        matches = QueueEntry.match_closest_distance(images_ts, timestamps, threshold)
        pairs = {}
        for i, img in enumerate(images):
            if matches[i] is None:
                continue
            pairs[str(img)] = matches[i]
        return pairs

    def get_timestamps_log(self):
        log = Path(f'/data/staff/common/ML-crystals/eiger_logs/{self.date}.log')
        sample, scan = '/'.join(Path(self.master_file).parts[-2:])[:-10].split('/')
        start_re = re.compile(r'^Starting.*{}\/{}_data_000001.*at\s(\d.+)$'.format(sample, scan))
        finish_re = re.compile(r'^Finished.*{}\/{}_data_000001.*at\s(\d.+)$'.format(sample, scan))
        matches = []
        with open(log, 'r') as f:
            for line in f:
                # No point in matching second match if we haven't found first
                match = start_re.match(line) if len(matches) == 0 else finish_re.match(line)
                if match is not None:
                    matches.append(utils.fmt(match.group(1)).timestamp())
        if len(matches) != 2:
            raise Exception(f'Could not find start and/or finish time for {self.master_file}')

        duration = self.nbr_frames * self.exposure_time
        diff = matches[1] - matches[0]
        start_time = matches[0] + (diff - duration)
        return [start_time + (frame_nbr * self.exposure_time) for frame_nbr in range(self.nbr_frames)]

    def get_timestamps(self) -> list:
        DATEPATH = 'entry/instrument/detector/detectorSpecific/data_collection_date'
        f = h5.File(str(self.master_file))
        try:
            datestr = f[DATEPATH].value
        except KeyError:
            print(f'file "{self.master_file}" does not have dataset "{DATEPATH}, exiting."', file=sys.stderr)
            sys.exit(1)
        start_time = datetime.strptime(datestr.decode(), '%Y-%m-%dT%H:%M:%S.%f')
        exposure_time = f['entry/instrument/detector/frame_time'].value
        readout_time = f['entry/instrument/detector/detector_readout_time'].value
        frame_time = exposure_time + readout_time
        return [start_time.timestamp() + frame_time * i for i in range(self.nbr_frames)]

    def __str__(self):
        return str(Path(self.master_file))

    def __repr__(self):
        return self.__str__()
