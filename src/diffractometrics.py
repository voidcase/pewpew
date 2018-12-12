from pathlib import Path
from datetime import datetime
import numpy as np
import h5py as h5
import config as cfg
import json
import re
import sys


LOG_TIMESTAMPS_WHITELIST = {
    'Sample-3-01',
    'Sample-4-01',
    'Sample-3-10',
    'Sample-4-02',
    'Sample-4-08',
    'Sample-3-02',
    'Sample-4-09',
    'Sample-2-07',
    'Sample-1-03',
    'Sample-1-05',
    'Sample-4-04',
    'Sample-3-05',
    'Sample-3-11',
    'Sample-4-14',
    'Sample-4-15',
    'Sample-3-12',
    'Sample-4-16',
    'Sample-3-03',
    'Sample-1-02',
    'Sample-3-04',
    'Sample-3-09',
    'Sample-4-12',
    'Sample-4-06',
    'Sample-4-07',
    'Sample-3-15',
    'Sample-3-16',
    'Sample-3-08',
    'Sample-3-06',
    'Sample-4-03',
    'Sample-4-11',
    'Sample-3-13',
    'Sample-4-13',
    'Sample-2-08',
    'Sample-3-07',
}


class QueueEntry:
    DEFAULT_TIME_OFFSET = 99.992_831_366_402_77

    def __init__(self, meta):
        self.meta = meta
        self.master_file = Path(meta["fileinfo"]["filename"])
        self.sample_dir = self.master_file.parent
        self.nbr_frames = meta["oscillation_sequence"][0]["number_of_images"]
        self.exposure_time = meta["oscillation_sequence"][0]["exposure_time"]
        self.zoom = meta.get("zoom1", None)
        self.frontlight = meta.get("backlight", None)
        self.prefix = f"{meta['fileinfo']['prefix']}_{meta['fileinfo']['run_number']}"

    def write_data_pairs(self):
        thres = 0.5
        pairs = self.__find_closest_images(thres)
        name = f"{self.sample_dir.name}_{self.prefix}_{thres}.json"
        with open(cfg.DATA_DIR / name, "w") as f:
            json.dump(pairs, f, indent=4)

    def __find_closest_images(self, threshold=0.5):
        print(f'matching pairs for {self.master_file}')
        data_pairs = {}
        timestamps = (
            self.__get_timestamps_log() if self.sample_dir.name in LOG_TIMESTAMPS_WHITELIST else self.__get_timestamps()
        )
        candidates = sorted((self.sample_dir / cfg.SNAPSHOT_DIR).glob(f"{self.prefix}*.jpeg"), key=lambda p: p.name)

        ts_i = 0
        i = 0
        while i < len(candidates) - 1:
            # We found matches for all timestamps
            if ts_i == len(timestamps):
                break
            ts = timestamps[ts_i]
            img_ts = (
                    np.array(
                        [
                            float(candidates[i].stem[len(self.prefix) + 1:]),
                            float(candidates[i + 1].stem[len(self.prefix) + 1:]),
                        ]
                    )
                    - self.DEFAULT_TIME_OFFSET
            )

            # Skip images taken before collection
            if ts > img_ts[1]:
                i += 2
                continue

            diff = np.abs(img_ts - ts)
            closest_idx = np.argmin(diff)

            # Only add images if they are a taken close enough
            if diff[closest_idx] < threshold:
                data_pairs[candidates[i + closest_idx].name] = ts_i + 1
                ts_i += 1
                i += 2
            else:  # Skip frame if there is not image taken close enough
                ts_i += 1
                continue

        return data_pairs

    def __get_timestamps_log(self):
        fmt = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f').timestamp()
        log = Path('/data/staff/common/ML-crystals/eiger_logs/timestamps.log')
        sample, scan = '/'.join(Path(self.master_file).parts[-2:])[:-10].split('/')
        start_re = re.compile(r'^Starting.*{}\/{}_data_000001.*at\s(\d.+)$'.format(sample, scan))
        finish_re = re.compile(r'^Finished.*{}\/{}_data_000001.*at\s(\d.+)$'.format(sample, scan))
        matches = []
        with open(log, 'r') as f:
            for line in f:
                # No point in matching second match if we haven't found first
                match = start_re.match(line) if len(matches) == 0 else finish_re.match(line)
                if match is not None:
                    matches.append(fmt(match.group(1)))
        if len(matches) != 2:
            raise Exception(f'Could not find start and/or finish time for {self.master_file}')

        duration = self.nbr_frames * self.exposure_time
        diff = matches[1] - matches[0]
        start_time = matches[0] + (diff - duration)
        return [start_time + (frame_nbr * self.exposure_time) for frame_nbr in range(self.nbr_frames)]

    def __get_timestamps(self) -> list:
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
