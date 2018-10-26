import sys
import h5py as h5

DATE_PATH = 'entry/instrument/detector/detectorSpecific/data_collection_date'


def get_timestamps(h5filepath: str, num_frames: int) -> list:
    from datetime import datetime

    f = h5.File(h5filepath)
    try:
        datestr = f[DATE_PATH].value
    except KeyError:
        print(
            f"File '{h5filepath}' does not have dataset '{DATE_PATH}, exiting.'",
            file=sys.stderr,
        )
        sys.exit(1)
    start_time = datetime.strptime(datestr.decode(), '%Y-%m-%dT%H:%M:%S.%f')
    exposure_time = f['entry/instrument/detector/frame_time'].value
    readout_time = f['entry/instrument/detector/detector_readout_time'].value
    frame_time = exposure_time + readout_time
    return [start_time.timestamp() + frame_time * i for i in range(num_frames)]


if __name__ == '__main__':
    somepath = (
        '/home/isak/maxiv-data/common/ML-crystals/h5/1149_2-Lysozyme_10_master.h5'
    )
    print(get_timestamps(somepath, 10))
