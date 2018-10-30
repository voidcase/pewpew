from pathlib import Path


class DevConfig:
    _PATH_DIR_PROJECT = Path('/mnt/maxiv/common/ML-crystals/')
    PATH_DIR_H5 = _PATH_DIR_PROJECT / 'h5'
    PATH_DIR_CBF = _PATH_DIR_PROJECT / 'cbf'
    PATH_DIR_CAM = Path('./data/cam/images/')
    PATH_DB = Path('./data/cam/meta.db')


class TestConfig:
    _PATH_DIR_PROJECT = Path('data/testing')
    PATH_DIR_H5 = Path('/mnt/maxiv/common/ML-crystals/h5')
    PATH_DIR_CAM = _PATH_DIR_PROJECT / 'cam'
    PATH_DIR_CBF = _PATH_DIR_PROJECT / 'cbf'
    PATH_DB = _PATH_DIR_PROJECT / 'meta.db'
