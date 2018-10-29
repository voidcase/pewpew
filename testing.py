import dataset as ds


def test_compile():
    print('compiling dataset')
    ds.compile_dataset()
    print('done!')


def test_extract_ys():
    path = ds.cfg.PATH_DIR_H5 / '1149_2-Lysozyme_5_master.h5'
    ys = ds.extract_ys(path)
    print(ys)


def test_num_frames():
    from diffractometrics import number_frames
    path = ds.cfg.PATH_DIR_H5 / '1149_2-Lysozyme_5_master.h5'
    n = number_frames(path)
    print(n)


if __name__ == '__main__':
    test_extract_ys()
