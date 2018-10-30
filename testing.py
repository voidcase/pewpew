import dataset as ds


def test_num_frames():
    from diffractometrics import number_frames
    path = ds.cfg.PATH_DIR_H5 / '1149_2-Lysozyme_5_master.h5'
    n = number_frames(path)
    print('number of frames:', n)
    assert isinstance(n, int)
    assert n > 0


def test_signal_strength():
    from diffractometrics import signal_strength
    cbfpath = ds.cfg.PATH_DIR_CBF / '1149_2-Lysozyme_5_master.h5' / 'out000020.cbf'
    ss = signal_strength(cbfpath)
    assert isinstance(ss, int)


# def test_compile():
#     print('compiling dataset')
#     ds.compile_dataset()
#     print('done!')


def test_extract_ys():
    path = ds.cfg.PATH_DIR_H5 / '1149_2-Lysozyme_5_master.h5'
    ys = ds.extract_ys(path)
    print(ys)
    assert isinstance(ys, list)
    assert all(y >= 0 for y in ys)
