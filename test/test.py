from diffractometrics import QueueEntry


def max_val(matches: list):
    return max(filter(lambda v: v is not None, matches))


def test_match_closest_distance():
    li = [1, 5, 7, 11]
    lj = [3, 5, 12]
    matches = QueueEntry.match_closest_distance(li, lj, threshold=10)
    expected = [0, 1, None, 2]
    assert max_val(matches) < len(lj)
    assert all([a == b for a, b in zip(matches, expected)])

    li = [0.1, 0.7, 0.8, 3, 3.4, 3.6]
    lj = [3.1, 3.9]
    matches = QueueEntry.match_closest_distance(li, lj, threshold=10)
    expected = [None, None, None, 0, None, 1]
    assert max_val(matches) < len(lj)
    assert all([a == b for a, b in zip(matches, expected)])

    li = [0.1, 0.7, 0.8, 3, 3.4, 3.6]
    lj = [0.2, 0.2, 1, 3.6, 3.6]
    matches = QueueEntry.match_closest_distance(li, lj, threshold=0.2)
    expected = [0, None, 2, None, None, 3]
    assert max_val(matches) < len(lj)
    assert all([a == b for a, b in zip(matches, expected)])

    li = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    lj = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    matches = QueueEntry.match_closest_distance(li, lj, threshold=0.2)
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert max_val(matches) < len(lj)
    assert all([a == b for a, b in zip(matches, expected)])
