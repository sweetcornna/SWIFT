from swift.core import (
    compute_minimum_distance,
    compute_path_length,
    compute_path_smoothness,
)


def test_compute_path_length_sums_segments():
    points = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 12.0)]

    assert compute_path_length(points) == 17.0


def test_compute_path_smoothness_is_zero_for_straight_path():
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

    assert compute_path_smoothness(points) == 0.0


def test_compute_minimum_distance_between_two_tracks():
    ownship = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    intruder = [(0.0, 3.0, 0.0), (2.0, 4.0, 0.0)]

    assert compute_minimum_distance(ownship, intruder) == 3.0
