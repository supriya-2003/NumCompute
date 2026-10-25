import numpy as np
import pytest

from numcompute.rank import percentile, rank


@pytest.mark.parametrize("method,expected", [
    ("average", [1, 2.5, 2.5, 4]),
    ("min", [1, 2, 2, 4]),
    ("max", [1, 3, 3, 4]),
    ("dense", [1, 2, 2, 3]),
    ("ordinal", [1, 2, 3, 4]),
])
def test_rank_tie_methods(method, expected):
    np.testing.assert_array_equal(rank([10, 20, 20, 30], method=method), expected)


def test_rank_descending_all_equal_and_strings():
    np.testing.assert_array_equal(rank([10, 20, 20, 30], "dense", ascending=False), [3, 2, 2, 1])
    np.testing.assert_array_equal(rank([4, 4, 4], "average"), [2, 2, 2])
    np.testing.assert_array_equal(rank(["b", "a", "c"], "ordinal"), [2, 1, 3])


def test_rank_nan_keep_and_raise():
    r = rank([3.0, np.nan, 1.0], "average")
    assert np.isnan(r[1]) and r[0] == 2 and r[2] == 1
    with pytest.raises(ValueError):
        rank([1.0, np.nan], nan_policy="raise")


def test_rank_axis_and_empty():
    X = np.array([[3, 1, 2], [1, 1, 0]])
    np.testing.assert_array_equal(rank(X, "min", axis=1), [[3, 1, 2], [2, 2, 1]])
    assert rank([], "average").shape == (0,)
    with pytest.raises(ValueError):
        rank([1, 2], method="bogus")


@pytest.mark.parametrize("interp", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_percentile_matches_numpy(rng, interp):
    X = rng.normal(size=(17, 5))
    q = [0, 10, 33.3, 50, 99, 100]
    for axis in (None, 0, 1):
        np.testing.assert_allclose(percentile(X, q, interp, axis=axis),
                                   np.percentile(X, q, axis=axis, method=interp))


def test_percentile_nan_policies():
    x = np.array([1.0, np.nan, 3.0, 5.0])
    assert np.isnan(percentile(x, 50))
    assert percentile(x, 50, nan_policy="omit") == 3.0
    with pytest.raises(ValueError):
        percentile(x, 50, nan_policy="raise")
    X = np.array([[np.nan, np.nan], [1.0, 2.0]])
    out = percentile(X, 50, axis=1, nan_policy="omit")
    assert np.isnan(out[0]) and out[1] == 1.5


def test_percentile_errors_and_single_value():
    assert percentile([7.0], 90) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)
