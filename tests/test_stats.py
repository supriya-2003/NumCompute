import numpy as np
import pytest

from numcompute import stats


def test_basic_stats_axis_shapes(rng):
    X = rng.normal(size=(30, 4))
    assert stats.mean(X, axis=0).shape == (4,)
    assert stats.std(X, axis=1).shape == (30,)
    assert np.isscalar(stats.median(X)) or np.ndim(stats.median(X)) == 0
    np.testing.assert_allclose(stats.var(X, axis=0, ddof=1), np.var(X, axis=0, ddof=1))
    np.testing.assert_allclose(stats.minimum(X, axis=0), X.min(0))
    np.testing.assert_allclose(stats.maximum(X, axis=1, keepdims=True), X.max(1, keepdims=True))


def test_stats_nan_handling():
    x = np.array([1.0, np.nan, 3.0])
    assert np.isnan(stats.mean(x))
    assert stats.mean(x, skipna=True) == 2.0
    assert stats.median(x, skipna=True) == 2.0
    assert np.isnan(stats.mean(np.array([np.nan, np.nan]), skipna=True))  # no crash


def test_stats_errors():
    with pytest.raises(ValueError):
        stats.mean([])
    with pytest.raises(ValueError):
        stats.mean([[1.0]], axis=3)
    with pytest.raises(ValueError):
        stats.quantile([1, 2], 1.5)


def test_quantile_matches_numpy(rng):
    X = rng.normal(size=(20, 3))
    np.testing.assert_allclose(stats.quantile(X, [0.1, 0.5, 0.9], axis=0),
                               np.quantile(X, [0.1, 0.5, 0.9], axis=0))


@pytest.mark.parametrize("bins", [1, 7, 50])
def test_histogram_matches_numpy(rng, bins):
    x = rng.normal(size=5000)
    c, e = stats.histogram(x, bins=bins)
    c2, e2 = np.histogram(x, bins=bins)
    np.testing.assert_array_equal(c, c2)
    np.testing.assert_allclose(e, e2)


def test_histogram_edges_density_nan_and_constant(rng):
    x = np.append(rng.uniform(0, 1, 1000), [np.nan, np.inf])
    edges = [0, 0.1, 0.5, 1.0]
    c, _ = stats.histogram(x, bins=edges)
    np.testing.assert_array_equal(c, np.histogram(x[np.isfinite(x)], bins=edges)[0])
    d, e = stats.histogram(x, bins=10, density=True)
    assert np.isclose((d * np.diff(e)).sum(), 1.0)
    c, e = stats.histogram([5.0, 5.0, 5.0], bins=3)          # all equal values
    assert c.sum() == 3 and e[0] < 5 < e[-1]


def test_running_stats_matches_batch(rng):
    X = rng.normal(1e6, 1.0, size=(1000, 3))           # large offset: tests stability
    X[5, 1] = np.nan
    rs = stats.RunningStats()
    for chunk in np.array_split(X, 13):
        rs.update(chunk)
    np.testing.assert_allclose(rs.mean, np.nanmean(X, 0), rtol=1e-12)
    np.testing.assert_allclose(rs.var(ddof=1), np.nanvar(X, 0, ddof=1), rtol=1e-8)
    np.testing.assert_allclose(rs.min, np.nanmin(X, 0))
    assert rs.count.tolist() == [1000, 999, 1000]


def test_running_stats_merge_and_scalar_updates():
    a, b = stats.RunningStats(), stats.RunningStats()
    for v in [1.0, 2.0, 3.0]:
        a.update(v)
    b.update([4.0, 5.0])
    a.merge(b)
    assert a.mean[0] == 3.0 and np.isclose(a.var()[0], 2.0)
    with pytest.raises(RuntimeError):
        stats.RunningStats().var()


def test_streaming_histogram(rng):
    x = rng.normal(size=1000)
    sh = stats.StreamingHistogram(np.linspace(-2, 2, 9))
    for ch in np.array_split(x, 7):
        sh.update(ch)
    np.testing.assert_array_equal(sh.counts, np.histogram(x, bins=np.linspace(-2, 2, 9))[0])
    assert sh.n_out_of_range == int(((x < -2) | (x > 2)).sum())


def test_describe_keys():
    d = stats.describe(np.array([[1.0, 2.0], [3.0, np.nan]]))
    assert d["count"].tolist() == [2, 1] and d["n_missing"].tolist() == [0, 1]
