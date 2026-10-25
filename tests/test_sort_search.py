import numpy as np
import pytest

from numcompute.sort_search import (binary_search, multikey_argsort, multikey_sort, quickselect,
                                    stable_sort, topk)


def test_stable_sort_ascending_descending_keeps_tie_order():
    a = np.array([3, 1, 2, 1, 3])
    s, idx = stable_sort(a, return_indices=True)
    assert s.tolist() == [1, 1, 2, 3, 3] and idx.tolist() == [1, 3, 2, 0, 4]
    s, idx = stable_sort(a, descending=True, return_indices=True)
    assert s.tolist() == [3, 3, 2, 1, 1] and idx.tolist() == [0, 4, 2, 1, 3]


def test_stable_sort_axis_and_strides(rng):
    X = rng.integers(0, 5, size=(6, 8))[:, ::2]
    np.testing.assert_array_equal(stable_sort(X, axis=0), np.sort(X, axis=0))
    np.testing.assert_array_equal(stable_sort(X, axis=None), np.sort(X, axis=None))
    np.testing.assert_array_equal(stable_sort(X, axis=1, descending=True), -np.sort(-X, axis=1))


def test_multikey_sort_mixed_directions_and_strings():
    X = np.array([["b", "2"], ["a", "3"], ["b", "1"], ["a", "1"]])
    out = multikey_sort(X, keys=[0, 1], descending=[False, True])
    assert out.tolist() == [["a", "3"], ["a", "1"], ["b", "2"], ["b", "1"]]
    with pytest.raises(IndexError):
        multikey_argsort(X, keys=[5])


def test_multikey_matches_lexsort(rng):
    X = rng.integers(0, 3, size=(50, 3))
    np.testing.assert_array_equal(multikey_argsort(X, [2, 0]), np.lexsort((X[:, 0], X[:, 2])))


@pytest.mark.parametrize("k", [0, 1, 5, 20])
def test_topk_extreme_k(rng, k):
    v = rng.normal(size=20)
    vals, idx = topk(v, k)
    np.testing.assert_array_equal(vals, np.sort(v)[::-1][:k])
    np.testing.assert_array_equal(v[idx], vals)
    small = topk(v, k, largest=False, return_indices=False)
    np.testing.assert_array_equal(small, np.sort(v)[:k])


def test_topk_ties_axis_and_errors():
    v = np.array([5, 1, 5, 3, 5])
    vals, idx = topk(v, 2)
    assert vals.tolist() == [5, 5] and idx.tolist() == [0, 2]      # ties by index
    M = np.array([[1, 9, 3], [7, 2, 8]])
    assert topk(M, 1, axis=1)[0].ravel().tolist() == [9, 8]
    assert topk(M, 1, axis=0)[0].ravel().tolist() == [7, 9, 8]
    with pytest.raises(ValueError):
        topk(v, 6)
    with pytest.raises(ValueError):
        topk(v, -1)


def test_quickselect_matches_sort_with_duplicates(rng):
    v = rng.integers(0, 4, size=101)                 # heavy duplicates
    s = np.sort(v)
    for k in [0, 1, 50, 99, 100, -1]:
        assert quickselect(v, k) == s[k]
    assert quickselect([7, 7, 7], 1) == 7             # all equal


def test_quickselect_errors():
    with pytest.raises(ValueError):
        quickselect([], 0)
    with pytest.raises(ValueError):
        quickselect([1.0, np.nan], 0)
    with pytest.raises(ValueError):
        quickselect([1, 2], 2)


def test_binary_search_scalar_vector_and_iterative():
    a = np.array([1, 3, 3, 5])
    assert binary_search(a, 3) == (1, True)
    assert binary_search(a, 4) == (3, False)
    assert binary_search(a, 3, side="right") == (3, True)
    idx, found = binary_search(a, [0, 3, 6])
    assert idx.tolist() == [0, 1, 4] and found.tolist() == [False, True, False]
    for x in [0, 1, 3, 4, 5, 6]:
        for side in ("left", "right"):
            assert binary_search(a, x, side=side, method="iterative") == binary_search(a, x, side=side)


def test_binary_search_empty_and_unsorted():
    assert binary_search([], 3) == (0, False)
    with pytest.raises(ValueError):
        binary_search([3, 1, 2], 1, check_sorted=True)
