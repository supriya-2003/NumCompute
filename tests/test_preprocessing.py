import numpy as np
import pytest

from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_standard_scaler_zero_mean_unit_var(rng):
    X = rng.normal(5, 3, size=(200, 4))
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(0), 1, atol=1e-12)


def test_standard_scaler_constant_column_and_inverse(rng):
    X = np.column_stack([np.full(10, 3.0), rng.normal(size=10)])
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    assert np.all(Z[:, 0] == 0)                       
    np.testing.assert_allclose(sc.inverse_transform(Z), X)


def test_standard_scaler_ignores_nan_in_fit():
    X = np.array([[1.0], [np.nan], [3.0]])
    sc = StandardScaler().fit(X)
    assert sc.mean_[0] == 2.0
    assert np.isnan(sc.transform(X)[1, 0])


def test_standard_scaler_non_contiguous_input(rng):
    big = rng.normal(size=(50, 6))
    X = big[::2, ::3]                                 # strided view
    assert not X.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(StandardScaler().fit_transform(X),
                               StandardScaler().fit_transform(np.ascontiguousarray(X)))


def test_scaler_errors():
    with pytest.raises(RuntimeError):
        StandardScaler().transform([[1.0]])
    with pytest.raises(ValueError, match="2D"):
        StandardScaler().fit([1.0, 2.0])
    sc = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="features"):
        sc.transform([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="empty"):
        StandardScaler().fit(np.empty((0, 2)))
    with pytest.raises(TypeError):
        StandardScaler().fit([["a", "b"]])


def test_minmax_scaler_range_and_constant():
    X = np.array([[0.0, 5.0], [5.0, 5.0], [10.0, 5.0]])
    out = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    np.testing.assert_allclose(out[:, 0], [-1, 0, 1])
    np.testing.assert_allclose(out[:, 1], -1)          # constant -> lower bound
    mm = MinMaxScaler(clip=True).fit(X)
    assert mm.transform([[20.0, 5.0]])[0, 0] == 1.0
    np.testing.assert_allclose(MinMaxScaler().fit(X).inverse_transform(MinMaxScaler().fit_transform(X)), X)
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0)).fit(X)


@pytest.mark.parametrize("strategy,expected", [("mean", 2.0), ("median", 1.0),
                                                ("most_frequent", 1.0), ("constant", 0.0)])
def test_simple_imputer(strategy, expected):
    X = np.array([[1.0], [1.0], [np.nan], [4.0]])
    out = SimpleImputer(strategy=strategy).fit_transform(X)
    assert out[2, 0] == expected
    assert not np.isnan(out).any()


def test_simple_imputer_all_nan_column():
    X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    out = SimpleImputer(fill_value=-9).fit_transform(X)
    np.testing.assert_array_equal(out[:, 0], [-9, -9])


def test_one_hot_strings_and_feature_names():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]])
    enc = OneHotEncoder()
    out = enc.fit_transform(X)
    np.testing.assert_array_equal(out, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
    assert enc.get_feature_names(["color", "size"]) == ["color=blue", "color=red", "size=M", "size=S"]
    np.testing.assert_array_equal(enc.inverse_transform(out), X)


def test_one_hot_unknown_handling_and_drop_first():
    enc = OneHotEncoder().fit([[1], [2]])
    with pytest.raises(ValueError, match="unknown"):
        enc.transform([[3]])
    out = OneHotEncoder(handle_unknown="ignore").fit([[1], [2]]).transform([[3], [2]])
    np.testing.assert_array_equal(out, [[0, 0], [0, 1]])
    assert OneHotEncoder(drop_first=True).fit_transform([[1], [2], [3]]).shape == (3, 2)


def test_one_hot_nan_category_and_all_equal():
    out = OneHotEncoder().fit_transform(np.array([[1.0], [np.nan], [1.0]]))
    np.testing.assert_array_equal(out, [[1, 0], [0, 1], [1, 0]])
    assert OneHotEncoder().fit_transform([[7], [7], [7]]).tolist() == [[1], [1], [1]]
