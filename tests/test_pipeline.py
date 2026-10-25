import numpy as np
import pytest

from numcompute.metrics import accuracy
from numcompute.pipeline import (ColumnTransformer, FeatureUnion, FunctionTransformer,
                                 LeastSquaresRegressor, LogisticRegressionGD, MajorityClassifier,
                                 MeanRegressor, Pipeline, make_pipeline)
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_pipeline_spec_example():
    X = np.array([[1.0], [2.0], [1.0], [3.0]])
    pipe = Pipeline([("scale", StandardScaler()), ("encode", OneHotEncoder())])
    X_tr = pipe.fit_transform(X)
    assert X_tr.shape == (4, 3)
    np.testing.assert_array_equal(pipe.transform(X), X_tr)
    assert isinstance(pipe.named_steps["scale"], StandardScaler)
    assert isinstance(pipe[0], StandardScaler) and len(pipe) == 2


def test_pipeline_equals_manual_chain(rng):
    X = rng.normal(size=(40, 3))
    X[0, 0] = np.nan
    pipe = make_pipeline(SimpleImputer(), StandardScaler(), MinMaxScaler())
    manual = MinMaxScaler().fit_transform(StandardScaler().fit_transform(SimpleImputer().fit_transform(X)))
    np.testing.assert_allclose(pipe.fit_transform(X), manual)
    assert list(pipe.named_steps) == ["simpleimputer", "standardscaler", "minmaxscaler"]


def test_pipeline_with_estimator_predict(rng):
    X = rng.normal(size=(100, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1
    pipe = Pipeline([("scale", StandardScaler()), ("model", LeastSquaresRegressor())])
    pred = pipe.fit(X, y).predict(X)
    np.testing.assert_allclose(pred, y, atol=1e-8)
    with pytest.raises(AttributeError):
        pipe.transform(X)


def test_pipeline_validation_errors():
    with pytest.raises(TypeError):
        Pipeline([("model", MeanRegressor()), ("scale", StandardScaler())])
    with pytest.raises(ValueError):
        Pipeline([("a", StandardScaler()), ("a", MinMaxScaler())])
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_passthrough_and_params():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipe = Pipeline([("skip", "passthrough"), ("scale", MinMaxScaler())])
    np.testing.assert_array_equal(pipe.fit_transform(X), [[0, 0], [1, 1]])
    sc = StandardScaler().set_params(with_mean=False)
    assert sc.get_params()["with_mean"] is False
    with pytest.raises(ValueError):
        sc.set_params(bogus=1)


def test_feature_union_and_function_transformer():
    X = np.array([[1.0], [2.0], [3.0]])
    fu = FeatureUnion([("std", StandardScaler()), ("sq", FunctionTransformer(np.square))])
    out = fu.fit_transform(X)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out[:, 1], [1, 4, 9])


def test_column_transformer_mixed_types():
    X = np.array([[1.0, "a"], [2.0, "b"], [3.0, "a"]], dtype=object)
    ct = ColumnTransformer([("num", StandardScaler(), [0]), ("cat", OneHotEncoder(), [1])])
    out = ct.fit_transform(X)
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:, 1:], [[1, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(ct.transform(X), out)
    keep = ColumnTransformer([("num", "passthrough", [0])], remainder="drop").fit_transform(X)
    assert keep.shape == (3, 1)


def test_reference_estimators(rng):
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = LogisticRegressionGD(lr=0.5, n_iter=300).fit(X, y)
    assert accuracy(y, clf.predict(X)) > 0.95
    np.testing.assert_allclose(clf.predict_proba(X).sum(1), 1)
    maj = MajorityClassifier().fit(X, np.array([1, 1, 0] * 66 + [1, 1]))
    assert set(maj.predict(X)) == {1}
    assert MeanRegressor().fit(X, np.arange(200.0)).predict(X[:2]).tolist() == [99.5, 99.5]
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit(X, np.arange(200) % 3)
