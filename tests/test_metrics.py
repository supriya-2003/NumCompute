import numpy as np
import pytest

from numcompute import metrics as M


def test_accuracy_and_errors():
    assert M.accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    assert M.accuracy([1, 0], [1, 0], normalize=False) == 2
    with pytest.raises(ValueError):
        M.accuracy([1, 0], [1])
    with pytest.raises(ValueError):
        M.accuracy([], [])


def test_confusion_matrix_basic_labels_and_normalize():
    yt = ["cat", "dog", "dog", "bird"]
    yp = ["cat", "dog", "cat", "bird"]
    cm = M.confusion_matrix(yt, yp)                    # labels: bird, cat, dog
    np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 1, 0], [0, 1, 1]])
    cm2 = M.confusion_matrix(yt, yp, labels=["dog", "cat"])
    np.testing.assert_array_equal(cm2, [[1, 1], [0, 1]])
    np.testing.assert_allclose(M.confusion_matrix(yt, yp, normalize="true").sum(1), 1)


def test_binary_precision_recall_f1():
    yt = [1, 1, 1, 0, 0, 0]
    yp = [1, 1, 0, 1, 0, 0]
    assert M.precision(yt, yp) == pytest.approx(2 / 3)
    assert M.recall(yt, yp) == pytest.approx(2 / 3)
    assert M.f1(yt, yp) == pytest.approx(2 / 3)
    assert M.precision(yt, yp, pos_label=0) == pytest.approx(2 / 3)


def test_multiclass_averaging():
    yt = [0, 1, 2, 2, 1, 0]
    yp = [0, 2, 2, 2, 1, 1]
    p, r, f = M.precision_recall_f1(yt, yp, average=None)
    np.testing.assert_allclose(p, [1.0, 0.5, 2 / 3])
    np.testing.assert_allclose(r, [0.5, 0.5, 1.0])
    assert M.precision(yt, yp, average="macro") == pytest.approx(np.mean([1.0, 0.5, 2 / 3]))
    assert M.f1(yt, yp, average="micro") == pytest.approx(M.accuracy(yt, yp))
    w = M.recall(yt, yp, average="weighted")
    assert w == pytest.approx((0.5 * 2 + 0.5 * 2 + 1.0 * 2) / 6)
    with pytest.raises(ValueError, match="multiclass"):
        M.precision(yt, yp, average="binary")


def test_zero_division():
    assert M.precision([1, 1], [0, 0]) == 0.0           # no positive predictions
    assert M.precision([1, 1], [0, 0], zero_division=1.0) == 1.0


def test_regression_metrics():
    yt, yp = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    assert M.mse(yt, yp) == pytest.approx(4 / 3)
    assert M.rmse(yt, yp) == pytest.approx(np.sqrt(4 / 3))
    assert M.mae(yt, yp) == pytest.approx(2 / 3)
    assert M.r2_score(yt, yt) == 1.0
    np.testing.assert_allclose(M.mse(np.ones((3, 2)), np.zeros((3, 2)), axis=0), [1, 1])
    with pytest.raises(ValueError):
        M.mse([1, 2], [1, 2, 3])


def test_roc_curve_known_values_and_ties():
    yt = [0, 0, 1, 1]
    ys = [0.1, 0.4, 0.35, 0.8]
    fpr, tpr, thr = M.roc_curve(yt, ys)
    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
    assert thr[0] == np.inf
    assert M.roc_auc_score(yt, ys) == pytest.approx(0.75)
    assert M.roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)  # all tied


def test_roc_auc_rank_equivalence(rng):
    yt = rng.integers(0, 2, 300)
    ys = rng.random(300) + 0.5 * yt
    pos, neg = ys[yt == 1], ys[yt == 0]
    expected = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    assert M.roc_auc_score(yt, ys) == pytest.approx(expected)


def test_roc_errors_and_auc():
    with pytest.raises(ValueError):
        M.roc_curve([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError):
        M.roc_curve([0, 1], [np.nan, 0.3])
    assert M.auc([0, 1], [0, 1]) == 0.5
    with pytest.raises(ValueError):
        M.auc([0, 2, 1], [0, 1, 1])
