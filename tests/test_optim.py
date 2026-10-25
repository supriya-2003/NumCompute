import numpy as np
import pytest

from numcompute.optim import check_grad, gradient_descent, grad, jacobian, line_search


def quad(v):
    return float(np.sum(v ** 2) + 3 * v[0] * v[1])


def quad_grad(v):
    return np.array([2 * v[0] + 3 * v[1], 2 * v[1] + 3 * v[0]])


@pytest.mark.parametrize("method,tol", [("central", 1e-8), ("forward", 1e-4), ("backward", 1e-4)])
def test_grad_methods_accuracy(method, tol):
    x = np.array([1.5, -2.0])
    np.testing.assert_allclose(grad(quad, x, method=method), quad_grad(x), atol=tol)


def test_grad_scalar_input_and_large_x():
    assert grad(np.sin, 0.0) == pytest.approx(1.0, abs=1e-9)
    assert grad(lambda t: t ** 2, 1e6) == pytest.approx(2e6, rel=1e-6)   # relative step


def test_grad_vectorized_matches_loop():
    f_batch = lambda X: (X ** 2).sum(axis=1)
    f_single = lambda x: float((x ** 2).sum())
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(grad(f_batch, x, vectorized=True), grad(f_single, x))


def test_grad_matrix_input_shape():
    X = np.arange(6.0).reshape(2, 3)
    g = grad(lambda A: float((A ** 2).sum()), X)
    assert g.shape == (2, 3)
    np.testing.assert_allclose(g, 2 * X, atol=1e-6)


def test_jacobian():
    F = lambda v: np.array([v[0] * v[1], np.sin(v[0]), v[1] ** 2])
    x = np.array([1.0, 2.0])
    expected = np.array([[2.0, 1.0], [np.cos(1.0), 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(jacobian(F, x), expected, atol=1e-8)
    np.testing.assert_allclose(jacobian(F, x, method="forward"), expected, atol=1e-4)
    Fb = lambda X: np.column_stack([X[:, 0] * X[:, 1], np.sin(X[:, 0]), X[:, 1] ** 2])
    np.testing.assert_allclose(jacobian(Fb, x, vectorized=True), expected, atol=1e-8)


def test_grad_errors():
    with pytest.raises(ValueError):
        grad(quad, [1.0, 2.0], method="bad")
    with pytest.raises(ValueError):
        grad(quad, [1.0, 2.0], h=0)
    with pytest.raises(ValueError):
        grad(lambda v: v * 2, [1.0, 2.0])               # vector output
    with pytest.raises(ValueError):
        grad(quad, [])


def test_check_grad_line_search_and_descent():
    assert check_grad(quad, quad_grad, [0.3, 0.7]) < 1e-7
    f = lambda v: float(((v - 3) ** 2).sum())
    x = np.zeros(2)
    alpha = line_search(f, x, direction=np.array([6.0, 6.0]))
    assert f(x + alpha * np.array([6.0, 6.0])) < f(x)
    with pytest.raises(ValueError):
        line_search(f, x, direction=np.array([-1.0, -1.0]))  # ascent direction
    res = gradient_descent(f, [10.0, -4.0])
    np.testing.assert_allclose(res["x"], [3, 3], atol=1e-5)
    assert res["converged"]
