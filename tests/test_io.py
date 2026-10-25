import numpy as np
import pytest

from numcompute.io import count_missing, iter_csv_chunks, read_csv, read_csv_columns


def test_read_csv_numeric_fills_nan(csv_file):
    X, cols = read_csv(csv_file, usecols=["a", "b", "label"])
    assert cols == ["a", "b", "label"]
    assert X.shape == (4, 3)
    assert np.isnan(X[1, 1]) and np.isnan(X[2, 0])
    assert X[3, 2] == 0


def test_read_csv_custom_fill_and_drop(csv_file):
    X, _ = read_csv(csv_file, usecols=[0, 1], fill_value=-1)
    assert X[1, 1] == -1 and X[2, 0] == -1
    Xd, _ = read_csv(csv_file, usecols=[0, 1], drop_missing=True)
    np.testing.assert_array_equal(Xd, [[1, 2.5], [7, 8.5]])


def test_read_csv_string_and_tab_delimiter(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("name\tscore\nann\t3\nbob\t\n")
    S, cols = read_csv(p, delimiter="\t", dtype=str, fill_value="?")
    assert cols == ["name", "score"]
    assert S.tolist() == [["ann", "3"], ["bob", "?"]]


def test_read_csv_single_row_is_2d(tmp_path):
    p = tmp_path / "one.csv"
    p.write_text("x,y\n1,2\n")
    X, _ = read_csv(p)
    assert X.shape == (1, 2)


def test_read_csv_errors(tmp_path, csv_file):
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "missing.csv")
    with pytest.raises(KeyError):
        read_csv(csv_file, usecols=["nope"])
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n")
    with pytest.raises(ValueError):
        read_csv(empty)


def test_read_csv_columns_type_inference(csv_file):
    cols = read_csv_columns(csv_file)
    assert cols["a"].dtype == float and np.isnan(cols["a"][2])
    assert cols["c"].dtype.kind == "U" and cols["c"][3] == ""


def test_iter_csv_chunks_matches_full_read(tmp_path, rng):
    data = rng.normal(size=(103, 3))
    p = tmp_path / "big.csv"
    np.savetxt(p, data, delimiter=",", header="a,b,c", comments="")
    chunks = list(iter_csv_chunks(p, chunksize=25))
    assert [c.shape[0] for c in chunks] == [25, 25, 25, 25, 3]
    np.testing.assert_allclose(np.vstack(chunks), data)
    with pytest.raises(ValueError):
        next(iter_csv_chunks(p, chunksize=0))


def test_count_missing():
    assert count_missing(np.array([[1, np.nan], [np.nan, np.nan]])).tolist() == [1, 2]
    assert count_missing(np.array([["a", ""], ["", " "]])).tolist() == [1, 2]
