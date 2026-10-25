
import math


def standardize_loop(X):
    n, d = len(X), len(X[0])
    out = [[0.0] * d for _ in range(n)]
    for j in range(d):
        m = sum(X[i][j] for i in range(n)) / n
        s = math.sqrt(sum((X[i][j] - m) ** 2 for i in range(n)) / n) or 1.0
        for i in range(n):
            out[i][j] = (X[i][j] - m) / s
    return out


def pairwise_euclidean_loop(A, B):
    return [[math.sqrt(sum((a - b) ** 2 for a, b in zip(ra, rb))) for rb in B] for ra in A]


def softmax_rows_loop(X):
    out = []
    for row in X:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        out.append([v / s for v in e])
    return out


def welford_loop(x):
    n, mean, m2 = 0, 0.0, 0.0
    for v in x:
        n += 1
        d = v - mean
        mean += d / n
        m2 += d * (v - mean)
    return mean, m2 / n


def histogram_loop(x, bins):
    lo, hi = min(x), max(x)
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in x:
        i = int((v - lo) / width)
        counts[min(i, bins - 1)] += 1
    return counts


def rank_average_loop(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    return ranks


def topk_loop(x, k):
    return sorted(x, reverse=True)[:k]


def confusion_matrix_loop(y_true, y_pred, n_classes):
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def one_hot_loop(col, categories):
    index = {c: i for i, c in enumerate(categories)}
    out = []
    for v in col:
        row = [0.0] * len(categories)
        row[index[v]] = 1.0
        out.append(row)
    return out


def percentile_loop(x, q):
    s = sorted(x)
    pos = q / 100 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
