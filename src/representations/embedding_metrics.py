"""Metrics for judging a molecular embedding and how far a validation set sits from training.

Two groups, all computed from feature matrices alone (plus labels where noted):

  Quality of the embedding
    probe_r2        ridge regression on the embedding, validation R2 (labels)
    knn_r2          5-NN regression on the embedding, validation R2 (labels)
    recall@k        overlap of embedding k-NN with Morgan-Tanimoto k-NN
    eff_rank        exp(entropy) of the singular-value spectrum; low means collapse
    roughness       1-NN label difference over the mean random-pair difference (labels);
                    below 1 means neighbours share labels, lower is smoother

  Distance between train and validation
    nn_ratio        median val->train NN distance over median train leave-one-out NN distance
    frac_out        share of val molecules farther from train than the train 95th percentile
                    (0.05 when val looks like train)
    two_sample_auc  cross-validated AUROC of a classifier separating train from val
                    (0.5 same distribution, 1.0 fully separable)
    mmd_p           permutation p-value of the RBF MMD between train and val
                    (small means the two distributions differ)
    err_dist_rho    Spearman of val NN distance vs probe absolute error (labels);
                    positive means distance in this space predicts failure

The Morgan-Tanimoto versions of nn_ratio and frac_out are returned by `fingerprint_shift`
as the baseline for "the split is hard" against "this embedding sees the split as hard".
"""

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import pairwise_distances, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

_RIDGE_ALPHAS = np.logspace(-1, 4, 6)
_KNN_K = 5
_OUT_PERCENTILE = 95.0
_PCA_DIMS = 50
_MMD_PERMUTATIONS = 200
_MMD_MAX = 1000
_TANIMOTO_CHUNK = 2048


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both matrices with the train mean and std, dropping constant train columns."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    keep = std > 1e-8
    return (train[:, keep] - mean[keep]) / std[keep], (val[:, keep] - mean[keep]) / std[keep]


def subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted random row indices, all of range(n) when n <= cap."""
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def effective_rank(X: np.ndarray) -> float:
    """Roy & Vetterli effective rank of the centred matrix; also what RankMe reports."""
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def probe_r2(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, np.ndarray]:
    """Validation R2 of a ridge probe, and its predictions."""
    pred = RidgeCV(alphas=_RIDGE_ALPHAS).fit(X_train, y_train).predict(X_val)
    return float(r2_score(y_val, pred)), pred


def knn_r2(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Validation R2 of k-NN regression in the embedding space."""
    k = min(_KNN_K, len(X_train))
    pred = KNeighborsRegressor(n_neighbors=k, algorithm="brute").fit(X_train, y_train).predict(X_val)
    return float(r2_score(y_val, pred))


def _mean_pair_difference(y: np.ndarray) -> float:
    """Exact mean |yi - yj| over all pairs i < j."""
    n = len(y)
    ys = np.sort(y)
    weights = 2 * np.arange(n) - n + 1
    return float((weights * ys).sum() / (n * (n - 1) / 2))


def roughness(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Mean 1-NN label difference (leave-one-out) over the mean random-pair label difference."""
    if len(y_train) < 3:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(X_train)
    # column 0 is the point itself unless an exact duplicate ties it, so the other column
    # is always a neighbour; picking by index keeps the duplicate case correct
    idx = nn.kneighbors(X_train, return_distance=False)
    own = np.arange(len(X_train))
    other = np.where(idx[:, 0] == own, idx[:, 1], idx[:, 0])
    baseline = _mean_pair_difference(y_train)
    if baseline == 0:
        return float("nan")
    return float(np.abs(y_train - y_train[other]).mean() / baseline)


def nn_distances(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(val->train NN distance, train leave-one-out NN distance), Euclidean."""
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(train)
    d_val = nn.kneighbors(val, n_neighbors=1)[0][:, 0]
    # second column: the nearest other point, 0 for an exact duplicate
    d_train = nn.kneighbors(train)[0][:, 1]
    return d_val, d_train


def shift_stats(d_val: np.ndarray, d_train: np.ndarray) -> dict[str, float]:
    """How much farther val sits from train than train sits from itself."""
    ref = float(np.median(d_train))
    cutoff = np.percentile(d_train, _OUT_PERCENTILE)
    return {
        "nn_ratio": float(np.median(d_val) / ref) if ref > 0 else float("nan"),
        "frac_out": float((d_val > cutoff).mean()),
    }


def _tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def tanimoto_nn_distances(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(val->train, train leave-one-out) NN Tanimoto distance (1 - similarity) on bit matrices."""
    tr = train.astype(np.float32)
    va = val.astype(np.float32)
    d_val = np.empty(len(va), dtype=np.float32)
    for start in range(0, len(va), _TANIMOTO_CHUNK):
        d_val[start : start + _TANIMOTO_CHUNK] = 1.0 - _tanimoto(va[start : start + _TANIMOTO_CHUNK], tr).max(axis=1)
    d_train = np.empty(len(tr), dtype=np.float32)
    for start in range(0, len(tr), _TANIMOTO_CHUNK):
        sim = _tanimoto(tr[start : start + _TANIMOTO_CHUNK], tr)
        rows = np.arange(len(sim))
        sim[rows, start + rows] = -1.0
        d_train[start : start + len(sim)] = 1.0 - sim.max(axis=1)
    return d_val, d_train


def fingerprint_shift(train_fp: np.ndarray, val_fp: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Morgan-Tanimoto baseline: (nn_ratio and frac_out, per-val-molecule NN distance)."""
    d_val, d_train = tanimoto_nn_distances(train_fp, val_fp)
    return shift_stats(d_val, d_train), d_val


def neighbour_recall(Z: np.ndarray, fp: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each molecule's k nearest Tanimoto neighbours also among its k embedding neighbours."""
    n = len(Z)
    k = min(k, n - 1)
    if k < 1:
        return float("nan")
    emb = pairwise_distances(Z)
    np.fill_diagonal(emb, np.inf)
    fpf = fp.astype(np.float32)
    sim = _tanimoto(fpf, fpf)
    np.fill_diagonal(sim, -np.inf)
    emb_nn = np.argpartition(emb, k - 1, axis=1)[:, :k]
    fp_nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    hits = [len(set(a) & set(b)) / k for a, b in zip(emb_nn, fp_nn)]
    return float(np.mean(hits))


def two_sample_auc(train: np.ndarray, val: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUROC of a logistic classifier telling train from val, on PCA-reduced features."""
    X = np.vstack([train, val])
    y = np.r_[np.zeros(len(train)), np.ones(len(val))]
    dims = min(_PCA_DIMS, X.shape[1], len(X) - 1)
    X = PCA(n_components=dims, random_state=seed).fit_transform(X)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    proba = cross_val_predict(clf, X, y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def mmd_rbf(train: np.ndarray, val: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """(unbiased MMD^2, permutation p-value) with an RBF kernel at the median-heuristic bandwidth."""
    rng = np.random.default_rng(seed)
    a = train[subsample(len(train), _MMD_MAX, rng)]
    b = val[subsample(len(val), _MMD_MAX, rng)]
    pooled = np.vstack([a, b])
    sq = pairwise_distances(pooled, metric="sqeuclidean")
    off_diagonal = sq[np.triu_indices_from(sq, k=1)]
    bandwidth = float(np.median(off_diagonal))
    if bandwidth <= 0:
        return 0.0, 1.0
    K = np.exp(-sq / bandwidth)
    m = len(a)
    n = len(b)

    def statistic(order: np.ndarray) -> float:
        i, j = order[:m], order[m:]
        kxx = K[np.ix_(i, i)]
        kyy = K[np.ix_(j, j)]
        kxy = K[np.ix_(i, j)]
        return float(
            (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
            + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
            - 2 * kxy.mean()
        )

    observed = statistic(np.arange(m + n))
    exceed = sum(statistic(rng.permutation(m + n)) >= observed for _ in range(_MMD_PERMUTATIONS))
    return observed, float((1 + exceed) / (1 + _MMD_PERMUTATIONS))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho, NaN when either side is constant or too short."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def shift_metrics(
    Z_train: np.ndarray,
    Z_other: np.ndarray,
    *,
    max_samples: int = 3000,
    seed: int = 0,
) -> tuple[dict[str, float], np.ndarray]:
    """Label-free distance metrics of one set against train, and that set's per-molecule NN distance.

    Z_* are already standardized with the train statistics. Works for any held-out set,
    including the blinded test molecules, since it never reads a label.
    """
    rng = np.random.default_rng(seed)
    d_other, d_train = nn_distances(Z_train, Z_other)
    out = shift_stats(d_other, d_train)
    tr_rows = subsample(len(Z_train), max_samples, rng)
    other_rows = subsample(len(Z_other), max_samples, rng)
    out["two_sample_auc"] = two_sample_auc(Z_train[tr_rows], Z_other[other_rows], seed)
    out["mmd"], out["mmd_p"] = mmd_rbf(Z_train, Z_other, seed)
    return out, d_other


def evaluate_shift(
    X_train: np.ndarray,
    X_other: np.ndarray,
    *,
    max_samples: int = 3000,
    seed: int = 0,
) -> dict[str, float]:
    """shift_metrics on raw feature matrices, standardizing with the train statistics."""
    Z_train, Z_other = standardize(X_train, X_other)
    return shift_metrics(Z_train, Z_other, max_samples=max_samples, seed=seed)[0]


def evaluate_embedding(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: dict[str, np.ndarray],
    y_val: dict[str, np.ndarray],
    fp_train: np.ndarray,
    fp_val: np.ndarray,
    *,
    max_samples: int = 3000,
    recall_k: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Every metric for one embedding.

    X_*/fp_* hold the same molecules in the same order, finite everywhere. y_* map a target
    name to a per-molecule array that may contain NaN for molecules not assayed on it;
    label metrics are computed per target and averaged over the targets that have data.
    """
    rng = np.random.default_rng(seed)
    Z_train, Z_val = standardize(X_train, X_val)

    out: dict[str, float] = {"dim": float(X_train.shape[1])}
    shift, d_val = shift_metrics(Z_train, Z_val, max_samples=max_samples, seed=seed)
    out.update(shift)

    pool = np.vstack([Z_train, Z_val])
    rank_rows = subsample(len(pool), max_samples, rng)
    out["eff_rank"] = effective_rank(pool[rank_rows])

    tr_rows = subsample(len(Z_train), max_samples, rng)
    out[f"recall@{recall_k}"] = neighbour_recall(Z_train[tr_rows], fp_train[tr_rows], recall_k)

    per_target: dict[str, list[float]] = {"probe_r2": [], "knn_r2": [], "roughness": [], "err_dist_rho": []}
    for target, y_tr_all in y_train.items():
        tr = np.isfinite(y_tr_all)
        va = np.isfinite(y_val[target])
        if tr.sum() < 10 or va.sum() < 10:
            continue
        r2, pred = probe_r2(Z_train[tr], y_tr_all[tr], Z_val[va], y_val[target][va])
        per_target["probe_r2"].append(r2)
        per_target["knn_r2"].append(knn_r2(Z_train[tr], y_tr_all[tr], Z_val[va], y_val[target][va]))
        per_target["roughness"].append(roughness(Z_train[tr], y_tr_all[tr]))
        per_target["err_dist_rho"].append(spearman(d_val[va], np.abs(y_val[target][va] - pred)))
    for name, values in per_target.items():
        finite = [v for v in values if np.isfinite(v)]
        out[name] = float(np.mean(finite)) if finite else float("nan")
    return out
