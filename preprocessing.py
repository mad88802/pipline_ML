"""Step 02 — generic data preprocessing (works on any dataset).

Pipeline, in this order (every step can be switched off or tuned):
  dates -> manual column drop -> duplicates -> useless columns (id / constant / too many missing)
  -> missing target rows -> missing values (KNN imputation) -> outliers (One-Class SVM, flagged train rows removed) -> noise (moving average, time series)
  -> encoding (target encoding) -> class balancing (SMOTE)
SPLIT first: the chronological train/test split is made right after dates / duplicates / missing target, BEFORE every learned
step (imputation, outliers, encoding, SMOTE): they are fitted on train and only applied to test. Result column: __split.

get_profile(dataset)         -> what the page shows before running (per-column profile + recommended config)
process(dataset, cfg, save)  -> runs the pipeline, saves the cleaned data + the report in results/analysis.db (SQLite)
load_report(dataset)         -> the saved report (dict) | load_clean(dataset) -> the saved DataFrame
"""
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import analysis_engine as ae
import step_cache

PREVIEW_ROWS = 8
SPLIT_COL = "__split"            # column added to the result : "train" / "test"

DEFAULTS = {
    "split_strategy": "auto",        # auto | random | stratified | chronological (chosen in the Split step)
    "test_size": 0.2,                # part of the rows kept for the test (split BEFORE every learned step) ; 0 = no split
    "drop_duplicates": True,
    "drop_useless": True,            # id-like columns, constant columns, columns with too many missing values
    "missing_col_threshold": 60,     # % of missing above which a column is dropped
    "num_missing": "knn",            # knn | mean | constant (0) | interpolate | drop_rows | none
    "knn_k": 5,                      # neighbours used by the KNN imputation
    "cat_missing": "knn",            # knn (vote of the k nearest rows) | mode | constant ("Inconnu") | drop_rows | none
    "outliers": "ocsvm",             # none | ocsvm (One-Class SVM: the flagged TRAIN rows are removed, test rows are never touched)
    "ocsvm_nu": 0.05,                # One-Class SVM nu = upper bound of the share of train rows flagged as outliers
    "smooth_window": 0,              # rolling mean window (time series only), 0 = off
    "encoding": "target",            # target | onehot | label | frequency | none
    "te_smoothing": 10,              # target encoding: weight of the global mean (higher = more shrinkage)
    "max_onehot": 10,                # (one-hot only) columns with more classes than this are label-encoded instead
    "scaling": "none",               # none | standard | minmax | robust (fitted on train, numeric non-binary columns)
    "balance": False,                # SMOTE oversampling of the minority classes (classification target)
    "smote_k": 5,                    # neighbours used by SMOTE
    "encode_target": True,           # label-encode a text target
    "target": None,
    "drop_columns": [],
}
CHOICES = {
    "num_missing": {"knn", "mean", "constant", "interpolate", "drop_rows", "none"},
    "cat_missing": {"knn", "mode", "constant", "drop_rows", "none"},
    "outliers": {"none", "ocsvm"},
    "encoding": {"target", "onehot", "label", "frequency", "none"},
    "scaling": {"none", "standard", "minmax", "robust"},
    "split_strategy": {"auto", "random", "stratified", "chronological"},
}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def clean_config(raw):
    """Merge a (possibly partial / untrusted) config into the defaults and clamp every value."""
    raw = raw if isinstance(raw, dict) else {}
    cfg = dict(DEFAULTS)
    for k in DEFAULTS:
        if k in raw:
            cfg[k] = raw[k]
    for k, ok in CHOICES.items():
        if cfg[k] not in ok:
            cfg[k] = DEFAULTS[k]
    for k in ("drop_duplicates", "drop_useless", "encode_target", "balance"):
        cfg[k] = bool(cfg[k])
    cfg["test_size"] = min(0.5, max(0.0, _num(cfg["test_size"], 0.2)))
    cfg["missing_col_threshold"] = min(100.0, max(1.0, _num(cfg["missing_col_threshold"], 60)))
    cfg["ocsvm_nu"] = float(min(0.5, max(0.005, _num(cfg["ocsvm_nu"], 0.05))))
    cfg["smooth_window"] = int(min(365, max(0, _num(cfg["smooth_window"], 0))))
    cfg["max_onehot"] = int(min(100, max(2, _num(cfg["max_onehot"], 10))))
    cfg["smote_k"] = int(min(15, max(1, _num(cfg["smote_k"], 5))))
    cfg["knn_k"] = int(min(30, max(1, _num(cfg["knn_k"], 5))))
    cfg["te_smoothing"] = float(min(1000.0, max(0.0, _num(cfg["te_smoothing"], 10))))
    cols = cfg["drop_columns"] if isinstance(cfg["drop_columns"], (list, tuple)) else []
    cfg["drop_columns"] = [str(c) for c in cols]
    cfg["target"] = str(cfg["target"]) if cfg["target"] not in (None, "") else None
    return cfg


def _knn_impute(tr, te, cols, k=5, max_fit=10000, chunk=5000):
    """KNN imputation of the numeric columns: each missing value = mean of that column over the k nearest rows
    (distance on standardised columns, computed on the columns both rows have). Fitted on TRAIN only, then applied to train and test.
    Anything KNN cannot fill (all-empty columns, rows without any observed value) gets the train mean."""
    from sklearn.impute import KNNImputer
    tr, te = tr.copy(), te.copy()
    avg = tr[cols].mean()
    use = [c for c in cols if tr[c].notna().any()]
    if use:
        mu, sd = tr[use].mean(), tr[use].std().replace(0, 1).fillna(1)
        fit_df = (tr[use] - mu) / sd
        if len(fit_df) > max_fit:                                   # keeps the cost bounded on big datasets
            fit_df = fit_df.sample(max_fit, random_state=42)
        imp = KNNImputer(n_neighbors=int(max(1, min(k, len(fit_df)))), weights="distance").fit(fit_df.to_numpy())
        for part in (tr, te):
            miss = part[use].isna().any(axis=1).to_numpy()
            if not miss.any():
                continue
            idx = np.flatnonzero(miss)
            for i in range(0, len(idx), chunk):
                sel = idx[i:i + chunk]
                z = ((part.iloc[sel][use] - mu) / sd).to_numpy()
                part.iloc[sel, [part.columns.get_loc(c) for c in use]] = imp.transform(z) * sd.to_numpy() + mu.to_numpy()
    tr[cols] = tr[cols].fillna(avg).fillna(0)
    te[cols] = te[cols].fillna(avg).fillna(0)
    return tr, te


def _knn_impute_cat(tr, te, cat_cols, num_cols, k=5, max_fit=10000):
    """KNN imputation of text / categorical columns: a missing category = majority vote of the k nearest rows that have it
    (distance on the standardised, already imputed numeric columns). Fitted on TRAIN only. Without usable numeric columns,
    or for a column KNN cannot predict, the most frequent train value is used."""
    from sklearn.neighbors import KNeighborsClassifier
    tr, te = tr.copy(), te.copy()
    feats = [c for c in num_cols if c in tr.columns and tr[c].notna().all() and tr[c].nunique() > 1]
    if feats:
        mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1).fillna(1)
    for c in cat_cols:
        if not (tr[c].isna().any() or te[c].isna().any()):
            continue
        known = tr[c].notna()
        if feats and known.sum() >= 2:
            fit = tr.loc[known, feats]
            if len(fit) > max_fit:
                fit = fit.sample(max_fit, random_state=42)
            y = tr.loc[fit.index, c].astype(str)
            clf = KNeighborsClassifier(n_neighbors=int(max(1, min(k, len(fit)))), weights="distance").fit(((fit - mu) / sd).to_numpy(), y)
            for part in (tr, te):
                m = part[c].isna()
                if m.any() and all(f in part.columns and part[f].notna().all() for f in feats):
                    part[c] = part[c].astype(object)
                    part.loc[m, c] = clf.predict(((part.loc[m, feats] - mu) / sd).to_numpy())
        mode = tr[c].mode(dropna=True)
        if len(mode):
            tr[c] = tr[c].fillna(mode.iloc[0])
            te[c] = te[c].fillna(mode.iloc[0])
    return tr, te


def _if_features(df, exclude=()):
    """Numeric columns the Isolation Forest looks at (same rule as the EDA page, target/date excluded)."""
    feats = []
    for c in df.columns:
        if c in exclude or not _is_num(df[c]) or df[c].nunique(dropna=True) <= 1:
            continue
        try:
            if ae.is_id_like(df, c):
                continue
        except Exception:
            pass
        feats.append(c)
    return feats[:ae.IF_MAX_FEATURES]


SVM_MAX_FIT_ROWS = 4000       # One-Class SVM is O(n^2): it is fitted on a sample above this size, every row is scored


def _svm_features(df, exclude=()):
    """Numeric columns the One-Class SVM looks at: continuous ones only (binary / 0-1 flags would make their rare value look anomalous)."""
    return [c for c in _if_features(df, exclude) if df[c].nunique(dropna=True) > 2]


def _svm_matrix(d, cols, mu=None, sd=None):
    raw = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if mu is None:
        mu = raw.mean()
        sd = raw.std().replace(0, 1).fillna(1)
    return ((raw - mu) / sd).fillna(0.0), mu, sd


def _svm_fit(df, cols, nu=0.05):
    """One-Class SVM (RBF kernel, gamma='scale') fitted on `df` (the TRAIN rows) on standardised numeric columns.
    Returns a fit, or None if there is nothing to analyse."""
    from sklearn.svm import OneClassSVM
    cols = [c for c in cols if c in df.columns]
    if not cols or len(df) < 20:
        return None
    X, mu, sd = _svm_matrix(df, cols)
    X = X.loc[:, X.std() > 0]
    cols = list(X.columns)
    if not cols:
        return None
    fit_X = X if len(X) <= SVM_MAX_FIT_ROWS else X.sample(SVM_MAX_FIT_ROWS, random_state=42)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=float(nu)).fit(fit_X.to_numpy())
    return model, cols, mu[cols], sd[cols]


def _svm_predict(fit, d, chunk=20000):
    """Boolean Series (True = outlier row) for the rows of `d`, using a model fitted elsewhere."""
    model, cols, mu, sd = fit
    if len(d) == 0:
        return pd.Series([], index=d.index, dtype=bool)
    X, _, _ = _svm_matrix(d, cols, mu, sd)
    X = X.to_numpy()
    pred = np.concatenate([model.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])
    return pd.Series(pred == -1, index=d.index)


def _svm_flag(df, cols, nu=0.05, target=None):
    """Fit AND predict on the same rows (profile page and step views)."""
    fit = _svm_fit(df, cols, nu)
    return None if fit is None else _protect_classes(_svm_predict(fit, df), df[target] if target and target in df.columns else None)


def _svm_view(df, cols, nu=0.05, target=None, max_pts=1500, grid_n=60):
    """One-Class SVM diagnostics for the outliers step: fit on `df` (TRAIN), then return the row flags and a
    payload made for this method (decision score, 2-D PCA slice with the decision boundary, per-variable profile of
    the flagged rows). Returns (flag, view) or (None, None)."""
    fit = _svm_fit(df, cols, nu)
    if fit is None:
        return None, None
    model, fcols, mu, sd = fit
    X, _, _ = _svm_matrix(df, fcols, mu, sd)
    Xn = X.to_numpy()

    def dec(M, chunk=20000):
        return np.concatenate([model.decision_function(M[i:i + chunk]) for i in range(0, len(M), chunk)]) if len(M) else np.array([])

    score = dec(Xn)
    raw = pd.Series(score <= 0, index=df.index)                      # predict == -1  <=>  decision_function <= 0
    y = df[target] if target and target in df.columns else None
    flag = _protect_classes(raw, y)
    fl = flag.to_numpy()
    view = {"n_protected": int(raw.sum() - flag.sum()), "features": [str(c) for c in fcols]}
    rng = np.random.default_rng(42)

    # score distribution (negative = outside the learned region)
    view["score_kept"] = _sample(score[~fl], 3000)
    view["score_flagged"] = _sample(score[fl], 3000)

    # 2-D PCA slice of the model: scatter + decision boundary
    view["pca"] = None
    if Xn.shape[1] >= 2 and len(Xn) >= 20:
        mean = Xn.mean(axis=0)
        Xc = Xn - mean
        fit_rows = Xc if len(Xc) <= 5000 else Xc[rng.choice(len(Xc), 5000, replace=False)]
        _, S, Vt = np.linalg.svd(fit_rows, full_matrices=False)
        V = Vt[:2]
        P = Xc @ V.T
        var = (S ** 2) / max(float((S ** 2).sum()), 1e-12)
        idx_f, idx_k = np.where(fl)[0], np.where(~fl)[0]
        n_f = min(len(idx_f), max_pts // 2)
        n_k = min(len(idx_k), max_pts - n_f)
        sf = rng.choice(idx_f, n_f, replace=False) if n_f else idx_f[:0]
        sk = rng.choice(idx_k, n_k, replace=False) if n_k else idx_k[:0]
        lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
        pad = (hi - lo) * 0.15
        gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid_n)
        gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid_n)
        GX, GY = np.meshgrid(gx, gy)
        G = np.c_[GX.ravel(), GY.ravel()] @ V + mean          # back to the standardised space (other axes at 0)
        Z = dec(G).reshape(grid_n, grid_n)
        view["pca"] = {
            "kept": {"x": [_r(v, 3) for v in P[sk, 0]], "y": [_r(v, 3) for v in P[sk, 1]]},
            "flagged": {"x": [_r(v, 3) for v in P[sf, 0]], "y": [_r(v, 3) for v in P[sf, 1]]},
            "grid": {"x": [_r(v, 3) for v in gx], "y": [_r(v, 3) for v in gy], "z": [[_r(v, 3) for v in row] for row in Z]},
            "var": [_r(100 * var[0], 1), _r(100 * var[1], 1)],
            "n_shown": [int(len(sk)), int(len(sf))],
        }

    # per-variable profile: which variables make the flagged rows suspicious (mean |z|, flagged vs kept)
    prof = []
    if fl.any() and (~fl).any():
        for j_, c in enumerate(fcols):
            zf, zk = float(np.abs(Xn[fl, j_]).mean()), float(np.abs(Xn[~fl, j_]).mean())
            prof.append({"name": str(c), "z_flagged": _r(zf, 2), "z_kept": _r(zk, 2), "ratio": _r(zf / zk, 2) if zk > 1e-9 else None,
                         "_raw": c})
        prof.sort(key=lambda r: -(r["z_flagged"] or 0))
        prof = prof[:15]
        for r in prof:
            c = r.pop("_raw")
            r["kept"] = _sample(df.loc[~flag, c], 1000)
            r["flagged"] = _sample(df.loc[flag, c], 500)
    view["profile"] = prof
    return flag, view


def _protect_classes(flag, y=None, max_share=0.5):
    """Never let the outlier removal wipe out a class: when more than `max_share` of a class would go, its rows are kept."""
    if y is None or not _is_classification(y):
        return flag
    flag = flag.copy()
    for cls, grp in y.groupby(y, dropna=True):
        idx = grp.index.intersection(flag.index)
        if len(idx) and flag.loc[idx].mean() > max_share:
            flag.loc[idx] = False
    return flag


def _label_encode(s):
    cats = sorted(s.dropna().astype(str).unique())
    mapping = {v: i for i, v in enumerate(cats)}
    out = s.astype(str).where(s.notna()).map(mapping)
    return (out.astype("int64") if not out.isna().any() else out), mapping


def _is_classification(s):
    """A column that can act as a class label: 2..20 distinct values, text or integer-valued."""
    n = int(s.nunique(dropna=True))
    if not 2 <= n <= 20:
        return False
    if not _is_num(s):
        return True
    v = s.dropna().to_numpy(dtype=float)
    return bool(len(v) and np.all(np.mod(v, 1) == 0))


def _class_counts(s):
    vc = s.value_counts(dropna=True)
    return {str(k): int(v) for k, v in vc.items()}


def _target_encode(df, cats, target, smoothing=10.0, folds=5, seed=42):
    """Smoothed, out-of-fold target encoding (K-fold, so a row never sees its own label).
    Regression / binary target -> the column is replaced in place; multi-class -> one column per class (minus one)."""
    y = df[target]
    if _is_num(y) and not _is_classification(y):
        Y = pd.DataFrame({"": y.astype(float)})
    else:
        classes = sorted(y.dropna().astype(str).unique())
        keep = classes[-1:] if len(classes) == 2 else classes[:-1]
        ys = y.astype(str)
        Y = pd.DataFrame({c: (ys == c).astype(float) for c in keep})
    n = len(df)
    kf = list(KFold(n_splits=min(folds, max(2, n // 2)), shuffle=True, random_state=seed).split(np.arange(n))) if n >= 4 else None
    out = {}
    for c in cats:
        s = df[c].astype(object).where(df[c].notna(), "__NA__").astype(str).reset_index(drop=True)
        for yc in Y.columns:
            yv = Y[yc].reset_index(drop=True)
            prior = float(yv.mean())

            def mapping(idx):
                g = yv.iloc[idx].groupby(s.iloc[idx]).agg(["sum", "count"])
                return (g["sum"] + smoothing * prior) / (g["count"] + smoothing)

            enc = pd.Series(prior, index=range(n), dtype=float)
            if kf:
                for tr, va in kf:
                    m = mapping(tr)
                    enc.iloc[va] = s.iloc[va].map(m).fillna(float(yv.iloc[tr].mean())).to_numpy()
            else:
                enc[:] = s.map(mapping(np.arange(n))).fillna(prior).to_numpy()
            out[c if yc == "" or len(Y.columns) == 1 else f"{c}_te_{yc}"] = enc.to_numpy()
    new = df.drop(columns=cats)
    for name, vals in out.items():
        new[name] = vals
    return new, len(out)


def _smote(df, target, k=5, seed=42):
    """SMOTE: for each minority class, synthesise samples by interpolating between a point and one of its
    k nearest neighbours *of the same class* until it matches the majority class. Returns (df, info)."""
    y = df[target]
    counts = y.value_counts()
    info = {"added": 0, "skipped": None}
    if len(counts) < 2:
        info["skipped"] = "une seule classe"
        return df, info
    num_cols = [c for c in df.columns if c != target and _is_num(df[c])]
    if not num_cols:
        info["skipped"] = "aucune colonne numérique"
        return df, info
    X = df[num_cols].to_numpy(dtype=float)
    if np.isnan(X).any():
        info["skipped"] = "valeurs manquantes restantes (activez l'imputation)"
        return df, info
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(seed)
    majority, parts = int(counts.max()), []
    yv = y.to_numpy()
    for cls, n in counts.items():
        need = majority - int(n)
        idx = np.where(yv == cls)[0]
        if need <= 0:
            continue
        if len(idx) < 2:
            info["skipped"] = f"classe « {cls} » trop petite (1 ligne)"
            continue
        kk = min(int(k), len(idx) - 1)
        neigh = NearestNeighbors(n_neighbors=kk + 1).fit(Xs[idx]).kneighbors(Xs[idx], return_distance=False)[:, 1:]
        base = rng.integers(0, len(idx), need)
        nb = neigh[base, rng.integers(0, kk, need)]
        lam = rng.random((need, 1))
        syn = X[idx][base] + lam * (X[idx][nb] - X[idx][base])
        part = df.iloc[idx[base]].reset_index(drop=True)   # copies target + any non-numeric column from the base sample
        for j, c in enumerate(num_cols):
            col = syn[:, j]
            if np.all(np.mod(df[c].to_numpy(dtype=float), 1) == 0):
                col = np.round(col)                         # integer / 0-1 columns stay integer
            part[c] = col.astype(df[c].dtype) if pd.api.types.is_integer_dtype(df[c]) else col
        parts.append(part)
    if parts:
        df = pd.concat([df] + parts, ignore_index=True)
        info["added"] = int(sum(len(p) for p in parts))
    return df, info


def _state(df):
    return len(df), df.shape[1] - (SPLIT_COL in df.columns), int(df.isna().sum().sum())


def _names(cols, n=5):
    cols = [str(c) for c in cols]
    return ", ".join(cols[:n]) + (f" +{len(cols) - n}" if len(cols) > n else "")


def _preview(df, n=PREVIEW_ROWS):
    head = df.head(n)
    data = json.loads(head.to_json(orient="split", date_format="iso", double_precision=4))
    return {"columns": data["columns"], "rows": data["data"]}


# ---------------------------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------------------------
def _split_mask(n, y, is_ts, test_size, seed=42, strategy="auto"):
    """Boolean test mask + label.
    strategy : auto (time series -> chronological, classification -> stratified, else random) | random | stratified | chronological.
    Chronological = the LAST rows are the test (file / date order). Stratified falls back to random if y is not a classification target."""
    n_test = min(max(1, int(round(n * test_size))), n - 1)
    test = np.zeros(n, dtype=bool)
    if strategy == "chronological" or (strategy == "auto" and is_ts):
        test[-n_test:] = True
        return test, "chronologique"
    rng = np.random.default_rng(seed)
    if strategy in ("auto", "stratified") and y is not None and _is_classification(y):
        yv = y.to_numpy()
        for cls in pd.unique(yv):
            idx = np.flatnonzero(yv == cls)
            k = int(round(len(idx) * test_size))
            if len(idx) > 1:
                k = min(max(k, 1), len(idx) - 1)
            test[rng.choice(idx, size=k, replace=False)] = True
        return test, "stratifié"
    test[rng.choice(n, size=n_test, replace=False)] = True
    return test, "aléatoire"


def _te_apply(tr, te, cats, target, smoothing=10.0):
    """Target encoding of the TEST rows from the mapping learned on the whole train (same column names as _target_encode)."""
    y = tr[target]
    if _is_num(y) and not _is_classification(y):
        Y = pd.DataFrame({"": y.astype(float)})
    else:
        classes = sorted(y.dropna().astype(str).unique())
        keep = classes[-1:] if len(classes) == 2 else classes[:-1]
        ys = y.astype(str)
        Y = pd.DataFrame({c: (ys == c).astype(float) for c in keep})
    out = {}
    for c in cats:
        s = tr[c].astype(object).where(tr[c].notna(), "__NA__").astype(str).reset_index(drop=True)
        st = te[c].astype(object).where(te[c].notna(), "__NA__").astype(str).reset_index(drop=True)
        for yc in Y.columns:
            yv = Y[yc].reset_index(drop=True)
            prior = float(yv.mean())
            g = yv.groupby(s).agg(["sum", "count"])
            m = (g["sum"] + smoothing * prior) / (g["count"] + smoothing)
            out[c if yc == "" or len(Y.columns) == 1 else f"{c}_te_{yc}"] = st.map(m).fillna(prior).to_numpy()
    new = te.drop(columns=cats).reset_index(drop=True)
    for name, vals in out.items():
        new[name] = vals
    return new


def _label_fit_apply(tr, te, c):
    """Label-encode c with the mapping learned on train; categories unseen in train -> -1 in test."""
    tr[c], mapping = _label_encode(tr[c])
    te[c] = te[c].astype(str).where(te[c].notna()).map(mapping).fillna(-1).astype("int64")
    return mapping


def run(df, cfg, dataset_type="tabular", date_col=None):
    """Apply the pipeline to a DataFrame. Returns (clean_df, steps, extra).

    The chronological split (time series; stratified / random otherwise) happens BEFORE every learned step:
    imputation values, outlier model + bounds, encodings and SMOTE are fitted on the TRAIN part only, then applied
    to the test part. The result carries a SPLIT_COL column ("train" / "test"); SMOTE rows (train) come last."""
    cfg = clean_config(cfg)
    df = df.copy()
    steps, extra = [], {"target_mapping": None}
    tr = te = None

    target = cfg["target"] if cfg["target"] in df.columns else None
    date_col = date_col if date_col in df.columns else None
    protected = {c for c in (target, date_col) if c}
    is_ts = dataset_type == "time_series" and date_col is not None
    start = _state(df)

    def whole():
        if tr is None:
            return df
        return tr if len(te) == 0 else pd.concat([tr, te], ignore_index=True)

    def log(key, label, detail=""):
        r, c, m = _state(whole())
        steps.append({"key": key, "label": label, "detail": detail, "rows": r, "cols": c, "missing": m})

    log("raw", "Données brutes")

    for c in df.columns:                                   # bool -> 0/1 so they behave like numbers
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("int8")

    # dates
    if date_col:
        df[date_col] = ae.to_datetime_safe(df[date_col])
        if is_ts:
            n = int(df[date_col].isna().sum())
            df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
            log("dates", "Dates", f"triées chronologiquement" + (f", {n} sans date supprimées" if n else ""))

    # manual column drop
    manual = [c for c in cfg["drop_columns"] if c in df.columns and c not in protected]
    if manual:
        df = df.drop(columns=manual)
        log("manual", "Colonnes retirées", _names(manual))

    # duplicates
    if cfg["drop_duplicates"]:
        n = int(df.duplicated().sum())
        if n:
            df = df.drop_duplicates().reset_index(drop=True)
        log("duplicates", "Doublons", f"{n} supprimés" if n else "aucun")

    # rows without target
    if target:
        n = int(df[target].isna().sum())
        if n:
            df = df.dropna(subset=[target]).reset_index(drop=True)
            log("target_na", "Cible manquante", f"{n} lignes supprimées")

    # ---- SPLIT : everything below learns on train only ----
    if cfg["test_size"] > 0 and len(df) >= 10:
        mask, mode = _split_mask(len(df), df[target] if target else None, is_ts, cfg["test_size"], strategy=cfg["split_strategy"])
        tr, te = df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
        log("split", f"Split {mode}",
            f"{fmt_int(len(tr))} train · {fmt_int(len(te))} test — les étapes suivantes apprennent sur le train uniquement")
    else:
        tr, te = df, df.iloc[0:0]

    # useless columns (decided on train)
    if cfg["drop_useless"]:
        ids, consts, sparse = [], [], []
        for c in tr.columns:
            if c in protected:
                continue
            if len(tr) and tr[c].isna().mean() * 100 > cfg["missing_col_threshold"]:
                sparse.append(c)
            elif tr[c].nunique(dropna=True) <= 1:
                consts.append(c)
            else:
                try:
                    if ae.is_id_like(tr, c):
                        ids.append(c)
                except Exception:
                    pass
        drop = ids + consts + sparse
        if drop:
            tr, te = tr.drop(columns=drop), te.drop(columns=drop)
        parts = [f"{len(v)} {k}" for k, v in (("identifiant", ids), ("constante", consts), ("trop vide", sparse)) if v]
        log("useless", "Colonnes inutiles", (", ".join(parts) + " : " + _names(drop)) if drop else "aucune")

    # missing values (statistics learned on train)
    num_cols = [c for c in tr.columns if c not in protected and _is_num(tr[c])]
    cat_cols = [c for c in tr.columns if c not in protected and not _is_num(tr[c])]

    def n_missing():
        return sum(int(p[[c for c in num_cols + cat_cols if c in p.columns]].isna().sum().sum()) for p in (tr, te))

    before = n_missing() if (num_cols or cat_cols) else 0
    if before:
        nm, cm = cfg["num_missing"], cfg["cat_missing"]
        if num_cols:
            if nm == "knn":
                tr, te = _knn_impute(tr, te, num_cols, cfg["knn_k"])
            elif nm == "mean":
                stat = tr[num_cols].mean()
                tr[num_cols] = tr[num_cols].fillna(stat)
                te[num_cols] = te[num_cols].fillna(stat)
            elif nm == "constant":
                tr[num_cols] = tr[num_cols].fillna(0)
                te[num_cols] = te[num_cols].fillna(0)
            elif nm == "interpolate":
                avg = tr[num_cols].mean()
                for p in (tr, te):                         # each part on its own: no value crosses the split
                    p[num_cols] = p[num_cols].interpolate(method="linear", limit_direction="both")
                    p[num_cols] = p[num_cols].fillna(avg)
            elif nm == "drop_rows":
                tr = tr.dropna(subset=num_cols).reset_index(drop=True)
                te = te.dropna(subset=num_cols).reset_index(drop=True)
        if cat_cols:
            if cm == "knn":
                tr, te = _knn_impute_cat(tr, te, cat_cols, num_cols, cfg["knn_k"])
            elif cm == "mode":
                for c in cat_cols:
                    m = tr[c].mode(dropna=True)
                    if len(m):
                        tr[c] = tr[c].fillna(m.iloc[0])
                        te[c] = te[c].fillna(m.iloc[0])
            elif cm == "constant":
                for c in cat_cols:
                    tr[c] = tr[c].astype(object).fillna("Inconnu")
                    te[c] = te[c].astype(object).fillna("Inconnu")
            elif cm == "drop_rows":
                tr = tr.dropna(subset=cat_cols).reset_index(drop=True)
                te = te.dropna(subset=cat_cols).reset_index(drop=True)
        after = n_missing()
        meth = f"imputation KNN (k={cfg['knn_k']})" if nm == "knn" else nm
        log("missing", "Valeurs manquantes", f"{before - after} cellules traitées · {meth} (valeurs apprises sur le train)" + (f", {after} restantes" if after else ""))
    elif num_cols or cat_cols:
        log("missing", "Valeurs manquantes", "aucune")

    # outliers (One-Class SVM fitted on train; flagged TRAIN rows are removed, the test rows are never touched)
    if cfg["outliers"] == "ocsvm":
        feats = _svm_features(tr, protected)
        fit = _svm_fit(tr, feats, cfg["ocsvm_nu"])
        if fit is None:
            log("outliers", "Outliers", "ignoré (pas assez de données numériques)")
        else:
            ftr = _protect_classes(_svm_predict(fit, tr), tr[target] if target and target in tr.columns else None)
            fte = _svm_predict(fit, te)
            n_tr, n_te = int(ftr.sum()), int(fte.sum())
            if n_tr:
                tr = tr[~ftr].reset_index(drop=True)
            log("outliers", "Outliers",
                f"{n_tr} lignes supprimées du train (One-Class SVM, ν={cfg['ocsvm_nu']:g}, appris sur le train)"
                + (f" · {n_te} lignes de test détectées mais conservées" if n_te else "") if n_tr or n_te else "aucun détecté")

    # noise smoothing (each part on its own, so the window never crosses the split)
    if cfg["smooth_window"] >= 2:
        if is_ts:
            cols = [c for c in num_cols if c in tr.columns and tr[c].nunique() > 2]
            for p in (tr, te):
                for c in cols:
                    p[c] = p[c].astype(float).rolling(cfg["smooth_window"], center=True, min_periods=1).mean()
            log("noise", "Bruit", f"moyenne mobile {cfg['smooth_window']} sur {len(cols)} colonnes")
        else:
            log("noise", "Bruit", "ignoré (pas une série temporelle)")

    # encoding (mappings learned on train)
    cats = [c for c in tr.columns if c not in protected and not _is_num(tr[c])]
    if cfg["encoding"] != "none" and cats:
        enc, note = cfg["encoding"], ""
        if enc == "target" and not (target and (_is_classification(tr[target]) or _is_num(tr[target]))):
            enc = "frequency"
            note = "pas de cible exploitable → fréquence"
        if enc == "target":
            te_new = _te_apply(tr, te, cats, target, cfg["te_smoothing"])
            tr, n_new = _target_encode(tr, cats, target, cfg["te_smoothing"])
            te = te_new[list(tr.columns)]
            detail = f"{len(cats)} colonnes en target encoding (lissage {cfg['te_smoothing']:g}, 5 plis sur le train)"
        elif enc == "onehot":
            small = [c for c in cats if tr[c].nunique(dropna=True) <= cfg["max_onehot"]]
            big = [c for c in cats if c not in small]
            for c in small:
                cs = sorted(tr[c].dropna().unique(), key=str)
                tr[c] = pd.Categorical(tr[c], categories=cs)
                te[c] = pd.Categorical(te[c], categories=cs)
            if small:
                tr = pd.get_dummies(tr, columns=small, drop_first=True, dtype="int8")
                te = pd.get_dummies(te, columns=small, drop_first=True, dtype="int8")
            for c in big:
                _label_fit_apply(tr, te, c)
            detail = f"{len(small)} one-hot" + (f", {len(big)} label (> {cfg['max_onehot']} classes)" if big else "")
        elif enc == "label":
            for c in cats:
                _label_fit_apply(tr, te, c)
            detail = f"{len(cats)} colonnes en label"
        else:
            for c in cats:
                f = tr[c].value_counts(normalize=True)
                tr[c] = tr[c].map(f).astype(float)
                te[c] = te[c].map(f).fillna(0.0).astype(float)
            detail = f"{len(cats)} colonnes en fréquence" + (f" ({note})" if note else "")
        log("encoding", "Encodage", f"{detail} → {tr.shape[1]} colonnes")
    elif cats:
        log("encoding", "Encodage", f"désactivé ({len(cats)} colonnes texte conservées)")

    # target
    if target and cfg["encode_target"] and target in tr.columns and not _is_num(tr[target]):
        tr[target], mapping = _label_encode(tr[target])
        te[target] = te[target].astype(str).where(te[target].notna()).map(mapping)
        te = te.dropna(subset=[target]).reset_index(drop=True)
        te[target] = te[target].astype("int64")
        extra["target_mapping"] = mapping
        log("target_enc", "Cible encodée", f"{len(mapping)} classes")

    # feature scaling (statistics learned on train, applied to test) : numeric, non-binary columns, target/date excluded
    if cfg["scaling"] != "none":
        sc_cols = [c for c in tr.columns if c not in protected and c != SPLIT_COL and _is_num(tr[c]) and tr[c].nunique(dropna=True) > 2]
        if sc_cols:
            xtr = tr[sc_cols].astype(float)
            m = cfg["scaling"]
            if m == "standard":
                center, scale = xtr.mean(), xtr.std(ddof=0)
            elif m == "minmax":
                center, scale = xtr.min(), xtr.max() - xtr.min()
            else:                                                   # robust : median / IQR
                center, scale = xtr.median(), xtr.quantile(0.75) - xtr.quantile(0.25)
            scale = scale.replace(0, 1.0).fillna(1.0)
            center = center.fillna(0.0)
            tr[sc_cols] = (xtr - center) / scale
            if len(te):
                te[sc_cols] = (te[sc_cols].astype(float) - center) / scale
            name = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}[m]
            log("scaling", "Feature scaling", f"{name} sur {len(sc_cols)} colonnes numériques (paramètres appris sur le train)")
        else:
            log("scaling", "Feature scaling", "aucune colonne numérique à mettre à l'échelle")

    # class imbalance (SMOTE) : TRAIN only, test stays untouched
    n_orig = len(tr)
    if cfg["balance"]:
        if is_ts:
            log("smote", "Déséquilibre des classes", "ignoré (série temporelle)")
        elif not target or target not in tr.columns or not _is_classification(tr[target]):
            log("smote", "Déséquilibre des classes", "ignoré (choisissez une cible de classification)")
        else:
            extra["class_counts"] = {"before": _class_counts(tr[target])}
            tr, info = _smote(tr, target, cfg["smote_k"])
            extra["class_counts"]["after"] = _class_counts(tr[target])
            log("smote", "Déséquilibre des classes",
                f"SMOTE (k={cfg['smote_k']}) sur le train : {info['added']} lignes synthétiques" if info["added"]
                else f"SMOTE non appliqué ({info['skipped'] or 'classes déjà équilibrées'})")

    if len(te):
        out = pd.concat([tr.iloc[:n_orig], te, tr.iloc[n_orig:]], ignore_index=True)
        out[SPLIT_COL] = ["train"] * n_orig + ["test"] * len(te) + ["train"] * (len(tr) - n_orig)
    else:
        out = tr
    extra["start"] = {"rows": start[0], "cols": start[1], "missing": start[2]}
    extra["split"] = {"train": int(n_orig), "test": int(len(te))} if len(te) else None
    return out, steps, extra


def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


# ---------------------------------------------------------------------------------------------
# dataset level API
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# SQLITE STORAGE  (same results/analysis.db as analysis_engine, no more CSV / JSON files)
# ---------------------------------------------------------------------------------------------
#   preprocessing_runs (dataset, report JSON, sql_table, n_rows, bool_columns, datetime_columns, created_at)
#                                   one row per dataset = the latest run (replaces the previous one)
#   p__<dataset>                    the cleaned data of that run
# These tables are independent from analysis_engine's t__/o__ tables: re-running the analysis does not erase them.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS preprocessing_runs (
    dataset          TEXT PRIMARY KEY,
    report           TEXT NOT NULL,
    sql_table        TEXT NOT NULL,
    n_rows           INTEGER NOT NULL DEFAULT 0,
    bool_columns     TEXT NOT NULL DEFAULT '[]',
    datetime_columns TEXT NOT NULL DEFAULT '[]',
    created_at       TEXT NOT NULL
);
"""
MAX_SQLITE_COLUMNS = 2000      # SQLite's default SQLITE_MAX_COLUMN
INSERT_CHUNK = 10_000          # rows converted + inserted at a time (keeps memory low on big datasets)


def _connect(results_dir=None):
    conn = ae._connect(results_dir or ae.RESULTS_DIR)      # same file, same pragmas as analysis_engine
    try:
        conn.executescript(_SCHEMA)
    except Exception:
        pass
    return conn


def _drop_run(conn, dataset):
    row = conn.execute("SELECT sql_table FROM preprocessing_runs WHERE dataset=?", (dataset,)).fetchone()
    if row:
        conn.execute(f"DROP TABLE IF EXISTS {ae._quote(row[0])}")
    conn.execute("DELETE FROM preprocessing_runs WHERE dataset=?", (dataset,))


def _save_run(dataset, clean, report, results_dir=None):
    """Store the cleaned DataFrame + its report in ONE transaction (all-or-nothing, replaces the previous run)."""
    if clean.shape[1] > MAX_SQLITE_COLUMNS:
        raise ValueError(f"{clean.shape[1]} colonnes : SQLite en accepte au maximum {MAX_SQLITE_COLUMNS}")
    df = clean.copy()
    df.columns = ae._dedupe_columns(df.columns)

    bool_cols, dt_cols, col_defs = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt_cols.append(c)
            df[c] = s.map(lambda v: v.isoformat(sep=" ") if pd.notna(v) else None)
            col_defs.append(ae._quote(c))
        elif pd.api.types.is_bool_dtype(s):
            bool_cols.append(c)
            col_defs.append(f"{ae._quote(c)} INTEGER")
        elif pd.api.types.is_integer_dtype(s):
            col_defs.append(f"{ae._quote(c)} INTEGER")
        elif pd.api.types.is_float_dtype(s):
            col_defs.append(f"{ae._quote(c)} REAL")
        else:
            col_defs.append(ae._quote(c))          # no declared type: numbers stay numbers, text stays text

    conn = _connect(results_dir)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _drop_run(conn, dataset)
        # SQLite table names are case-insensitive: never reuse another dataset's table
        taken = {r[0].lower() for r in conn.execute("SELECT sql_table FROM preprocessing_runs")}
        sql_table, n = f"p__{dataset}", 2
        while sql_table.lower() in taken:
            sql_table, n = f"p__{dataset}_{n}", n + 1
        conn.execute(f"DROP TABLE IF EXISTS {ae._quote(sql_table)}")
        conn.execute(f"CREATE TABLE {ae._quote(sql_table)} ({', '.join(col_defs)})")

        sql = f"INSERT INTO {ae._quote(sql_table)} VALUES ({','.join('?' * len(df.columns))})"
        for i in range(0, len(df), INSERT_CHUNK):
            part = df.iloc[i:i + INSERT_CHUNK].astype(object)
            part = part.where(part.notna(), None)
            conn.executemany(sql, [tuple(ae._py_value(v) for v in row)
                                   for row in part.itertuples(index=False, name=None)])

        conn.execute(
            "INSERT INTO preprocessing_runs (dataset, report, sql_table, n_rows, bool_columns, datetime_columns, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (dataset, json.dumps(report, ensure_ascii=False, default=str), sql_table, int(len(df)),
             json.dumps(bool_cols), json.dumps(dt_cols), report["created_at"]),
        )
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _read_run(dataset_name, results_dir=None, with_data=False):
    """(report dict, DataFrame | None) of the saved run, or None if there is none."""
    dataset_name = ae.clean_dataset_name(dataset_name)
    if not ae.get_db_path(results_dir or ae.RESULTS_DIR).exists():
        return None
    conn = _connect(results_dir)
    try:
        row = conn.execute(
            "SELECT report, sql_table, bool_columns, datetime_columns FROM preprocessing_runs WHERE dataset=?",
            (dataset_name,)).fetchone()
        if row is None:
            return None
        report = json.loads(row[0])
        df = None
        if with_data:
            df = pd.read_sql_query(f"SELECT * FROM {ae._quote(row[1])}", conn)
            for c in json.loads(row[2]):
                if c in df.columns:
                    df[c] = df[c].astype(bool)
            for c in json.loads(row[3]):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            df.columns = [ae._restore_column(c) for c in df.columns]
    finally:
        conn.close()
    return report, df


def load_report(dataset_name, results_dir=None):
    """The report of the last saved run (dict), or None."""
    try:
        res = _read_run(dataset_name, results_dir)
    except Exception:
        return None
    return res[0] if res else None


def load_clean(dataset_name, results_dir=None):
    """The cleaned DataFrame of the last saved run, or None."""
    res = _read_run(dataset_name, results_dir, with_data=True)
    return res[1] if res else None


def has_preprocessed(dataset_name, results_dir=None):
    """Check whether a preprocessed run already exists in SQLite for this dataset."""
    return load_report(dataset_name, results_dir) is not None


def get_preprocessed_meta(dataset_name, results_dir=None):
    """Return lightweight metadata on the preprocessed run, or None."""
    rep = load_report(dataset_name, results_dir)
    if not rep:
        return None
    return {
        "exists": True,
        "created_at": rep.get("created_at"),
        "rows": rep.get("after", {}).get("rows"),
        "cols": rep.get("after", {}).get("cols"),
        "missing": rep.get("after", {}).get("missing"),
        "target": rep.get("config", {}).get("target"),
        "config": rep.get("config"),
    }


def get_pipeline_data(dataset_name, results_dir=None):
    """
    Returns the DataFrame to be used automatically by all subsequent pipeline stages
    (Split, Model Selection).
    Prioritizes the cleaned/preprocessed data directly from SQLite if available,
    otherwise falls back gracefully to the original raw dataset.

    Returns:
        df (pd.DataFrame): the dataset ready for downstream steps.
        is_preprocessed (bool): True if preprocessed data is being used.
        info (dict): summary / metadata about the data source.
    """
    clean_df = load_clean(dataset_name, results_dir)
    report = load_report(dataset_name, results_dir)
    if clean_df is not None and not clean_df.empty:
        return clean_df, True, {
            "source": "preprocessed",
            "n_rows": int(len(clean_df)),
            "n_cols": int(clean_df.shape[1]),
            "columns": list(clean_df.columns),
            "report": report,
            "created_at": report.get("created_at") if report else None,
            "target": report.get("config", {}).get("target") if report else None,
        }

    summary, raw_df = _load(dataset_name)
    return raw_df, False, {
        "source": "raw",
        "n_rows": int(len(raw_df)),
        "n_cols": int(raw_df.shape[1]),
        "columns": list(raw_df.columns),
        "report": None,
        "created_at": None,
        "target": summary.get("target_class"),
    }


def delete_preprocessed(dataset_name, results_dir=None):
    """Remove the saved preprocessing of a dataset. Returns True if there was one."""
    dataset_name = ae.clean_dataset_name(dataset_name)
    if not ae.get_db_path(results_dir or ae.RESULTS_DIR).exists():
        return False
    conn = _connect(results_dir)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existed = conn.execute("SELECT 1 FROM preprocessing_runs WHERE dataset=?", (dataset_name,)).fetchone() is not None
        _drop_run(conn, dataset_name)
        conn.execute("COMMIT")
        return existed
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _dtype(summary):
    """Dataset type used by the pipeline. The EDA labels a dataset with a date column but REPEATED timestamps
    (e.g. weatherHistory, DST hours) 'time_indexed_tabular': for preprocessing it is still a time series
    (chronological sort, moving average, no SMOTE)."""
    t = summary.get("dataset_type", "tabular")
    return "time_series" if t == "time_indexed_tabular" and summary.get("date_column") else t


def _load(dataset_name):
    summary, src = ae._resolve_source(dataset_name, None, ae.RESULTS_DIR)
    return summary, ae.load_data(src)


def get_profile(dataset_name, refresh=False):
    """Profile of the dataset (Isolation Forest included) : cached, only `last_run` is always read fresh."""
    prof = dict(step_cache.get_or_compute(dataset_name, "pp_profile", {}, lambda: _get_profile_compute(dataset_name), refresh=refresh))
    last = load_report(dataset_name)
    prof["last_run"] = {"created_at": last["created_at"], "after": last["after"], "config": last.get("config")} if last else None
    return prof


def _get_profile_compute(dataset_name):
    summary, df = _load(dataset_name)
    date_col = summary.get("date_column") if summary.get("date_column") in df.columns else None
    target = summary.get("target_class") if summary.get("target_class") in df.columns else None
    import split as _split                       # étape 02 : cible + part du test choisies là-bas
    sp = _split.get_config(dataset_name)
    if sp:
        target = sp["target"] if sp["target"] in df.columns else None
    cols = []
    for c in df.columns:
        s = df[c]
        n_unique = int(s.nunique(dropna=True))
        n_miss = int(s.isna().sum())
        n_out = 0
        classes = _class_counts(s) if (c != date_col and _is_classification(s)) else None
        if c == date_col:
            kind = "date"
        elif _is_num(s):
            try:
                kind = "id" if ae.is_id_like(df, c) else ("constante" if n_unique <= 1 else "numérique")
            except Exception:
                kind = "numérique"
        else:
            kind = "constante" if n_unique <= 1 else "catégorielle"
        cols.append({"name": str(c), "kind": kind, "dtype": str(s.dtype), "n_missing": n_miss,
                     "pct_missing": round(100 * n_miss / max(len(df), 1), 1), "n_unique": n_unique, "n_outliers": n_out,
                     "classes": classes})

    # outliers : One-Class SVM -> number of rows that the treatment would remove
    n_out_total = 0
    try:
        feats = _svm_features(df, {x for x in (target, date_col) if x})
        flag = _svm_flag(df, feats, DEFAULTS["ocsvm_nu"], target)
    except Exception:
        flag = None
    if flag is not None:
        n_out_total = int(flag.sum())

    tc = next((c["classes"] for c in cols if c["name"] == target), None) if target else None
    imbalanced = bool(tc and len(tc) >= 2 and min(tc.values()) / max(tc.values()) < 0.5)
    recommended = dict(DEFAULTS, target=target, **({"test_size": sp["test_size"]} if sp else {}), outliers="ocsvm" if n_out_total else "none", balance=imbalanced, scaling="standard")
    last = load_report(dataset_name)
    return {
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()), "n_duplicates": int(df.duplicated().sum()),
        "n_outliers": n_out_total,
        "dataset_type": _dtype(summary), "date_column": date_col, "target": target,
        "columns": cols, "recommended": recommended, "split": sp,
        "last_run": {"created_at": last["created_at"], "after": last["after"], "config": last.get("config")} if last else None,
    }


def process(dataset_name, raw_cfg, save=True, refresh=False):
    """Full pipeline. Same settings on the same data = result read back from SQLite, nothing is recomputed."""
    raw_cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
    if "target" not in raw_cfg:
        raw_cfg["target"] = (ae.load_summary(dataset_name) or {}).get("target_class")
    cfg = clean_config(raw_cfg)
    kind = "pp_run" if save else "pp_preview"
    if not refresh:
        hit = step_cache.get(dataset_name, kind, cfg)
        if hit is not None:
            if not save:
                return hit
            saved = load_report(dataset_name)          # the cached run is only valid if it is still THE saved one
            if saved and saved.get("created_at") == hit.get("created_at"):
                return hit
    result = _process_compute(dataset_name, raw_cfg, save)
    step_cache.put(dataset_name, kind, cfg, result)
    return result


def _process_compute(dataset_name, raw_cfg, save=True):
    summary, df = _load(dataset_name)
    raw_cfg = dict(raw_cfg)
    date_col = summary.get("date_column")
    clean, steps, extra = run(df, raw_cfg, _dtype(summary), date_col)

    r, c, m = _state(clean)
    result = {
        "dataset": ae.clean_dataset_name(dataset_name),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": clean_config(raw_cfg),
        "before": extra["start"],
        "after": {"rows": r, "cols": c, "missing": m},
        "steps": steps,
        "target_mapping": extra["target_mapping"],
        "class_counts": extra.get("class_counts"),
        "split": extra.get("split"),
        "preview": _preview(clean),
        "saved": False,
    }
    if save:
        name = ae.clean_dataset_name(dataset_name)
        report = {k: v for k, v in result.items() if k != "preview"}
        report["columns"] = [str(x) for x in clean.columns]
        report["saved"] = True
        _save_run(name, clean, report)
        result["saved"] = True
        result["download_name"] = f"{name}_preprocessed.csv"
    return result


# ---------------------------------------------------------------------------------------------
# step by step view : the result of ONE step, with every earlier (enabled) step already applied
# ---------------------------------------------------------------------------------------------
STEP_ORDER = ["duplicates", "useless", "missing", "outliers", "noise", "encoding", "scaling", "smote"]
_OFF = {
    "duplicates": {"drop_duplicates": False},
    "useless": {"drop_useless": False},
    "missing": {"num_missing": "none", "cat_missing": "none"},
    "outliers": {"outliers": "none"},
    "noise": {"smooth_window": 0},
    "encoding": {"encoding": "none", "encode_target": False},
    "scaling": {"scaling": "none"},
    "smote": {"balance": False},
}
MAX_PTS = 1500


def _cfg_without(cfg, steps):
    out = dict(cfg)
    for st in steps:
        out.update(_OFF[st])
    return out


def _r(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _sample(values, m=MAX_PTS):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) > m:
        v = v[np.linspace(0, len(v) - 1, m).astype(int)]
    return [round(float(x), 4) for x in v]


def _st(df):
    r, c, m = _state(df)
    return {"rows": r, "cols": c, "missing": m}


def _step_payload(step, b, a, cfg, is_ts, date_col, target, steps_after, extra):
    enabled = {
        "duplicates": cfg["drop_duplicates"], "useless": cfg["drop_useless"],
        "missing": cfg["num_missing"] != "none" or cfg["cat_missing"] != "none",
        "outliers": cfg["outliers"] != "none", "noise": cfg["smooth_window"] >= 2 and is_ts,
        "encoding": cfg["encoding"] != "none", "scaling": cfg["scaling"] != "none", "smote": cfg["balance"],
    }[step]
    key = {"duplicates": "duplicates", "useless": "useless", "missing": "missing", "outliers": "outliers",
           "noise": "noise", "encoding": "encoding", "scaling": "scaling", "smote": "smote"}[step]
    detail = next((x["detail"] for x in reversed(steps_after) if x["key"] == key), "")
    out = {"step": step, "enabled": bool(enabled), "before": _st(b), "after": _st(a), "detail": detail}
    protected = {c for c in (target, date_col) if c}
    num_b = [c for c in b.columns if c not in protected and _is_num(b[c])]

    if step == "duplicates":
        b0 = b.drop(columns=[SPLIT_COL], errors="ignore")
        dup = b0[b0.duplicated(keep=False)]
        out["n_duplicates"] = int(b0.duplicated().sum())
        out["sample"] = _preview(dup, 10) if len(dup) else None

    elif step == "useless":
        dropped = []
        for c in b.columns:
            if c in a.columns:
                continue
            pct = float(b[c].isna().mean() * 100) if len(b) else 0.0
            if pct > cfg["missing_col_threshold"]:
                why = f"Trop de valeurs vides ({pct:.0f} %)"
            elif b[c].nunique(dropna=True) <= 1:
                why = "Constante (une seule valeur)"
            else:
                why = "Identifiant (une valeur différente par ligne)"
            dropped.append({"name": str(c), "reason": why})
        out["dropped"] = dropped

    elif step == "missing":
        cols = []
        for c in b.columns:
            if c in a.columns and c != date_col and b[c].isna().any():
                row = {"name": str(c), "kind": "num" if _is_num(b[c]) else "cat",
                       "before": int(b[c].isna().sum()), "after": int(a[c].isna().sum())}
                if row["kind"] == "num":
                    row.update(mean_before=_r(b[c].mean()), mean_after=_r(a[c].mean()),
                               std_before=_r(b[c].std()), std_after=_r(a[c].std()))
                cols.append(row)
        out["columns"] = cols
        ex = []
        if len(a) == len(b):
            for row in cols:
                c = row["name"]
                for pos in np.where(b[c].isna().to_numpy())[0][:2]:
                    v = a[c].iloc[pos]
                    ex.append({"col": c, "row": int(pos) + 1, "value": _r(v) if _is_num(a[c]) else str(v)})
                if len(ex) >= 8:
                    break
        out["examples"] = ex[:8]

    elif step == "outliers":
        rows = []
        bt = b[b[SPLIT_COL] == "train"] if SPLIT_COL in b.columns else b      # learned on the train rows only
        feats = _svm_features(bt, protected | {SPLIT_COL})
        flag, view = _svm_view(bt, feats, cfg["ocsvm_nu"], target) if enabled else (None, None)
        n_flagged = int(flag.sum()) if flag is not None else 0
        out["n_rows_flagged"] = n_flagged
        out["nu"] = cfg["ocsvm_nu"]
        out["pct_rows_flagged"] = round(100 * n_flagged / max(len(bt), 1), 2)
        out["rows_before"], out["rows_after"] = int(len(b)), int(len(a))
        for c in num_b:
            if c not in a.columns:
                continue
            rows.append({"name": str(c), "min_before": _r(b[c].min()), "max_before": _r(b[c].max()),
                         "min_after": _r(a[c].min()), "max_after": _r(a[c].max()),
                         "std_before": _r(b[c].std()), "std_after": _r(a[c].std())})
        rows.sort(key=lambda r: -((r["std_before"] or 0) - (r["std_after"] or 0)))
        out["columns"] = rows
        out["svm"] = view                  # None when disabled / not enough data
        out["method"] = "one_class_svm"

    elif step == "noise":
        out["applicable"] = bool(is_ts)
        if enabled:
            x = (pd.to_datetime(b[date_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("").tolist()
                 if date_col and date_col in b.columns else list(range(len(b))))
            idx = np.linspace(0, len(b) - 1, min(len(b), MAX_PTS)).astype(int) if len(b) else []
            cols, series = [], []
            for c in [c for c in num_b if c in a.columns and b[c].nunique() > 2][:12]:
                rb, ra = float(b[c].diff().std()), float(a[c].diff().std())
                cols.append({"name": str(c), "rough_before": _r(rb), "rough_after": _r(ra),
                             "reduction": _r(100 * (1 - ra / rb), 1) if rb else None})
                series.append({"name": str(c), "raw": [_r(b[c].iloc[i], 4) for i in idx], "smooth": [_r(a[c].iloc[i], 4) for i in idx]})
            out.update(x=[x[i] for i in idx], columns=cols, series=series, window=cfg["smooth_window"])

    elif step == "encoding":
        cats = [c for c in b.columns if c not in protected and c != SPLIT_COL and not _is_num(b[c])]
        cols = []
        if len(a) == len(b):
            for c in cats[:12]:
                names = [x for x in a.columns if x == c or x.startswith(f"{c}_te_")]
                mapping = None
                if names and _is_num(a[names[0]]):
                    g = a[names[0]].groupby(b[c].astype(str).to_numpy()).agg(["mean", "count"]).sort_values("count", ascending=False).head(15)
                    mapping = [{"cat": str(i), "count": int(r["count"]), "value": _r(r["mean"], 4)} for i, r in g.iterrows()]
                cols.append({"name": str(c), "n_classes": int(b[c].nunique(dropna=True)), "out": names[0] if names else None, "mapping": mapping})
            show = [x for x in cats[:4]]
            outc = [x["out"] for x in cols[:4] if x["out"]]
            out["sample"] = {"before": _preview(b[show], 6), "after": _preview(a[outc], 6)} if show and outc else None
        out["columns"] = cols
        out["method"] = cfg["encoding"]
        out["target_mapping"] = extra.get("target_mapping")
        out["needs_target"] = bool(cfg["encoding"] == "target" and not target)

    elif step == "scaling":
        cols, series = [], []
        if enabled:
            for c in [c for c in num_b if c in a.columns and b[c].nunique(dropna=True) > 2][:30]:
                cols.append({"name": str(c),
                             "mean_before": _r(b[c].mean()), "std_before": _r(b[c].std()),
                             "min_before": _r(b[c].min()), "max_before": _r(b[c].max()),
                             "mean_after": _r(a[c].mean()), "std_after": _r(a[c].std()),
                             "min_after": _r(a[c].min()), "max_after": _r(a[c].max())})
            for r in cols[:15]:
                series.append({"name": r["name"], "before": _sample(b[r["name"]]), "after": _sample(a[r["name"]])})
        out["columns"], out["series"], out["method"] = cols, series, cfg["scaling"]

    elif step == "smote":
        out["applicable"] = bool(target and target in a.columns and _is_classification(a[target]) and not is_ts)
        if target and target in b.columns and _is_classification(b[target]):
            out["counts_before"] = _class_counts(b[target])
            out["counts_after"] = _class_counts(a[target])
        n0 = len(b)
        if enabled and len(a) > n0 and target in a.columns:
            feats = [c for c in a.columns if c != target and _is_num(a[c]) and a[c].nunique() > 5][:6]
            o, sy = a.iloc[:n0], a.iloc[n0:]
            def pick(d):
                if len(d) > 800:
                    d = d.iloc[np.linspace(0, len(d) - 1, 800).astype(int)]
                return d
            o, sy = pick(o), pick(sy)
            out["features"] = feats
            out["orig"] = {c: [_r(v, 4) for v in o[c]] for c in feats}
            out["synth"] = {c: [_r(v, 4) for v in sy[c]] for c in feats}
            out["orig_y"] = [str(v) for v in o[target]]
            out["synth_y"] = [str(v) for v in sy[target]]
    return out


def step_result(dataset_name, step, raw_cfg, refresh=False):
    """Result of a single step (read from the SQLite cache when the same settings were already computed)."""
    if step not in STEP_ORDER:
        raise KeyError(f"étape inconnue : {step}")
    raw_cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
    if "target" not in raw_cfg:
        raw_cfg["target"] = (ae.load_summary(dataset_name) or {}).get("target_class")
    cfg = clean_config(raw_cfg)
    return step_cache.get_or_compute(dataset_name, f"pp_step:{step}", cfg,
                                     lambda: _step_result_compute(dataset_name, step, cfg), refresh=refresh)


def _step_result_compute(dataset_name, step, cfg):
    """Pipeline run twice (steps >= `step` disabled, then only later steps disabled)."""
    summary, df = _load(dataset_name)
    dtype, date_col = _dtype(summary), summary.get("date_column")
    i = STEP_ORDER.index(step)
    b, _, _ = run(df, _cfg_without(cfg, STEP_ORDER[i:]), dtype, date_col)
    a, steps_after, extra = run(df, _cfg_without(cfg, STEP_ORDER[i + 1:]), dtype, date_col)
    target = cfg["target"] if cfg["target"] in df.columns else None
    dc = date_col if date_col in df.columns else None
    is_ts = dtype == "time_series" and dc is not None
    return _step_payload(step, b, a, cfg, is_ts, dc, target, steps_after, extra)
