"""
Étape 03 — SPLIT TRAIN / TEST (séparée du prétraitement).

Pipeline : EDA -> Prétraitement -> Split -> Model Selection

Cette étape se fait en 3 temps :
  1. identifier X (variables explicatives) et y (variable cible),
  2. choisir la stratégie de découpage (aléatoire / stratifié / chronologique),
  3. choisir le ratio (part du jeu de test, 0 = pas de split),
et affiche l'aperçu du découpage.

Le prétraitement relit cette configuration : il applique le découpage avant toute étape apprise
(imputation, outliers, encodage, SMOTE) pour éviter toute fuite de données du test vers le train.
La config est stockée dans la table `split_config` (même base SQLite que le reste du pipeline).
"""
import json
from datetime import datetime

import analysis_engine as ae
import preprocessing as pp
import step_cache

DEFAULT_TEST_SIZE = 0.2
MAX_TEST_SIZE = 0.5
STRATEGIES = ("random", "stratified", "chronological")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS split_config (
    dataset    TEXT PRIMARY KEY,
    config     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _connect():
    conn = ae._connect(ae.RESULTS_DIR)
    conn.executescript(_SCHEMA)
    return conn


def get_config(dataset_name):
    """Config sauvegardée {'test_size': float, 'target': str|None, 'created_at': str} ou None."""
    name = ae.clean_dataset_name(dataset_name)
    conn = _connect()
    try:
        row = conn.execute("SELECT config, created_at FROM split_config WHERE dataset=?", (name,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    cfg = json.loads(row[0])
    cfg["created_at"] = row[1]
    return cfg


def delete_config(dataset_name):
    name = ae.clean_dataset_name(dataset_name)
    conn = _connect()
    try:
        conn.execute("DELETE FROM split_config WHERE dataset=?", (name,))
        conn.commit()
    finally:
        conn.close()


def _clean(raw, columns, default_target):
    raw = raw if isinstance(raw, dict) else {}
    try:
        ts = float(raw.get("test_size", DEFAULT_TEST_SIZE))
    except (TypeError, ValueError):
        ts = DEFAULT_TEST_SIZE
    if ts > 1:                      # saisi en pourcentage
        ts /= 100.0
    ts = min(MAX_TEST_SIZE, max(0.0, ts))
    target = raw.get("target", default_target)
    target = target if target in columns else None
    # X : liste de variables choisies (None = toutes sauf la cible)
    feats = raw.get("features")
    if isinstance(feats, (list, tuple)):
        feats = [str(c) for c in feats if str(c) in columns and str(c) != target]
        if len(feats) == len([c for c in columns if c != target]):
            feats = None
    else:
        feats = None
    strategy = raw.get("strategy")
    strategy = strategy if strategy in STRATEGIES else "auto"
    excluded = [c for c in columns if c != target and feats is not None and c not in feats]
    return {"test_size": round(ts, 4), "target": target, "features": feats, "excluded": excluded, "strategy": strategy}


def _preview(dataset_name, cfg, refresh=False):
    """Aperçu du découpage : relu dans la base si ces réglages ont déjà été calculés sur ces données."""
    return step_cache.get_or_compute(dataset_name, "split_preview", cfg,
                                     lambda: _preview_compute(dataset_name, cfg), refresh=refresh, with_split=False)


def _preview_compute(dataset_name, cfg):
    """Aperçu du découpage avec la même fonction que le prétraitement (mêmes lignes train/test)."""
    summary, df = pp._load(dataset_name)
    is_ts = pp._dtype(summary) == "time_series"
    target = cfg["target"]
    if target:
        df = df.dropna(subset=[target])
    n = int(len(df))
    out = {"n_rows": n, "n_train": n, "n_test": 0, "mode": "aucun split", "classes": None, "note": None}
    if cfg["test_size"] <= 0 or n < 10:
        if cfg["test_size"] > 0:
            out["mode"] = "trop peu de lignes (< 10) : pas de split"
        return out
    y = df[target].reset_index(drop=True) if target else None
    strategy = cfg.get("strategy", "auto")
    if strategy == "stratified" and not (y is not None and pp._is_classification(y)):
        out["note"] = "Stratifié demandé mais la cible n'est pas une variable de classification : découpage aléatoire utilisé."
    mask, mode = pp._split_mask(n, y, is_ts, cfg["test_size"], strategy=strategy)
    out.update(n_train=int((~mask).sum()), n_test=int(mask.sum()), mode=mode)
    if y is not None and pp._is_classification(y):
        tr = y[~mask].astype(str).value_counts()
        te = y[mask].astype(str).value_counts()
        out["classes"] = [{"label": str(k), "train": int(tr.get(k, 0)), "test": int(te.get(k, 0))}
                          for k in y.astype(str).value_counts().index[:12]]
    return out


def get_profile(dataset_name):
    summary, df = pp._load(dataset_name)
    date_col = summary.get("date_column") if summary.get("date_column") in df.columns else None
    detected = summary.get("target_class") if summary.get("target_class") in df.columns else None
    saved = get_config(dataset_name)
    columns = [str(c) for c in df.columns if c != date_col]
    cfg = _clean(saved or {"test_size": DEFAULT_TEST_SIZE, "target": detected}, columns, detected)
    info = []
    for c in columns:
        s = df[c]
        try:
            classif = bool(pp._is_classification(s))
        except Exception:
            classif = False
        info.append({"name": c, "numeric": bool(pp._is_num(s)), "classif": classif, "n_unique": int(s.nunique(dropna=True))})
    return {
        "columns": columns,
        "column_info": info,
        "n_rows": int(len(df)),
        "dataset_type": pp._dtype(summary),
        "date_column": date_col,
        "detected_target": detected,
        "config": cfg,
        "saved": bool(saved),
        "saved_at": saved["created_at"] if saved else None,
        "preview": _preview(dataset_name, cfg),
    }


def preview(dataset_name, raw_cfg):
    """Aperçu sans sauvegarde (pour mettre à jour la page pendant que l'on règle les paramètres)."""
    summary, df = pp._load(dataset_name)
    date_col = summary.get("date_column")
    columns = [str(c) for c in df.columns if c != date_col]
    cfg = _clean(raw_cfg, columns, summary.get("target_class"))
    return {"config": cfg, "preview": _preview(dataset_name, cfg)}


def save(dataset_name, raw_cfg):
    summary, df = pp._load(dataset_name)
    date_col = summary.get("date_column")
    columns = [str(c) for c in df.columns if c != date_col]
    cfg = _clean(raw_cfg, columns, summary.get("target_class"))
    name = ae.clean_dataset_name(dataset_name)
    now = datetime.now().isoformat(timespec="seconds")
    conn = _connect()
    try:
        conn.execute("DELETE FROM split_config WHERE dataset=?", (name,))
        conn.execute("INSERT INTO split_config (dataset, config, created_at) VALUES (?,?,?)",
                     (name, json.dumps(cfg), now))
        conn.commit()
    finally:
        conn.close()
    return {"config": cfg, "saved_at": now, "preview": _preview(dataset_name, cfg)}
