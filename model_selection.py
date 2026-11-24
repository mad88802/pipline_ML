"""
Étape 04 — MODEL SELECTION (classification + régression).

Pipeline : EDA -> Prétraitement -> Split -> Model Selection

Deux familles, chacune en version classification ET régression :
  - Modèles individuels : SVM (SVC / SVR), KNN, Arbre de décision
  - Ensemble learning   : Random Forest, AdaBoost, XGBoost, CatBoost, LightGBM
  (XGBoost, CatBoost et LightGBM sont des bibliothèques optionnelles : pip install xgboost catboost lightgbm)

Principe (anti-fuite de données) :
  * les données viennent automatiquement du prétraitement (SQLite), sinon du dataset brut ;
  * le train / test est celui de l'étape Split (colonne `__split` du prétraitement, sinon même découpage recalculé) ;
  * l'imputation / l'encodage / la mise à l'échelle sont dans un Pipeline sklearn : ils sont appris sur le TRAIN seulement ;
  * les modèles sont CLASSÉS sur une validation (20 % du train mis de côté) ; le test ne sert qu'à l'évaluation finale ;
  * métriques : classification = F1 pondéré (+ AUC ROC), régression = R².

Les résultats sont stockés dans la table `model_results` (même base SQLite que le reste du pipeline).
"""
import importlib.util
import os
import json
import math
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import step_cache
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import analysis_engine as ae
import preprocessing as pp
import split as sp

SEED = 42
SVM_MAX_TRAIN = 3000          # SVM : coût ~ n² -> on limite le nombre de lignes d'entraînement
MAX_CAT_UNIQUE = 50           # colonne texte avec plus de modalités : ignorée (one-hot trop large)
MAX_POINTS = 500              # points affichés pour « prédit vs réel »
VAL_FRACTION = 0.2            # part du train mise de côté pour la validation (choix des variantes / hyperparamètres / classement)
VAL_MAX_ROWS = 15000          # validation / GridSearch calculées sur un échantillon (le modèle final est ré-entraîné sur tout le train)
TUNE_MAX_ROWS = 6000          # GridSearch : échantillon plus petit
TRAIN_EVAL_ROWS = 5000        # score « train » estimé sur un échantillon (KNN / SVM prédisent lentement)
MAX_CLASSES = 50              # au-delà, la cible n'est pas une variable de classification

GROUPS = {
    "individual": {"title": "Modèles individuels", "sub": "SVM · KNN · Arbre", "models": ["svm", "knn", "tree"]},
    "ensemble": {"title": "Ensemble learning", "sub": "RF · AdaBoost · XGBoost · CatBoost · LightGBM",
                 "models": ["rf", "ada", "xgb", "cat", "lgbm"]},
}


def _P(k, label, typ, default, **kw):
    return {"k": k, "label": label, "type": typ, "default": default, **kw}


MODELS = {
    "svm": {
        "name": "SVM", "group": "individual", "scaled": True, "lib": None,
        "names": {"classification": "SVC", "regression": "SVR"},
        "desc": "Cherche la frontière (classification) ou la courbe (régression) qui sépare / suit le mieux les points, avec une marge maximale. Sensible à l'échelle : les variables sont standardisées.",
        "params": [_P("C", "C", "float", 1.0, min=0.001, max=1000), _P("kernel", "Noyau", "choice", "rbf", choices=["rbf", "linear", "poly", "sigmoid"])],
        "grid": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    },
    "knn": {
        "name": "KNN", "group": "individual", "scaled": True, "lib": None,
        "names": {"classification": "KNeighborsClassifier", "regression": "KNeighborsRegressor"},
        "desc": "Prédit à partir des k observations les plus proches (vote majoritaire ou moyenne). Aucun apprentissage, mais sensible à l'échelle : les variables sont standardisées.",
        "params": [_P("n_neighbors", "k (voisins)", "int", 5, min=1, max=100), _P("weights", "Poids", "choice", "uniform", choices=["uniform", "distance"])],
        "grid": {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
    },
    "tree": {
        "name": "Arbre de décision", "group": "individual", "scaled": False, "lib": None,
        "names": {"classification": "DecisionTreeClassifier", "regression": "DecisionTreeRegressor"},
        "desc": "Enchaîne des questions « variable ≤ seuil ? » pour arriver à une prédiction. Lisible et sans mise à l'échelle, mais tend à sur-apprendre sans limite de profondeur.",
        "params": [_P("max_depth", "Profondeur max", "int", None, min=1, max=50, nullable=True), _P("min_samples_leaf", "Min. éch. / feuille", "int", 1, min=1, max=200)],
        "grid": {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 10]},
    },
    "rf": {
        "name": "Random Forest", "group": "ensemble", "scaled": False, "lib": None,
        "names": {"classification": "RandomForestClassifier", "regression": "RandomForestRegressor"},
        "desc": "Bagging : de nombreux arbres entraînés sur des échantillons aléatoires dont on moyenne (ou fait voter) les prédictions. Robuste et peu sensible au sur-apprentissage.",
        "params": [_P("n_estimators", "Nb d'arbres", "int", 100, min=10, max=1000), _P("max_depth", "Profondeur max", "int", None, min=1, max=50, nullable=True)],
        "grid": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    },
    "ada": {
        "name": "AdaBoost", "group": "ensemble", "scaled": False, "lib": None,
        "names": {"classification": "AdaBoostClassifier", "regression": "AdaBoostRegressor"},
        "desc": "Boosting : des modèles faibles (petits arbres) entraînés à la suite, chacun insistant sur les observations mal prédites par les précédents.",
        "params": [_P("n_estimators", "Nb d'estimateurs", "int", 50, min=10, max=1000), _P("learning_rate", "Learning rate", "float", 1.0, min=0.001, max=5)],
        "grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
    },
    "xgb": {
        "name": "XGBoost", "group": "ensemble", "scaled": False, "lib": "xgboost",
        "names": {"classification": "XGBClassifier", "regression": "XGBRegressor"},
        "desc": "Gradient boosting très optimisé : chaque arbre corrige les erreurs des précédents, avec régularisation. Souvent parmi les meilleurs sur données tabulaires.",
        "params": [_P("n_estimators", "Nb d'arbres", "int", 100, min=10, max=2000), _P("learning_rate", "Learning rate", "float", 0.1, min=0.001, max=1), _P("max_depth", "Profondeur max", "int", 6, min=1, max=20)],
        "grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.3], "max_depth": [3, 6]},
    },
    "cat": {
        "name": "CatBoost", "group": "ensemble", "scaled": False, "lib": "catboost",
        "names": {"classification": "CatBoostClassifier", "regression": "CatBoostRegressor"},
        "desc": "Gradient boosting à arbres symétriques, très stable avec peu de réglages et bon sur les variables catégorielles.",
        "params": [_P("iterations", "Itérations", "int", 200, min=10, max=2000), _P("learning_rate", "Learning rate", "float", 0.1, min=0.001, max=1), _P("depth", "Profondeur", "int", 6, min=1, max=12)],
        "grid": {"depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
    },
    "lgbm": {
        "name": "LightGBM", "group": "ensemble", "scaled": False, "lib": "lightgbm",
        "names": {"classification": "LGBMClassifier", "regression": "LGBMRegressor"},
        "desc": "Gradient boosting rapide qui fait pousser les arbres feuille par feuille (leaf-wise) : très efficace sur de gros volumes de données.",
        "params": [_P("n_estimators", "Nb d'arbres", "int", 100, min=10, max=2000), _P("learning_rate", "Learning rate", "float", 0.1, min=0.001, max=1), _P("num_leaves", "Nb de feuilles", "int", 31, min=2, max=256)],
        "grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [15, 31, 63]},
    },
}


_IMPORT_ERR = {}


def available(key):
    """True si la bibliothèque du modèle s'importe VRAIMENT dans ce processus (find_spec seul ne détecte pas une DLL cassée)."""
    lib = MODELS[key]["lib"]
    if lib is None:
        return True
    if lib not in _IMPORT_ERR:
        try:
            importlib.import_module(lib)
            _IMPORT_ERR[lib] = None
        except Exception as e:                                # absente, ou installée mais inutilisable
            _IMPORT_ERR[lib] = f"{type(e).__name__}: {e}"
    return _IMPORT_ERR[lib] is None


def install_hint(key):
    """Commande pip qui cible exactement l'interpréteur Python qui exécute cette application."""
    lib = MODELS[key]["lib"]
    return f'"{sys.executable}" -m pip install {lib}' if lib else None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    dataset    TEXT NOT NULL,
    task       TEXT NOT NULL,
    grp        TEXT NOT NULL,
    result     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset, task, grp)
);
"""


def _connect():
    conn = ae._connect(ae.RESULTS_DIR)
    conn.executescript(_SCHEMA)
    return conn


def _purge_old_schema(conn):
    """Supprime les anciens résultats (validation croisée, autres métriques) : ils ne sont plus lisibles par l'interface."""
    conn.execute("""DELETE FROM model_runs WHERE result NOT LIKE '%"schema": 2%'""")


def get_results(dataset_name, task, group):
    """Résultats sauvegardés de (task, group) : dict ou None."""
    name = ae.clean_dataset_name(dataset_name)
    conn = _connect()
    try:
        _purge_old_schema(conn)
        row = conn.execute("SELECT result, created_at FROM model_runs WHERE dataset=? AND task=? AND grp=?",
                           (name, task, group)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    res = json.loads(row[0])
    res["created_at"] = row[1]
    return res


def saved_overview(dataset_name):
    """{task: {group: created_at}} des entraînements déjà faits."""
    name = ae.clean_dataset_name(dataset_name)
    out = {"classification": {}, "regression": {}}
    try:
        conn = _connect()
        try:
            _purge_old_schema(conn)
            for task, grp, at in conn.execute("SELECT task, grp, created_at FROM model_runs WHERE dataset=?", (name,)):
                out.setdefault(task, {})[grp] = at
        finally:
            conn.close()
    except Exception:
        pass
    return out


def has_results(dataset_name):
    return any(v for v in saved_overview(dataset_name).values())


def delete_results(dataset_name):
    """Supprime tous les résultats (obsolètes si le split ou le prétraitement change)."""
    name = ae.clean_dataset_name(dataset_name)
    try:
        conn = _connect()
        try:
            conn.execute("DELETE FROM model_runs WHERE dataset=?", (name,))
        finally:
            conn.close()
    except Exception:
        pass


def _save(dataset_name, task, group, result, created_at=None):
    name = ae.clean_dataset_name(dataset_name)
    now = created_at or datetime.now().isoformat(timespec="seconds")
    conn = _connect()
    try:
        conn.execute("DELETE FROM model_runs WHERE dataset=? AND task=? AND grp=?", (name, task, group))
        conn.execute("INSERT INTO model_runs (dataset, task, grp, result, created_at) VALUES (?,?,?,?,?)",
                     (name, task, group, json.dumps(result), now))
    finally:
        conn.close()
    return now


# ---------------------------------------------------------------------------------------------
# données
# ---------------------------------------------------------------------------------------------
def _num(v, nd=4):
    """float JSON-safe (NaN / inf -> None)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _target_and_split(dataset_name, summary, info):
    sc = sp.get_config(dataset_name) or {}
    target = sc.get("target") or info.get("target") or summary.get("target_class")
    return target, sc


def _dataset(dataset_name, data=None):
    """Charge les données de l'étape précédente + cible + masque de test + liste des colonnes explicatives.

    `data` = (DataFrame, is_preprocessed, info) permet de fournir des données déjà prétraitées (ex. variantes
    « avec / sans outliers » du rapport de comparaison) au lieu de relire celles enregistrées."""
    if data is None:
        summary, _ = pp._load(dataset_name)
        df, is_pre, info = pp.get_pipeline_data(dataset_name)
    else:
        summary = ae.load_summary(dataset_name) or {}
        df, is_pre, info = data
    target, sc = _target_and_split(dataset_name, summary, info)
    if not target or target not in df.columns:
        raise ValueError("Aucune variable cible : choisissez y dans l'étape Split (03).")

    df = df.copy()
    date_col = summary.get("date_column")
    drop = {target, pp.SPLIT_COL}
    if date_col in df.columns:
        drop.add(date_col)
    drop |= {c for c in df.columns if str(df[c].dtype).startswith("datetime")}
    drop |= set(sc.get("excluded") or [])

    df = df[df[target].notna()].reset_index(drop=True)
    is_ts = pp._dtype(summary) == "time_series"
    strategy = sc.get("strategy", "auto")
    chronological = strategy == "chronological" or (strategy == "auto" and is_ts)

    y_raw = df[target]
    if pp.SPLIT_COL in df.columns:
        test = (df[pp.SPLIT_COL].astype(str) == "test").to_numpy()
        mode = "défini par l'étape Split / prétraitement"
    else:
        ts = sc.get("test_size", sp.DEFAULT_TEST_SIZE)
        if ts <= 0 or len(df) < 10:
            test = np.zeros(len(df), dtype=bool)
            mode = "aucun"
        else:
            test, mode = pp._split_mask(len(df), y_raw.reset_index(drop=True), is_ts, ts, seed=SEED, strategy=strategy)
    if test.sum() == 0 or (~test).sum() == 0:
        raise ValueError("Pas de jeu de test : choisissez un ratio de test > 0 % dans l'étape Split (03).")

    feats = [c for c in df.columns if c not in drop]
    X = df[feats].copy()
    for c in X.columns:                                  # bool -> 0/1
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    ignored = []
    for c in list(X.columns):
        if not pp._is_num(X[c]) and X[c].nunique(dropna=True) > MAX_CAT_UNIQUE:
            ignored.append(str(c))
            X = X.drop(columns=c)
    if X.shape[1] == 0:
        raise ValueError("Aucune variable explicative utilisable (X est vide).")
    return {"X": X, "y_raw": y_raw, "test": test, "target": target, "is_pre": is_pre, "mode": mode,
            "chronological": bool(chronological), "ignored": ignored, "summary": summary}


def _task_of(y_raw, requested):
    can_reg = pp._is_num(y_raw)
    if requested == "regression":
        if not can_reg:
            raise ValueError("La régression exige une cible numérique ; la cible choisie est du texte.")
        return "regression"
    if requested == "classification":
        n = y_raw.nunique(dropna=True)
        if n < 2:
            raise ValueError("La classification exige au moins 2 classes dans la cible.")
        if n > MAX_CLASSES:
            raise ValueError(f"La cible a {n} valeurs distinctes (> {MAX_CLASSES}) : c'est une variable continue, utilisez la régression.")
        return "classification"
    return "classification" if pp._is_classification(y_raw) or not can_reg else "regression"


# ---------------------------------------------------------------------------------------------
# modèles
# ---------------------------------------------------------------------------------------------
class _XGBClf(BaseEstimator, ClassifierMixin):
    """XGBClassifier qui réencode lui-même les classes (XGBoost exige 0..k-1 sans trou, y compris sur le sous-échantillon d'entraînement)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=SEED):
        self.n_estimators, self.learning_rate, self.max_depth, self.random_state = n_estimators, learning_rate, max_depth, random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.m_ = XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
                                random_state=self.random_state, n_jobs=-1, verbosity=0)
        self.m_.fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        return self.le_.inverse_transform(np.asarray(self.m_.predict(X)).astype(int))

    def predict_proba(self, X):
        return self.m_.predict_proba(X)


def _estimator(key, task, params):
    clf = task == "classification"
    p = dict(params or {})
    if key == "svm":
        cls = SVC if clf else SVR
        return cls(C=float(p.get("C", 1.0)), kernel=p.get("kernel", "rbf"), degree=int(p.get("degree", 3)), max_iter=int(p.get("max_iter", -1)),
                   **({"random_state": SEED} if clf else {}))
    if key == "knn":
        cls = KNeighborsClassifier if clf else KNeighborsRegressor
        return cls(n_neighbors=int(p.get("n_neighbors", 5)), weights=p.get("weights", "uniform"), metric=p.get("metric", "minkowski"))
    if key == "tree":
        cls = DecisionTreeClassifier if clf else DecisionTreeRegressor
        return cls(max_depth=p.get("max_depth"), min_samples_leaf=int(p.get("min_samples_leaf", 1)),
                   criterion=p.get("criterion", "gini" if clf else "squared_error"), splitter=p.get("splitter", "best"), random_state=SEED)
    if key == "rf":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(n_estimators=int(p.get("n_estimators", 100)), max_depth=p.get("max_depth"),
                   max_features=(1.0 if p.get("max_features") == "all" else p.get("max_features", "sqrt" if clf else 1.0)), random_state=SEED, n_jobs=-1)
    if key == "ada":
        cls = AdaBoostClassifier if clf else AdaBoostRegressor
        base = {}
        if "base_depth" in p:
            base["estimator"] = (DecisionTreeClassifier if clf else DecisionTreeRegressor)(max_depth=int(p["base_depth"]), random_state=SEED)
        return cls(n_estimators=int(p.get("n_estimators", 50)), learning_rate=float(p.get("learning_rate", 1.0)), random_state=SEED, **base)
    if key == "xgb":
        if clf:
            return _XGBClf(n_estimators=int(p.get("n_estimators", 100)), learning_rate=float(p.get("learning_rate", 0.1)),
                           max_depth=int(p.get("max_depth", 6)))
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=int(p.get("n_estimators", 100)), learning_rate=float(p.get("learning_rate", 0.1)),
                            max_depth=int(p.get("max_depth", 6)), random_state=SEED, n_jobs=-1, verbosity=0)
    if key == "cat":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if clf else CatBoostRegressor
        return cls(iterations=int(p.get("iterations", 200)), learning_rate=float(p.get("learning_rate", 0.1)),
                   depth=int(p.get("depth", 6)), random_seed=SEED, verbose=0, allow_writing_files=False)
    from lightgbm import LGBMClassifier, LGBMRegressor
    cls = LGBMClassifier if clf else LGBMRegressor
    return cls(n_estimators=int(p.get("n_estimators", 100)), learning_rate=float(p.get("learning_rate", 0.1)),
               num_leaves=int(p.get("num_leaves", 31)), random_state=SEED, verbose=-1, n_jobs=-1)


def _pipeline(X, key, task, params):
    num = [c for c in X.columns if pp._is_num(X[c])]
    cat = [c for c in X.columns if c not in num]
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if MODELS[key]["scaled"]:
        num_steps.append(("sc", StandardScaler()))
    parts = []
    if num:
        parts.append(("num", Pipeline(num_steps), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return Pipeline([("pre", ColumnTransformer(parts)), ("m", _estimator(key, task, params))])


def _clean_param(spec, raw):
    """Valeur validée d'un hyperparamètre (retombe sur le défaut si invalide)."""
    default = spec["default"]
    if raw is None or raw == "":
        return default if not spec.get("nullable") else None
    try:
        if spec["type"] == "choice":
            return raw if raw in spec["choices"] else default
        v = float(raw) if spec["type"] == "float" else int(float(raw))
        return min(spec["max"], max(spec["min"], v))
    except (TypeError, ValueError):
        return default


def _clean_params(raw):
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for key, meta in MODELS.items():
        r = raw.get(key) if isinstance(raw.get(key), dict) else {}
        out[key] = {sp_["k"]: _clean_param(sp_, r.get(sp_["k"], sp_["default"])) for sp_ in meta["params"]}
    return out


def clean_config(raw):
    raw = raw if isinstance(raw, dict) else {}
    group = raw.get("group") if raw.get("group") in GROUPS else "individual"
    wanted = [m for m in (raw.get("models") or GROUPS[group]["models"]) if m in GROUPS[group]["models"]]
    models = [m for m in wanted if available(m)]
    if not models:
        missing = sorted({MODELS[m]["lib"] for m in wanted if MODELS[m]["lib"]})
        raise ValueError("Aucun modèle utilisable sélectionné" + (f" : installez {' '.join(missing)} (pip install {' '.join(missing)})." if missing else "."))
    return {
        "task": raw.get("task") if raw.get("task") in ("classification", "regression") else "auto",
        "group": group,
        "models": models,
        "skipped": [m for m in wanted if m not in models],
        "tune": bool(raw.get("tune", False)),
        "params": _clean_params(raw.get("params")),
    }


def _val_split(task, y, chronological):
    """(indices d'entraînement, indices de validation) : un seul découpage 80 / 20 DU TRAIN.
    Chronologique : les dernières lignes servent de validation. Classification : découpage stratifié si possible."""
    n = len(y)
    idx = np.arange(n)
    n_val = max(1, int(round(n * VAL_FRACTION)))
    if chronological:
        return idx[:n - n_val], idx[n - n_val:]
    if task == "classification":
        try:
            tr, va = train_test_split(idx, test_size=n_val, random_state=SEED, stratify=np.asarray(y))
            return np.sort(tr), np.sort(va)
        except ValueError:                                     # classe trop rare pour être stratifiée
            pass
    tr, va = train_test_split(idx, test_size=n_val, random_state=SEED)
    return np.sort(tr), np.sort(va)


def _auc(model, X, y, classes):
    """AUC ROC : binaire (score de la classe positive) ou multiclasse (un-contre-tous, pondéré). None si non calculable."""
    try:
        k = len(classes)
        if k == 2:
            score = model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]
            return roc_auc_score(y, score)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:                                                  # SVC sans probabilités : softmax des scores un-contre-tous
            s_ = np.asarray(model.decision_function(X), dtype=float)
            e = np.exp(s_ - s_.max(axis=1, keepdims=True))
            proba = e / e.sum(axis=1, keepdims=True)
        return roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=list(range(k)))
    except Exception:
        return None


def _classification_metrics(model, Xte, yte, pred, classes):
    labels = list(range(len(classes)))
    return {
        "f1": _num(f1_score(yte, pred, average="weighted", zero_division=0)),
        "roc_auc": _num(_auc(model, Xte, yte, classes)),
        "confusion": confusion_matrix(yte, pred, labels=labels).tolist(),
    }


def _regression_metrics(yte, pred):
    return {"r2": _num(r2_score(yte, pred))}


def _score(task, y_true, pred):
    """Score principal : F1 pondéré (classification) ou R² (régression)."""
    return f1_score(y_true, pred, average="weighted", zero_division=0) if task == "classification" else r2_score(y_true, pred)


def _holdout(key, task, X, y, params, chrono):
    """(score validation, score entraînement) : modèle entraîné sur 80 % du train, évalué sur les 20 % restants."""
    tr, va = _val_split(task, y, chrono)
    Xa, Xb = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True)
    m = _pipeline(Xa, key, task, params).fit(Xa, y[tr])
    return _score(task, y[va], m.predict(Xb)), _score(task, y[tr], m.predict(Xa))


def _sub(n, cap, y=None, seed=SEED):
    """Indices triés d'un échantillon de `cap` lignes (ordre conservé). Si `y` est donné (classification),
    l'échantillon est stratifié : chaque classe garde au moins min(effectif, 10) lignes."""
    if n <= cap:
        return None
    rng = np.random.default_rng(seed)
    if y is None:
        idx = rng.choice(n, size=cap, replace=False)
    else:
        y = np.asarray(y)
        frac = cap / n
        parts = []
        for c in np.unique(y):
            ids = np.flatnonzero(y == c)
            k = min(len(ids), max(10, int(round(len(ids) * frac))))
            parts.append(rng.choice(ids, size=k, replace=False))
        idx = np.concatenate(parts)
    idx.sort()
    return idx


def _take(X, y, idx):
    return (X, y) if idx is None else (X.iloc[idx].reset_index(drop=True), y[idx])


def _fit_one(key, task, Xtr, ytr, Xte, yte, cfg, chrono, classes):
    meta = MODELS[key]
    notes = []
    strat = ytr if task == "classification" and not chrono else None
    if key == "svm" and len(Xtr) > SVM_MAX_TRAIN:
        Xtr, ytr = _take(Xtr, ytr, _sub(len(Xtr), SVM_MAX_TRAIN, strat))
        notes.append(f"SVM entraîné sur un échantillon de {SVM_MAX_TRAIN:,} lignes (coût quadratique).".replace(",", " "))
    scoring = "f1_weighted" if task == "classification" else "r2"
    # les modèles déjà multi-threadés (RF, XGB, LGBM, CatBoost) ne parallélisent pas les candidats du GridSearch en plus
    jobs = 1 if key in ("rf", "xgb", "cat", "lgbm") else max(1, os.cpu_count() or 1)
    pipe = _pipeline(Xtr, key, task, cfg["params"][key])
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg["tune"]:
            Xg, yg = _take(Xtr, ytr, _sub(len(Xtr), TUNE_MAX_ROWS, strat))
            if len(Xg) < len(Xtr):
                notes.append(f"{meta['name']} : hyperparamètres optimisés sur un échantillon de {len(Xg):,} lignes, puis modèle ré-entraîné sur tout le train.".replace(",", " "))
            _, va_i = _val_split(task, yg, chrono)
            fold = np.full(len(yg), -1)
            fold[va_i] = 0                                      # -1 = toujours en entraînement, 0 = validation
            grid = {f"m__{k}": v for k, v in meta["grid"].items()}
            gs = GridSearchCV(pipe, grid, cv=PredefinedSplit(fold), scoring=scoring, n_jobs=jobs, refit=False)
            gs.fit(Xg, yg)
            val_score = gs.best_score_
            params = {**cfg["params"][key], **{k.replace("m__", ""): v for k, v in gs.best_params_.items()}}
            model = _pipeline(Xtr, key, task, params).fit(Xtr, ytr)
        else:
            Xc, yc = _take(Xtr, ytr, _sub(len(Xtr), VAL_MAX_ROWS, strat))
            if len(Xc) < len(Xtr):
                notes.append(f"{meta['name']} : validation calculée sur un échantillon de {len(Xc):,} lignes (modèle final entraîné sur tout le train).".replace(",", " "))
            val_score, _ = _holdout(key, task, Xc, yc, cfg["params"][key], chrono)
            model = pipe.fit(Xtr, ytr)
            params = dict(cfg["params"][key])
        pred = model.predict(Xte)
        Xe, ye = _take(Xtr, ytr, _sub(len(Xtr), TRAIN_EVAL_ROWS, strat))
        train_pred = model.predict(Xe)
    if task == "classification":
        test_m = _classification_metrics(model, Xte, yte, pred, classes)
        train_score = f1_score(ye, train_pred, average="weighted", zero_division=0)
        main = test_m["f1"]
    else:
        test_m = _regression_metrics(yte, pred)
        train_score = r2_score(ye, train_pred)
        main = test_m["r2"]
    res = {
        "id": key, "name": meta["name"], "params": params,
        "val_score": _num(val_score),
        "train_score": _num(train_score), "test_score": main,
        "test": test_m, "seconds": round(time.time() - t0, 2), "note": " ".join(n for n in notes if n.startswith("SVM")) or None,
        "overfit": bool(train_score - (main or 0) > 0.15),
    }
    res["sample_notes"] = [n for n in notes if not n.startswith("SVM")]
    return res, pred



# ---------------------------------------------------------------------------------------------
# Modèles individuels : comparaison des variantes -> meilleure variante -> réglage de l'hyperparamètre clé
# ---------------------------------------------------------------------------------------------
# modèles d'ensemble : la comparaison porte sur le nombre d'arbres / d'estimateurs / d'itérations
_N_EST = {"rf": ("n_estimators", "arbres", (50, 100, 200, 300)), "ada": ("n_estimators", "estimateurs", (25, 50, 100, 200)),
          "xgb": ("n_estimators", "arbres", (50, 100, 200, 400)), "cat": ("iterations", "itérations", (100, 200, 400)),
          "lgbm": ("n_estimators", "arbres", (50, 100, 200, 400))}


def _variants(key, task, n_train):
    """Variantes comparées pour un modèle individuel : [(id, libellé, params)]."""
    clf = task == "classification"
    if key == "svm":
        return [("rbf", "RBF (gaussien)", {"kernel": "rbf"}), ("linear", "Linéaire", {"kernel": "linear", "max_iter": 300000}),
                ("poly", "Polynomial (degré 3)", {"kernel": "poly", "degree": 3, "max_iter": 300000}),
                ("sigmoid", "Sigmoïde", {"kernel": "sigmoid", "max_iter": 300000})]
    if key == "knn":
        mets = [("euclidean", "Euclidienne"), ("manhattan", "Manhattan"), ("chebyshev", "Tchebychev")]
        return [(f"{m}-{w}", f"{ml} · {'poids = distance' if w == 'distance' else 'poids égaux'}", {"metric": m, "weights": w})
                for m, ml in mets for w in ("uniform", "distance")]
    if key in _N_EST:
        pn, unit, values = _N_EST[key]
        return [(f"n{n}", f"{n} {unit}", {pn: n}) for n in values]
    crits = [("gini", "Gini"), ("entropy", "Entropie"), ("log_loss", "Log-loss")] if clf else \
        [("squared_error", "Erreur quadratique"), ("friedman_mse", "Friedman MSE")] + ([("absolute_error", "Erreur absolue")] if n_train <= 5000 else [])
    return [(f"{c}-{sp_}", f"{cl} · {'splits aléatoires' if sp_ == 'random' else 'meilleur split'}", {"criterion": c, "splitter": sp_})
            for c, cl in crits for sp_ in ("best", "random")]


# hyperparamètre clé réglé après le choix de la variante : (nom du paramètre, libellé, valeurs, axe log ?)
_SWEEP = {
    "svm": ("C", "C (régularisation)", [0.01, 0.1, 1, 10, 100], True),
    "knn": ("n_neighbors", "k (nombre de voisins)", [1, 3, 5, 7, 9, 15, 21, 31, 51], False),
    "tree": ("max_depth", "Profondeur maximale", [1, 2, 3, 4, 5, 7, 10, 15, 20, None], False),
    "rf": ("max_features", "Variables par split", ["sqrt", "log2", "all"], False),
    "ada": ("learning_rate", "Learning rate", [0.01, 0.1, 0.5, 1.0], True),
    "xgb": ("max_depth", "Profondeur max", [3, 5, 7, 9], False),
    "cat": ("depth", "Profondeur", [4, 6, 8], False),
    "lgbm": ("num_leaves", "Nb de feuilles", [15, 31, 63, 127], False),
}


def _slim(r, label, vid):
    """Résultat d'une variante, sans matrice de confusion (poids du JSON)."""
    t = {k: v for k, v in r["test"].items() if k != "confusion"}
    return {"id": vid, "label": label, "params": r["params"], "val_score": r["val_score"],
            "train_score": r["train_score"], "test_score": r["test_score"], "test": t, "seconds": r["seconds"], "overfit": r["overfit"]}


def _sweep(key, task, Xtr, ytr, params, chrono):
    """Courbe score (train / validation) selon l'hyperparamètre clé, pour la variante gagnante."""
    pname, plabel, values, logx = _SWEEP[key]
    strat = ytr if task == "classification" and not chrono else None
    if key == "svm" and len(Xtr) > SVM_MAX_TRAIN:
        Xtr, ytr = _take(Xtr, ytr, _sub(len(Xtr), SVM_MAX_TRAIN, strat))
        strat = ytr if strat is not None else None
    Xc, yc = _take(Xtr, ytr, _sub(len(Xtr), VAL_MAX_ROWS, strat))
    if key == "knn":
        values = [v for v in values if v < max(2, int(len(Xc) * (1 - VAL_FRACTION)))]
    pts = []
    for v in values:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vm, tm = _holdout(key, task, Xc, yc, {**params, pname: v}, chrono)
            if not (math.isfinite(vm) and math.isfinite(tm)):
                continue
            pts.append({"value": v, "val": _num(vm), "train": _num(tm)})
        except Exception:
            continue
    best = max(pts, key=lambda q: q["val"]) if pts else None
    return {"param": pname, "label": plabel, "log": logx, "points": pts, "best": best["value"] if best else params.get(pname),
            "sample_rows": int(len(Xc))}


def _study(key, task, Xtr, ytr, Xte, yte, cfg, chrono, classes, tick):
    """Modèle individuel complet : variantes -> gagnante (validation) -> courbe de l'hyperparamètre clé -> modèle final."""
    base = dict(cfg["params"][key])
    cfg_nt = {**cfg, "tune": False}
    variants, win, win_pred = [], None, None
    for vid, label, vp in _variants(key, task, len(Xtr)):
        tick(f"{MODELS[key]['name']} · {label}")
        c2 = {**cfg_nt, "params": {**cfg["params"], key: {**base, **vp}}}
        try:
            r, pred = _fit_one(key, task, Xtr, ytr, Xte, yte, c2, chrono, classes)
        except Exception as e:                                       # une variante qui échoue ne bloque pas les autres
            variants.append({"id": vid, "label": label, "params": {**base, **vp}, "error": f"{type(e).__name__}: {e}"[:160]})
            continue
        variants.append(_slim(r, label, vid))
        if win is None or (r["val_score"] or -1e9) > (win[0]["val_score"] or -1e9):
            win, win_pred = (r, label, vid, vp), pred
    if win is None:
        raise ValueError(f"{MODELS[key]['name']} : aucune variante n'a pu être entraînée.")
    r_win, win_label, win_id, win_vp = win
    wparams = {**base, **win_vp}

    tick(f"{MODELS[key]['name']} · réglage de {_SWEEP[key][0]}")
    sw = _sweep(key, task, Xtr, ytr, wparams, chrono)
    final_params = {**wparams, sw["param"]: sw["best"]}
    if final_params == r_win["params"]:
        final, pred = r_win, win_pred
    else:
        tick(f"{MODELS[key]['name']} · modèle final")
        c3 = {**cfg_nt, "params": {**cfg["params"], key: final_params}}
        final, pred = _fit_one(key, task, Xtr, ytr, Xte, yte, c3, chrono, classes)
        if (final["val_score"] or -1e9) < (r_win["val_score"] or -1e9):  # le réglage ne doit jamais dégrader la variante gagnante
            final, pred, sw["best"] = r_win, win_pred, wparams.get(sw["param"])
    final["variant"] = win_label
    final["name"] = MODELS[key]["name"]
    final["label"] = f"{MODELS[key]['name']} · {win_label}"
    sw["chosen"] = sw["best"]
    return final, pred, {"id": key, "name": MODELS[key]["name"], "variants": variants, "best_variant": win_id, "best_label": win_label,
                         "sweep": sw, "final_params": final["params"]}


_OVERRIDDEN = {"svm": ["C", "kernel"], "knn": ["n_neighbors", "weights"], "tree": ["max_depth"], "rf": ["n_estimators", "max_features"], "ada": ["n_estimators", "learning_rate"],
               "xgb": ["n_estimators", "max_depth"], "cat": ["iterations", "depth"], "lgbm": ["n_estimators", "num_leaves"]}
_SHORT = {"svm": "4 noyaux", "knn": "3 distances × 2 poids", "tree": "critères × découpe", "rf": "4 nombres d'arbres", "ada": "4 nombres d'estimateurs",
          "xgb": "4 nombres d'arbres", "cat": "3 nombres d'itérations", "lgbm": "4 nombres d'arbres"}


def _study_info(key, task):
    """Ce que fait la comparaison pour un modèle (une ligne affichée avant l'entraînement)."""
    return {"short": f"{_SHORT[key]} → réglage : {_SWEEP[key][1]}", "overridden": _OVERRIDDEN[key]}


def _study_total(cfg, task, n_train):
    return sum(len(_variants(k, task, n_train)) + 2 for k in cfg["models"])

# ---------------------------------------------------------------------------------------------
# API du module
# ---------------------------------------------------------------------------------------------
def _task_ok(y_raw, task):
    try:
        _task_of(y_raw, task)
        return True
    except ValueError:
        return False


def _common(d):
    n_test = int(d["test"].sum())
    return {
        "target": d["target"], "n_rows": int(len(d["X"])), "n_train": int(len(d["X"]) - n_test), "n_test": n_test,
        "n_features": int(d["X"].shape[1]), "source": "preprocessed" if d["is_pre"] else "raw", "split_mode": d["mode"],
    }


def _target_info(y, ok):
    """Pourquoi la cible mène à la classification ou à la régression (affiché à l'utilisateur)."""
    n = int(y.nunique(dropna=True))
    numeric = bool(pp._is_num(y))
    integer = False
    if numeric:
        v = y.dropna().to_numpy(dtype=float)
        integer = bool(len(v) and np.all(np.mod(v, 1) == 0))
    if not numeric:
        kind = "texte / catégorielle"
    elif integer:
        kind = "numérique (entiers)"
    else:
        kind = "numérique continue (décimaux)"
    info = {"name": str(y.name), "kind": kind, "n_unique": n, "n_missing": int(y.isna().sum()), "numeric": numeric}
    if numeric:
        info["stats"] = {"min": _num(y.min(), 3), "max": _num(y.max(), 3), "mean": _num(y.mean(), 3)}
    vc = y.astype(str).value_counts()
    info["top_values"] = [{"label": str(k), "n": int(v)} for k, v in vc.head(6).items()]

    reasons = {}
    if ok["classification"]:
        if not numeric:
            reasons["classification"] = f"La cible est du texte avec {n} valeurs distinctes : chaque valeur est une classe à prédire."
        else:
            reasons["classification"] = f"La cible ne prend que {n} valeurs distinctes ({'entières' if integer else 'numériques'}) : on peut les traiter comme des classes."
    elif n < 2:
        reasons["classification"] = "La cible n'a qu'une seule valeur : il n'y a rien à classer."
    else:
        reasons["classification"] = f"La cible a {n} valeurs distinctes (> {MAX_CLASSES}) : trop pour des classes, c'est une variable continue."
    if ok["regression"]:
        if integer and 2 <= n <= 20:
            reasons["regression"] = "La cible est numérique, la régression reste possible, mais avec si peu de valeurs entières la classification est plus naturelle."
        else:
            reasons["regression"] = f"La cible est numérique avec {n} valeurs distinctes : on prédit une quantité continue."
    else:
        reasons["regression"] = "La cible est du texte : on ne peut pas prédire une valeur numérique."
    info["reasons"] = reasons
    return info


def get_overview(dataset_name):
    """Page d'accueil de l'étape : choix Régression / Classification."""
    d = _dataset(dataset_name)
    y = d["y_raw"]
    ok = {t: _task_ok(y, t) for t in ("classification", "regression")}
    rec = "classification" if pp._is_classification(y) or not pp._is_num(y) else "regression"
    return {**_common(d), "task_ok": ok, "recommended": rec if ok[rec] else next((t for t in ok if ok[t]), None),
            "n_unique": int(y.nunique(dropna=True)), "numeric": bool(pp._is_num(y)),
            "target_info": _target_info(y, ok),
            "saved": saved_overview(dataset_name)}


def get_task_profile(dataset_name, task):
    """Page d'un type de problème (régression ou classification) : modèles par famille + derniers résultats."""
    if task not in ("classification", "regression"):
        raise ValueError("Type de problème inconnu.")
    d = _dataset(dataset_name)
    y_raw = d["y_raw"]
    _task_of(y_raw, task)                                   # lève ValueError si la cible ne convient pas
    classes = None
    if task == "classification":
        vc = y_raw.astype(str).value_counts()
        classes = [{"label": str(k), "n": int(v)} for k, v in vc.head(12).items()]
    notes = []
    if not d["is_pre"]:
        notes.append("Le dataset n'est pas encore prétraité : les valeurs manquantes sont imputées (médiane) et les textes encodés (one-hot) dans le pipeline du modèle, mais le prétraitement (étape 02) donnera de meilleurs résultats.")
    if d["ignored"]:
        notes.append(f"Colonnes texte ignorées (plus de {MAX_CAT_UNIQUE} modalités) : " + ", ".join(d["ignored"][:6]) + ("…" if len(d["ignored"]) > 6 else "") + ".")
    if d["chronological"]:
        notes.append("Découpage chronologique : la validation utilise les dernières lignes du train (l'ordre du temps est respecté).")
    groups = {}
    for gid, g in GROUPS.items():
        groups[gid] = {
            "title": g["title"], "sub": g["sub"], "saved": get_results(dataset_name, task, gid),
            "models": [{"id": m, "name": MODELS[m]["name"], "sk": MODELS[m]["names"][task], "desc": MODELS[m]["desc"],
                        "params": MODELS[m]["params"], "available": available(m), "lib": MODELS[m]["lib"],
                        "pip": install_hint(m), "error": _IMPORT_ERR.get(MODELS[m]["lib"]),
                        "study": _study_info(m, task)}
                       for m in g["models"]],
        }
    return {**_common(d), "task": task, "classes": classes, "notes": notes, "groups": groups,
            "svm_max_train": SVM_MAX_TRAIN, "python": sys.executable}


def _input_signature(dataset_name):
    """Données d'entrée des modèles : le prétraitement enregistré (ou les données brutes) + le split."""
    rep = pp.load_report(dataset_name)
    return {"data": ("preprocessed", rep.get("created_at")) if rep else ("raw", None)}


def run(dataset_name, raw_cfg, save=True, progress=None, refresh=False):
    """Entraîne et compare les modèles d'une famille. Mêmes réglages + mêmes données = résultat relu dans SQLite."""
    cfg = clean_config(raw_cfg)
    params = {"cfg": cfg, "v": 4, **_input_signature(dataset_name)}
    if not refresh:
        hit = step_cache.get(dataset_name, "model_run", params)
        if hit is not None:
            if progress:
                n = len(cfg["models"])
                progress(n, n, "")
            if save:                                           # the saved view must show this very result
                cur = get_results(dataset_name, hit["task"], hit["group"])
                if not cur or cur.get("created_at") != hit.get("created_at"):
                    _save(dataset_name, hit["task"], hit["group"], hit, hit.get("created_at"))
            return hit
    out = _run_compute(dataset_name, cfg, save, progress)
    step_cache.put(dataset_name, "model_run", params, out)
    return out


def _split_xy(d, task):
    """(Xtr, Xte, ytr, yte, classes) pour une tâche donnée, à partir de `_dataset`."""
    test = d["test"]
    X, y_raw = d["X"], d["y_raw"]

    classes = None
    if task == "classification":
        y_enc, mapping = pp._label_encode(y_raw.reset_index(drop=True))
        names = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
        y_all = y_enc.to_numpy()
        present = np.unique(y_all[~test])                  # classes vues à l'entraînement (0..k-1 requis par XGBoost)
        if len(present) < 2:
            raise ValueError("Le jeu d'entraînement ne contient qu'une seule classe.")
        remap = {int(c): i for i, c in enumerate(present)}
        keep = np.isin(y_all, present)                     # lignes de test d'une classe jamais vue : retirées
        y = np.array([remap[int(v)] for v in y_all[keep]])
        X, test = X[keep].reset_index(drop=True), test[keep]
        classes = [names[int(c)] for c in present]
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        if not ok.all():
            X, y, test = X[ok].reset_index(drop=True), y[ok], test[ok]

    Xtr, Xte = X[~test].reset_index(drop=True), X[test].reset_index(drop=True)
    return Xtr, Xte, y[~test], y[test], classes


def _run_compute(dataset_name, cfg, save=True, progress=None):
    d = _dataset(dataset_name)
    task = _task_of(d["y_raw"], cfg["task"])
    X = d["X"]
    Xtr, Xte, ytr, yte, classes = _split_xy(d, task)
    results, preds, learners = [], {}, {}
    total = _study_total(cfg, task, len(Xtr))
    state = {"i": 0}

    def tick(name):
        if progress:
            progress(state["i"], total, name)
        state["i"] += 1
    for key in cfg["models"]:
        r, pred, lr = _study(key, task, Xtr, ytr, Xte, yte, cfg, d["chronological"], classes, tick)
        results.append(r)
        preds[key] = pred
        learners[key] = lr
    if progress:
        progress(state["i"], state["i"], "")
    results.sort(key=lambda r: (r["val_score"] is None, -(r["val_score"] or 0)))
    best = results[0]

    out = {
        "task": task, "group": cfg["group"], "target": d["target"],
        "metric": "F1 pondéré" if task == "classification" else "R²",
        "config": {k: cfg[k] for k in ("models", "tune")}, "skipped": cfg["skipped"],
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "n_features": int(X.shape[1]),
        "source": "preprocessed" if d["is_pre"] else "raw", "chronological": d["chronological"],
        "classes": classes, "results": results, "best": best["id"], "schema": 2,
    }
    if learners:
        out["learners"] = learners
    if task == "regression":
        p = preds[best["id"]]
        idx = np.linspace(0, len(yte) - 1, min(MAX_POINTS, len(yte))).astype(int)
        out["scatter"] = {"model": best["name"], "actual": [_num(v, 3) for v in yte[idx]],
                          "pred": [_num(v, 3) for v in p[idx]]}
    if save:
        out["created_at"] = _save(dataset_name, task, cfg["group"], out)
    return out
