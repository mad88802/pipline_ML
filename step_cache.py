"""
Cache SQLite commun à toutes les étapes du pipeline (prétraitement, split, modèles).

Principe (le même que pour l'EDA) : le résultat d'un calcul est rangé dans results/analysis.db avec une
clé qui décrit TOUT ce dont il dépend. Quand on relance avec les mêmes paramètres sur les mêmes données,
le résultat est relu dans la base au lieu d'être recalculé. Dès qu'une entrée change (fichier source,
cible / split, réglages de l'étape, version du code), la clé change et le calcul est refait.

    payload = step_cache.get_or_compute(dataset, "pp_step:outliers", params, lambda: calcul(...))

Table : step_cache (dataset, kind, cache_key, payload JSON, created_at)
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import analysis_engine as ae

# À incrémenter quand l'algorithme d'une étape change : tous les anciens résultats deviennent inutilisés.
CODE_VERSION = 2
MAX_ENTRIES_PER_KIND = 40          # anciens résultats d'un même type : les plus vieux sont supprimés

_SCHEMA = """
CREATE TABLE IF NOT EXISTS step_cache (
    dataset    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    cache_key  TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset, kind, cache_key)
);
"""

_FILE_HASH_MEMO = {}               # (chemin, mtime, taille) -> sha256 : on ne relit pas un gros CSV à chaque requête


def _connect():
    conn = ae._connect(ae.RESULTS_DIR)
    conn.executescript(_SCHEMA)
    return conn


def _file_hash(path):
    p = Path(path)
    st = p.stat()
    sig = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if sig not in _FILE_HASH_MEMO:
        if len(_FILE_HASH_MEMO) > 64:
            _FILE_HASH_MEMO.clear()
        _FILE_HASH_MEMO[sig] = ae.calculate_file_hash(p)
    return _FILE_HASH_MEMO[sig]


def _jsonable(o):
    """json.dumps(default=...) : types numpy / pandas -> types Python."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def source_signature(dataset_name):
    """Empreinte des données d'entrée : contenu du fichier source + réglages de l'EDA qui pilotent le pipeline."""
    name = ae.clean_dataset_name(dataset_name)
    summary = ae.load_summary(name) or {}
    src = summary.get("source_file")
    file_hash = None
    try:
        if src and Path(src).exists():
            file_hash = _file_hash(src)
    except OSError:
        file_hash = None
    eda = {k: summary.get(k) for k in ("dataset_type", "date_column", "target_class", "source_file")}
    return {"file": file_hash, "eda": eda}


def _split_signature(dataset_name):
    try:
        import split as _split                     # import tardif : split importe preprocessing
        cfg = _split.get_config(dataset_name) or {}
        return {k: cfg.get(k) for k in ("test_size", "target", "features", "excluded", "strategy")}
    except Exception:
        return None


def make_key(dataset_name, kind, params, with_split=True):
    payload = {
        "v": CODE_VERSION,
        "kind": kind,
        "source": source_signature(dataset_name),
        "split": _split_signature(dataset_name) if with_split else None,
        "params": params,
    }
    raw = json.dumps(payload, sort_keys=True, default=_jsonable, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get(dataset_name, kind, params, with_split=True):
    """Résultat déjà calculé pour ces paramètres, ou None."""
    name = ae.clean_dataset_name(dataset_name)
    key = make_key(name, kind, params, with_split)
    try:
        conn = _connect()
        try:
            row = conn.execute("SELECT payload FROM step_cache WHERE dataset=? AND kind=? AND cache_key=?",
                               (name, kind, key)).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None                                # un cache illisible ne doit jamais casser l'étape


def put(dataset_name, kind, params, result, with_split=True):
    name = ae.clean_dataset_name(dataset_name)
    key = make_key(name, kind, params, with_split)
    try:
        text = json.dumps(result, default=_jsonable, ensure_ascii=False)
        conn = _connect()
        try:
            conn.execute("INSERT OR REPLACE INTO step_cache (dataset, kind, cache_key, payload, created_at) VALUES (?,?,?,?,?)",
                         (name, kind, key, text, datetime.now().isoformat(timespec="seconds")))
            conn.execute(
                "DELETE FROM step_cache WHERE dataset=? AND kind=? AND cache_key NOT IN "
                "(SELECT cache_key FROM step_cache WHERE dataset=? AND kind=? ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                (name, kind, name, kind, MAX_ENTRIES_PER_KIND))
        finally:
            conn.close()
    except Exception:
        pass                                       # échec d'écriture du cache : le résultat est quand même renvoyé


def get_or_compute(dataset_name, kind, params, compute, refresh=False, with_split=True):
    """Relit le résultat dans la base s'il existe (sauf refresh=True), sinon calcule puis l'enregistre."""
    if not refresh:
        hit = get(dataset_name, kind, params, with_split)
        if hit is not None:
            return hit
    result = compute()
    put(dataset_name, kind, params, result, with_split)
    return result


def invalidate(dataset_name, kind_prefix=None):
    """Supprime les résultats en cache d'un dataset (tous, ou ceux dont le type commence par kind_prefix)."""
    name = ae.clean_dataset_name(dataset_name)
    try:
        conn = _connect()
        try:
            if kind_prefix:
                conn.execute("DELETE FROM step_cache WHERE dataset=? AND kind LIKE ?", (name, kind_prefix + "%"))
            else:
                conn.execute("DELETE FROM step_cache WHERE dataset=?", (name,))
        finally:
            conn.close()
    except Exception:
        pass
