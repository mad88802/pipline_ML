
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

import analysis_engine

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TIMEOUT = 45


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def _load_dotenv():
    """Mini-lecteur de .env (évite d'ajouter python-dotenv)."""
    env = Path(__file__).with_name(".env")
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class AIError(Exception):
    """Erreur affichable à l'utilisateur."""

    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


# ----------------------------------------------------------------------
# Contexte : stats exactes du graphique
# ----------------------------------------------------------------------
def _df_records(df, limit=25):
    if df is None or df.empty:
        return []
    out = df.head(limit)
    if not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    return json.loads(out.to_json(orient="records", double_precision=4))


def _column_label(dataset, feature):
    summary = analysis_engine.load_summary(dataset) or {}
    for c in (summary.get("dtypes") or {}):
        if c == feature or analysis_engine.safe_name(c) == feature:
            return c
    return feature


def build_context(dataset, kind, feature="", target="", fi_provider=None):
    """Retourne (titre_du_graphique, dict de stats) pour un type de graphique."""
    data = analysis_engine.load_dataset_results(dataset)
    if not data:
        raise AIError("Dataset introuvable", 404)
    summary, tables = data["summary"], data["tables"]
    base = {
        "dataset": dataset,
        "n_lignes": summary.get("n_rows"),
        "n_colonnes": summary.get("n_columns"),
        "type_dataset": summary.get("dataset_type"),
    }

    try:
        if kind == "distribution":
            s = analysis_engine.get_distribution_series(dataset, feature)
            ctx = {k: s[k] for k in ("column", "n", "n_unique", "mean", "median", "std", "skew", "kurtosis", "min", "max")}
            ctx["histogramme"] = {"centres_bins": s["hist_x"], "effectifs": s["hist_y"]}
            return f"Distribution (histogramme + KDE) de « {s['column']} »", {**base, **ctx}

        if kind == "outliers":
            s = analysis_engine.get_isolation_forest(dataset)
            keep = ("n", "n_features", "n_outliers", "outlier_pct", "contamination", "threshold")
            ctx = {k: s.get(k) for k in keep}
            ctx["variables_utilisees"] = s.get("features")
            ctx["variables_les_plus_extremes"] = s.get("drivers")
            ctx["lignes_les_plus_anormales"] = [
                {"ligne": r["row"], "score": r["score"], "pourquoi": r["why"]} for r in (s.get("table") or [])[:8]]
            return "Outliers globaux (Isolation Forest, toutes variables numériques)", {**base, **ctx}

        if kind == "noise":
            s = analysis_engine.get_noise_series(dataset, feature)
            keep = ("column", "time_kind", "method", "window", "n_total", "sigma", "bound_lo", "bound_hi",
                    "outside", "outside_pct", "min", "max")
            ctx = {k: s.get(k) for k in keep}
            sm = [v for v in (s.get("smooth") or []) if v is not None]
            raw = [v for v in (s.get("raw") or []) if v is not None]
            if sm:
                n = len(sm)
                ctx["tendance_lissee"] = {"debut": round(sum(sm[: max(1, n // 20)]) / max(1, n // 20), 3),
                                          "fin": round(sum(sm[-max(1, n // 20):]) / max(1, n // 20), 3),
                                          "min": min(sm), "max": max(sm)}
            if raw:
                ctx["signal_brut"] = {"min": min(raw), "max": max(raw), "moyenne": round(sum(raw) / len(raw), 3)}
            return f"Signal brut, tendance lissée et bruit de « {s['column']} »", {**base, **ctx}

        if kind == "correlation":
            s = analysis_engine.get_correlation_matrix(dataset)
            return "Matrice de corrélation de Pearson", {**base, "variables": s["labels"], "paires_les_plus_correlees": s["pairs"][:20]}

        if kind == "target":
            s = analysis_engine.get_target_balance(dataset)
            return f"Équilibre des classes de la cible « {s['target']} »", {**base, **{k: s[k] for k in ("target", "labels", "counts", "percents")}}

        if kind == "missing":
            rep = tables.get("missing_report")
            if rep is None:
                raise AIError("Pas de rapport de valeurs manquantes", 404)
            return "Valeurs manquantes par colonne", {**base, "n_cellules_manquantes": summary.get("n_missing_cells"),
                                                      "pct_cellules_manquantes": summary.get("missing_cell_percent"),
                                                      "colonnes": _df_records(rep, 40)}

        if kind == "reference_noise":
            t = tables.get("reference_noise")
            if t is None:
                raise AIError("Pas d'analyse de bruit de référence", 404)
            num = t.select_dtypes("number")
            return "Analyse du bruit de référence", {**base, "colonnes": list(t.columns)[:20],
                                                     "stats": json.loads(num.describe().round(4).to_json()) if not num.empty else {},
                                                     "extrait": _df_records(t, 8)}

        if kind in ("fi_anova", "fi_corr"):
            if fi_provider is None:
                raise AIError("Importance des variables indisponible", 404)
            payload = fi_provider(kind, target)
            title = f"Importance des variables par ANOVA (test F) vs « {target} »" if kind == "fi_anova" else f"Corrélation des variables avec la cible « {target} »"
            return title, {**base, **payload}
    except AIError:
        raise
    except (KeyError, FileNotFoundError) as e:
        raise AIError(str(e).strip("'\""), 404)

    raise AIError(f"Type de graphique inconnu : {kind}", 400)


# ----------------------------------------------------------------------
# Groq
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Tu es un data scientist senior qui accompagne un étudiant dans l'analyse exploratoire (EDA) d'un projet de machine learning. "
    "On te donne les statistiques exactes qui ont servi à tracer UN graphique. Interprète ce graphique en français, "
    "uniquement à partir de ces chiffres : n'invente aucune valeur, et dis-le si une information manque. "
    "Réponds en Markdown simple (pas de titres #), en 120 mots maximum, avec exactement ces trois blocs :\n"
    "**Ce que montre le graphique** : 1-2 phrases.\n"
    "**Points clés** : 2 à 4 puces chiffrées (asymétrie, outliers, tendance, déséquilibre, corrélations fortes...).\n"
    "**Impact pour le modèle** : 1-2 phrases d'action concrète (transformation, nettoyage, choix de variable...)."
)


def _api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise AIError("Clé GROQ_API_KEY manquante : ajoute-la dans un fichier .env à côté de app.py "
                      "(GROQ_API_KEY=gsk_...) puis redémarre Flask.", 503)
    return key


def call_groq(title, context, model=None):
    model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    user = f"Graphique : {title}\n\nStatistiques (JSON) :\n{json.dumps(context, ensure_ascii=False, default=str)[:12000]}"
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.3, "max_tokens": 500,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]},
            timeout=TIMEOUT,
        )
    except requests.Timeout:
        raise AIError("Groq a mis trop de temps à répondre, réessaie.", 504)
    except requests.RequestException as e:
        raise AIError(f"Impossible de joindre Groq : {e}", 502)

    if r.status_code == 401:
        raise AIError("Clé Groq refusée (401). Vérifie GROQ_API_KEY.", 401)
    if r.status_code == 429:
        raise AIError("Limite Groq atteinte (429). Patiente quelques secondes.", 429)
    if not r.ok:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:200])
        except ValueError:
            msg = r.text[:200]
        raise AIError(f"Erreur Groq {r.status_code} : {msg}", 502)

    text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    if not text:
        raise AIError("Réponse vide de Groq.", 502)
    return text, model


# ----------------------------------------------------------------------
# Interprétation à la demande (rien n'est stocké : chaque clic relance l'IA)
# ----------------------------------------------------------------------
def interpret(dataset, kind, feature="", target="", fi_provider=None):
    dataset = analysis_engine.clean_dataset_name(dataset)
    title, ctx = build_context(dataset, kind, feature, target, fi_provider)
    text, model = call_groq(title, ctx)
    return {"title": title, "text": text, "model": model, "ts": int(time.time()), "cached": False}
