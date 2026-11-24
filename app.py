import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import io
import contextlib
import hashlib
import json
import math
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, Response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import pandas as pd
import numpy as np

import analysis_engine
import covers
import ai_interpret
import agent
import preprocessing
import step_cache
import split
import model_selection
import scenario_report

_REQUIRED = ("AUTO", "SUPPORTED_EXTENSIONS", "clean_dataset_name", "run_full_analysis",
             "load_summary", "export_table_csv", "migrate_legacy_results", "get_noise_series", "get_outlier_series", "get_isolation_forest", "get_distribution_series", "get_anova_importance", "get_anova_targets")
_missing = [n for n in _REQUIRED if not hasattr(analysis_engine, n)]
if _missing:
    raise RuntimeError(
        f"{analysis_engine.__file__} is an OLD version of analysis_engine.py "
        f"(missing: {', '.join(_missing)}). Replace it with the new file and restart."
    )

app = Flask(__name__, static_folder="static", template_folder="templates")
app.json.sort_keys = False
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Import datasets analysed by the old CSV version into results/analysis.db (skips what is already there)
for _name in analysis_engine.migrate_legacy_results():
    print(f"Migrated legacy results of '{_name}' to SQLite")


def format_dataframe(df):
    """Helper to convert a pandas DataFrame into safe, clean records and columns for templates."""
    if df is None:
        return {"columns": [], "rows": []}

    df_clean = df.copy()

    # If index has name or meaningful values, reset index
    if df_clean.index.name or not isinstance(df_clean.index, pd.RangeIndex):
        df_clean = df_clean.reset_index()
        if "index" in df_clean.columns:
            df_clean = df_clean.rename(columns={"index": "Attribute"})
        elif "Unnamed: 0" in df_clean.columns:
            df_clean = df_clean.rename(columns={"Unnamed: 0": "Attribute"})

    # Round numeric values to 3 decimals for display
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].round(3)

    df_clean = df_clean.astype(object).where(df_clean.notna(), "—")

    columns = list(df_clean.columns)
    rows = df_clean.to_dict(orient="records")
    return {"columns": columns, "rows": rows}


def compute_feature_importance(summary, raw_tables):
    """
    Computes feature importance strictly from Variance and Correlation metrics,
    explaining how each score is mathematically obtained without mentioning ML preparation.
    """
    numeric_cols = summary.get("numeric_columns", [])
    if not numeric_cols:
        return []

    # 1. Variance Calculation from describe_numeric
    var_dict = {}
    std_dict = {}
    desc_df = raw_tables.get("describe_numeric")
    if desc_df is not None:
        for col in numeric_cols:
            std_val = 0.0
            if col in desc_df.index and "std" in desc_df.columns:
                try:
                    std_val = float(desc_df.loc[col, "std"])
                except Exception:
                    std_val = 0.0
            elif "Attribute" in desc_df.columns and col in desc_df["Attribute"].values:
                try:
                    row = desc_df[desc_df["Attribute"] == col].iloc[0]
                    std_val = float(row.get("std", 0.0))
                except Exception:
                    std_val = 0.0
            
            if np.isnan(std_val):
                std_val = 0.0
            std_dict[col] = std_val
            var_dict[col] = std_val ** 2
    else:
        for col in numeric_cols:
            std_dict[col] = 1.0
            var_dict[col] = 1.0

    # Max variance for normalization (avoid div by 0)
    max_var = max(var_dict.values()) if var_dict else 1.0
    min_var = min(var_dict.values()) if var_dict else 0.0
    var_range = max_var - min_var if (max_var - min_var) > 1e-9 else 1.0

    # 2. Correlation Calculation from correlation matrix
    corr_df = raw_tables.get("correlation")
    target_col = summary.get("target_class")
    corr_dict = {}

    for col in numeric_cols:
        corr_val = 0.0
        if corr_df is not None and col in corr_df.index:
            # Check if target is numeric and in the correlation matrix
            if target_col and target_col in corr_df.columns and target_col != col:
                val = corr_df.loc[col, target_col]
                corr_val = abs(float(val)) if pd.notna(val) else 0.0
            else:
                # Average absolute correlation with all other numeric features
                other_cols = [c for c in corr_df.columns if c != col and c in numeric_cols]
                if other_cols:
                    vals = [abs(float(corr_df.loc[col, c])) for c in other_cols if pd.notna(corr_df.loc[col, c])]
                    corr_val = float(np.mean(vals)) if vals else 0.0
        corr_dict[col] = corr_val

    max_corr = max(corr_dict.values()) if corr_dict else 1.0
    min_corr = min(corr_dict.values()) if corr_dict else 0.0
    corr_range = max_corr - min_corr if (max_corr - min_corr) > 1e-9 else 1.0

    # 3. Combine into normalized composite score
    results = []
    for col in numeric_cols:
        v = var_dict.get(col, 0.0)
        s = std_dict.get(col, 0.0)
        c = corr_dict.get(col, 0.0)

        # Variance normalized: features with 0 variance get 0
        if v == 0.0:
            var_norm = 0.0
        else:
            var_norm = (v - min_var) / var_range

        # Correlation normalized
        corr_norm = (c - min_corr) / corr_range if corr_range > 0 else c

        # Composite: 50% Variance + 50% Correlation
        importance_score = (0.5 * var_norm + 0.5 * corr_norm) * 100.0

        if v == 0.0:
            assessment = "Variable constante (Variance = 0) : aucune dispersion ni information portée."
        elif importance_score >= 70:
            assessment = "Importance Élevée : forte variance informationnelle et fortes corrélations structurelles."
        elif importance_score >= 40:
            assessment = "Importance Modérée : dispersion d'information stable et corrélations moyennes."
        else:
            assessment = "Importance Faible : faible variabilité relative ou liaisons linéaires réduites."

        results.append({
            "name": col,
            "std": round(s, 3),
            "variance": round(v, 3),
            "var_score_pct": round(var_norm * 100.0, 1),
            "corr_value": round(c, 3),
            "corr_score_pct": round(corr_norm * 100.0, 1),
            "importance_pct": round(importance_score, 1),
            "assessment": assessment,
        })

    # Sort descending by importance score
    results.sort(key=lambda x: x["importance_pct"], reverse=True)
    for idx, item in enumerate(results, 1):
        item["rank"] = idx

    return results


@app.errorhandler(Exception)
def handle_error(e):
    """/api/ routes always answer in JSON (the page's JavaScript expects it), never an HTML error page."""
    if request.path.startswith("/api/"):
        code = e.code if isinstance(e, HTTPException) else 500
        if code >= 500:
            app.logger.exception("Error on %s", request.path)  # full traceback in the terminal
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), code
    if isinstance(e, HTTPException):
        return e
    raise e



@app.route("/api/status")
def api_status():
    """Open http://127.0.0.1:5000/api/status to check which files the server is really running."""
    return jsonify({
        "app_file": str(Path(__file__).resolve()),
        "engine_file": str(Path(analysis_engine.__file__).resolve()),
        "datasets": analysis_engine.list_datasets(),
        "routes": sorted(r.rule for r in app.url_map.iter_rules()),
    })


# ---------------------------------------------------------------------------------------------
# DATASET PICTURES
# Every dataset gets its own picture: a unique cover generated from its name (always the same
# for the same name), or a custom image uploaded by the user (static/covers/<name>.<ext>).
# ---------------------------------------------------------------------------------------------
COVER_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
COVER_MAX_BYTES = 4 * 1024 * 1024
_EMOJI_HINTS = (
    ("weather", "🌤️"), ("meteo", "🌤️"), ("climate", "🌍"), ("diabet", "🩺"), ("heart", "❤️"), ("cancer", "🧬"),
    ("covid", "🦠"), ("health", "🩺"), ("electric", "⚡"), ("energy", "🔋"), ("power", "⚡"), ("fire", "🔥"),
    ("player", "⚽"), ("football", "⚽"), ("soccer", "⚽"), ("nba", "🏀"), ("titanic", "🚢"), ("iris", "🌸"),
    ("wine", "🍷"), ("house", "🏠"), ("housing", "🏠"), ("car", "🚗"), ("flight", "✈️"), ("air", "✈️"),
    ("stock", "📈"), ("bitcoin", "₿"), ("sale", "🛒"), ("retail", "🛒"), ("customer", "👥"), ("churn", "👥"),
    ("bank", "🏦"), ("credit", "💳"), ("student", "🎓"), ("school", "🎓"), ("music", "🎵"), ("movie", "🎬"),
    ("food", "🍽️"), ("water", "💧"), ("traffic", "🚦"), ("sensor", "📡"), ("solar", "☀️"), ("wind", "🌬️"),
)


def dataset_emoji(name):
    low = str(name).lower()
    for key, emo in _EMOJI_HINTS:
        if key in low:
            return emo
    return "📊"


def _cover_dir():
    d = Path(app.static_folder) / "covers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def custom_cover_path(name):
    """Path of the uploaded picture of a dataset, or None."""
    base = analysis_engine.clean_dataset_name(name)
    d = _cover_dir()
    for ext in COVER_EXTS:
        f = d / f"{base}{ext}"
        if f.exists():
            return f
    return None


def _rng_bytes(seed, n):
    out, i = [], 0
    while len(out) < n:
        out.extend(hashlib.sha256(f"{seed}:{i}".encode("utf-8")).digest())
        i += 1
    return out[:n]


def generate_cover_svg(name):
    """Cover of a dataset: themed palette + icon when the name matches a theme, else a neutral data cover.
    Always the same picture for the same name (see covers.py)."""
    return covers.scene(covers.theme_for(name), name)


def get_datasets_meta():
    """Helper to collect lightweight metadata for all available datasets."""
    datasets = analysis_engine.list_datasets()
    datasets_meta = []
    for d in datasets:
        s = analysis_engine.load_summary(d)
        meta = {"name": d, "n_rows": 0, "n_columns": 0, "target_class": None, "dataset_type": "tabular"}
        if s is not None:
            try:
                meta.update({
                    "n_rows": s.get("n_rows", 0),
                    "n_columns": s.get("n_columns", 0),
                    "target_class": s.get("target_class", None),
                    "dataset_type": s.get("dataset_type", "tabular"),
                })
            except Exception:
                pass
        cp = custom_cover_path(d)
        meta["emoji"] = dataset_emoji(d)
        meta["has_custom_cover"] = cp is not None
        meta["cover_v"] = int(cp.stat().st_mtime) if cp else 0
        datasets_meta.append(meta)
    return datasets, datasets_meta


def theme_photo_path(name):
    """Photo of the dataset's theme (static/covers/themes/<theme>.jpg, see fetch_theme_photos.py), or None."""
    theme = covers.theme_for(name)
    if not theme:
        return None
    f = _cover_dir() / "themes" / f"{theme}.jpg"
    return f if f.exists() else None


@app.route("/dataset-cover/<name>")
def dataset_cover(name):
    """Picture of a dataset: uploaded image > photo of its theme > generated cover."""
    clean = analysis_engine.clean_dataset_name(name)
    cp = custom_cover_path(name) or theme_photo_path(clean)
    if cp is not None:
        resp = send_from_directory(cp.parent, cp.name)
    else:
        resp = Response(generate_cover_svg(clean), mimetype="image/svg+xml")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/theme-bg/<theme>.svg")
def theme_bg(theme):
    """Illustration used as the page background for a dataset theme (weather, health, ...)."""
    if theme not in covers._SCENES:
        return Response("Not found", status=404)
    resp = Response(covers.scene(theme), mimetype="image/svg+xml")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


def dataset_theme(name):
    """Theme of a dataset: from its name, else from its target column, else the generic 'data' theme."""
    if not name:
        return None
    t = covers.theme_for(name)
    if t:
        return t
    try:
        s = analysis_engine.load_summary(name) or {}
        t = covers.theme_for(s.get("target_class") or "")
    except Exception:
        t = None
    return t or "data"


app.jinja_env.globals["dataset_theme"] = dataset_theme
app.jinja_env.globals["theme_palette"] = covers.palette


def _image_kind(head):
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return None


@app.route("/api/dataset-cover/<name>", methods=["POST", "DELETE"])
def api_dataset_cover(name):
    """Upload (POST, multipart field 'file') or remove (DELETE) the custom picture of a dataset."""
    if name not in analysis_engine.list_datasets():
        return jsonify({"status": "error", "message": f"Dataset introuvable : {name}"}), 404
    base = analysis_engine.clean_dataset_name(name)
    if request.method == "DELETE":
        for ext in COVER_EXTS:
            f = _cover_dir() / f"{base}{ext}"
            if f.exists():
                f.unlink()
        return jsonify({"status": "ok", "custom": False})

    up = request.files.get("file")
    if up is None or not up.filename:
        return jsonify({"status": "error", "message": "Aucune image reçue"}), 400
    data = up.read(COVER_MAX_BYTES + 1)
    if len(data) > COVER_MAX_BYTES:
        return jsonify({"status": "error", "message": "Image trop lourde (max 4 Mo)"}), 413
    ext = _image_kind(data[:16])
    if ext is None:
        return jsonify({"status": "error", "message": "Format non supporté (PNG, JPG, WEBP ou GIF)"}), 415
    for old in COVER_EXTS:
        f = _cover_dir() / f"{base}{old}"
        if f.exists():
            f.unlink()
    (_cover_dir() / f"{base}{ext}").write_bytes(data)
    return jsonify({"status": "ok", "custom": True, "v": int(custom_cover_path(name).stat().st_mtime)})


@app.route("/")
def index():
    datasets, datasets_meta = get_datasets_meta()
    if not datasets:
        return render_template("no_data.html")

    default_dataset = "weatherHistory" if "weatherHistory" in datasets else datasets[0]
    return render_template(
        "choose_dataset.html",
        datasets=datasets,
        datasets_meta=datasets_meta,
        default_dataset=default_dataset,
    )


@app.route("/upload")
@app.route("/new-dataset")
def upload_page():
    datasets, datasets_meta = get_datasets_meta()
    default_dataset = "weatherHistory" if "weatherHistory" in datasets else (datasets[0] if datasets else None)
    return render_template(
        "upload.html",
        datasets=datasets,
        datasets_meta=datasets_meta,
        default_dataset=default_dataset,
    )


ML_PIPELINE_PHASES = {
    "eda": {
        "id": "eda",
        "step_num": "01",
        "title": "Analyse Exploratoire (EDA)",
        "full_title": "Analyse Exploratoire des Données & Diagnostics",
        "subtitle": "Explorez les dimensions brutes, le bruit, les outliers (Isolation Forest), les distributions de probabilités et l'importance des variables.",
        "icon": "🔍",
        "theme": "cyan-theme",
        "status": "ready",
        "status_label": "Prêt & Actif",
        "badge_class": "badge-cyan",
        "pills": ["5 Sous-Modules", "Outliers (Isolation Forest)", "KDE & Heatmap"],
        "action_text": "Explorer l'EDA ➔",
        "endpoint": "view_case_detail",
    },
    "preprocessing": {
        "id": "preprocessing",
        "step_num": "02",
        "title": "Prétraitement des Données",
        "full_title": "Nettoyage & Prétraitement (Data Preprocessing)",
        "subtitle": "Imputation des valeurs manquantes, traitement du bruit résiduel, gestion des doublons et encodage catégoriel.",
        "icon": "🧹",
        "theme": "blue-theme",
        "status": "pending",
        "status_label": "À configurer",
        "badge_class": "badge-blue",
        "pills": ["Imputation", "Encodage", "Nettoyage"],
        "action_text": "Consulter l'Étape ➔",
        "endpoint": "view_phase_detail",
    },
    "split": {
        "id": "split",
        "step_num": "03",
        "title": "Split Train / Test",
        "full_title": "Séparation Train / Test",
        "subtitle": "Choix de la variable cible et de la part du jeu de test (chronologique, stratifié ou aléatoire) après le prétraitement et avant l'entraînement des modèles.",
        "icon": "✂️",
        "theme": "teal-theme",
        "status": "pending",
        "status_label": "À configurer",
        "badge_class": "badge-cyan",
        "pills": ["Train / Test", "Stratification", "Anti-fuite"],
        "action_text": "Consulter l'Étape ➔",
        "endpoint": "view_phase_detail",
    },
    "model": {
        "id": "model",
        "step_num": "04",
        "title": "Model Selection",
        "full_title": "Sélection & Comparaison des Modèles",
        "subtitle": "Régression ou classification : modèles individuels (SVM, KNN, Arbre) et ensemble learning (Random Forest, AdaBoost, XGBoost, CatBoost, LightGBM), avec validation (20 % du train), métriques F1 et AUC (classification) ou R² (régression) et réglage des hyperparamètres.",
        "icon": "🤖",
        "theme": "rose-theme",
        "status": "pending",
        "status_label": "À configurer",
        "badge_class": "badge-rose",
        "pills": ["Régression", "Classification", "Individuels & Ensemble"],
        "action_text": "Consulter l'Étape ➔",
        "endpoint": "view_phase_detail",
    },
}

PIPELINE_CASES = {
    "descriptive": {
        "id": "descriptive",
        "step_num": "01",
        "title": "Analyse Descriptive",
        "full_title": "Analyse Descriptive & Profiling du Dataset",
        "subtitle": "Examinez les dimensions du dataset, types de données, valeurs manquantes et doublons.",
        "icon": "📊",
        "prev_case": None,
        "next_case": "noise",
    },
    "noise": {
        "id": "noise",
        "step_num": "02",
        "title": "Noise (Bruit)",
        "full_title": "Détection du Bruit (Noise) & Fluctuations des Signaux",
        "subtitle": "Analyse des variations à haute fréquence et séparation signal/bruit par lissage.",
        "icon": "〰️",
        "prev_case": "descriptive",
        "next_case": "outliers",
    },
    "outliers": {
        "id": "outliers",
        "step_num": "03",
        "title": "Outliers (Aberrants)",
        "full_title": "Détection Globale des Valeurs Aberrantes (Isolation Forest)",
        "subtitle": "Une seule détection sur toutes les variables numériques à la fois : combien de lignes sont aberrantes dans le dataset.",
        "icon": "📦",
        "prev_case": "noise",
        "next_case": "distribution",
    },
    "distribution": {
        "id": "distribution",
        "step_num": "04",
        "title": "Distribution (Cible & KDE)",
        "full_title": "Distribution des Variables (Cible, Histogrammes KDE & Corrélations)",
        "subtitle": "Densités de probabilités, asymétrie, variable cible et carte des corrélations de Pearson.",
        "icon": "📈",
        "prev_case": "outliers",
        "next_case": "feature-importance",
    },
    "feature-importance": {
        "id": "feature-importance",
        "step_num": "05",
        "title": "Feature Importance",
        "full_title": "Feature Importance (Analyse par Variance & Corrélation)",
        "subtitle": "Hiérarchisation objective des variables basée sur l'ANOVA (test F) vers une cible catégorielle et sur les corrélations.",
        "icon": "⭐",
        "prev_case": "distribution",
        "next_case": "quality",
    },
    "quality": {
        "id": "quality",
        "step_num": "06",
        "title": "Data Quality",
        "full_title": "Qualité des Données : faut-il traiter le dataset ?",
        "subtitle": "Indicateurs de valeurs manquantes, doublons, outliers, bruit et déséquilibre des classes, avec un verdict de traitement pour chacun.",
        "icon": "🩺",
        "prev_case": "feature-importance",
        "next_case": None,
    },
}


@app.route("/dataset/<dataset_name>")
def view_dataset(dataset_name):
    datasets = analysis_engine.list_datasets()
    data = analysis_engine.load_dataset_results(dataset_name)

    if not data:
        return f"Dataset '{dataset_name}' not found.", 404

    datasets, datasets_meta = get_datasets_meta()

    # Dynamic copy of pipeline phases based on preprocessing state
    import copy
    phases = copy.deepcopy(ML_PIPELINE_PHASES)
    if split.get_config(dataset_name):
        phases["split"]["status"] = "ready"
        phases["split"]["status_label"] = "Split défini"
        phases["split"]["badge_class"] = "badge-emerald"
    if model_selection.has_results(dataset_name):
        phases["model"]["status"] = "ready"
        phases["model"]["status_label"] = "Modèles comparés"
        phases["model"]["badge_class"] = "badge-emerald"
    if preprocessing.has_preprocessed(dataset_name):
        phases["preprocessing"]["status"] = "ready"
        phases["preprocessing"]["status_label"] = "Nettoyé & Synchronisé"
        phases["preprocessing"]["badge_class"] = "badge-emerald"

    # Renders the High-Level ML Pipeline (EDA, Preprocessing, Split, Model Selection)
    return render_template(
        "ml_pipeline.html",
        datasets=datasets,
        datasets_meta=datasets_meta,
        active_dataset=dataset_name,
        summary=data["summary"],
        phases=phases,
    )


@app.route("/dataset/<dataset_name>/phase/<phase_name>")
def view_phase_detail(dataset_name, phase_name):
    if phase_name not in ML_PIPELINE_PHASES:
        return redirect(url_for("view_dataset", dataset_name=dataset_name))

    if phase_name == "eda":
        return redirect(url_for("view_case_detail", dataset_name=dataset_name, case_name="descriptive"))

    datasets = analysis_engine.list_datasets()
    data = analysis_engine.load_dataset_results(dataset_name)

    if not data:
        return f"Dataset '{dataset_name}' not found.", 404

    datasets, datasets_meta = get_datasets_meta()

    if phase_name == "split":
        try:
            sp, error = split.get_profile(dataset_name), None
        except Exception as e:
            sp, error = None, f"{type(e).__name__}: {e}"
        return render_template(
            "split.html",
            datasets=datasets,
            datasets_meta=datasets_meta,
            active_dataset=dataset_name,
            current_phase=ML_PIPELINE_PHASES[phase_name],
            summary=data["summary"],
            sp=sp,
            error=error,
        )

    if phase_name == "preprocessing":
        try:
            profile, error = preprocessing.get_profile(dataset_name), None
        except (KeyError, FileNotFoundError) as e:
            profile, error = None, f"Fichier source introuvable : {str(e).strip(chr(34) + chr(39))}"
        except Exception as e:
            profile, error = None, f"{type(e).__name__}: {e}"
        return render_template(
            "preprocessing.html",
            datasets=datasets,
            datasets_meta=datasets_meta,
            active_dataset=dataset_name,
            current_phase=ML_PIPELINE_PHASES[phase_name],
            summary=data["summary"],
            profile=profile,
            error=error,
        )

    if phase_name == "model":
        try:
            mp, error = model_selection.get_overview(dataset_name), None
        except (KeyError, FileNotFoundError) as e:
            mp, error = None, f"Fichier source introuvable : {str(e).strip(chr(34) + chr(39))}"
        except Exception as e:
            mp, error = None, f"{type(e).__name__}: {e}"
        return render_template(
            "model.html",
            datasets=datasets,
            datasets_meta=datasets_meta,
            active_dataset=dataset_name,
            current_phase=ML_PIPELINE_PHASES[phase_name],
            summary=data["summary"],
            mp=mp,
            error=error,
        )

    # For downstream phases (split, model), retrieve pipeline data info
    pipe_meta = preprocessing.get_preprocessed_meta(dataset_name)
    is_clean = pipe_meta is not None
    pipe_info = {
        "source": "preprocessed" if is_clean else "raw",
        "n_rows": pipe_meta["rows"] if is_clean else data["summary"].get("n_rows"),
        "n_cols": pipe_meta["cols"] if is_clean else data["summary"].get("n_columns"),
        "target": pipe_meta.get("target") if is_clean else None,
        "created_at": pipe_meta.get("created_at") if is_clean else None,
    }

    return render_template(
        "empty_phase.html",
        datasets=datasets,
        datasets_meta=datasets_meta,
        active_dataset=dataset_name,
        current_phase=ML_PIPELINE_PHASES[phase_name],
        all_phases=ML_PIPELINE_PHASES,
        summary=data["summary"],
        is_clean=is_clean,
        pipe_info=pipe_info,
    )


@app.route("/dataset/<dataset_name>/eda")
@app.route("/dataset/<dataset_name>/eda/<case_name>")
@app.route("/dataset/<dataset_name>/case/<case_name>")
def view_case_detail(dataset_name, case_name="descriptive"):
    if case_name not in PIPELINE_CASES:
        case_name = "descriptive"

    datasets = analysis_engine.list_datasets()
    data = analysis_engine.load_dataset_results(dataset_name)

    if not data:
        return f"Dataset '{dataset_name}' not found.", 404

    datasets, datasets_meta = get_datasets_meta()

    raw_tables = data.get("tables", {})
    tables = {name: format_dataframe(df) for name, df in raw_tables.items()}

    feature_importance = compute_feature_importance(data["summary"], raw_tables)

    outliers_dict = {}
    outliers_summary_df = raw_tables.get("outliers_summary")
    if outliers_summary_df is not None:
        col_attr = "column" if "column" in outliers_summary_df.columns else outliers_summary_df.columns[0]
        for _, row in outliers_summary_df.iterrows():
            col_name = str(row[col_attr])
            key = analysis_engine.safe_name(col_name)
            try:
                q1 = round(float(row.get("Q1", 0)), 3)
                q3 = round(float(row.get("Q3", 0)), 3)
                iqr = round(float(row.get("IQR", 0)), 3)
                lb = round(float(row.get("lower_bound", 0)), 3)
                ub = round(float(row.get("upper_bound", 0)), 3)
                cnt = int(float(row.get("outlier_count", 0)))
                pct = round(float(row.get("outlier_percent", 0)), 2)
            except Exception:
                q1, q3, iqr, lb, ub, cnt, pct = 0, 0, 0, 0, 0, 0, 0.0

            outliers_dict[key] = {
                "column": col_name,
                "key": key,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lb,
                "upper_bound": ub,
                "count": cnt,
                "percent": pct,
            }
    outliers_summary_list = list(outliers_dict.values())
    outliers_summary_list.sort(key=lambda x: x["count"], reverse=True)

    quality, quality_error = None, None
    if case_name == "quality":
        try:
            quality = analysis_engine.get_data_quality(dataset_name, target=request.args.get("target") or None)
        except (KeyError, FileNotFoundError) as e:
            quality_error = f"Fichier source introuvable : {str(e).strip(chr(34) + chr(39))}"
        except Exception as e:
            quality_error = f"{type(e).__name__}: {e}"

    # Renders the EDA sidebar layout with the selected sub-step
    anova_targets, anova_default = analysis_engine.get_anova_targets(dataset_name, data)

    return render_template(
        "case_detail.html",
        datasets=datasets,
        datasets_meta=datasets_meta,
        active_dataset=dataset_name,
        current_case=PIPELINE_CASES[case_name],
        all_cases=PIPELINE_CASES,
        summary=data["summary"],
        plots=data["plots"],
        outliers_columns=data["outliers_columns"],
        tables=tables,
        feature_importance=feature_importance,
        anova_targets=anova_targets,
        anova_default=anova_default,
        outliers_dict=outliers_dict,
        outliers_summary_list=outliers_summary_list,
        quality=quality,
        quality_error=quality_error,
    )

# Alias for backwards compatibility
view_eda = view_case_detail


@app.route("/results/<dataset_name>/plots/<path:filename>")
def serve_plot(dataset_name, filename):
    plots_dir = Path("results") / analysis_engine.clean_dataset_name(dataset_name) / "plots"
    return send_from_directory(plots_dir, filename)


@app.route("/api/data-quality/<dataset_name>")
def api_data_quality(dataset_name):
    """The five data-quality indicators (+ verdicts) as JSON. Optional ?target=<column>."""
    try:
        return jsonify(analysis_engine.get_data_quality(dataset_name, target=request.args.get("target") or None))
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("\"'")}), 404


@app.route("/api/outliers/<dataset_name>/<column_key>")
def api_outliers(dataset_name, column_key):
    limit = request.args.get("limit", default=100, type=int)
    result = analysis_engine.get_outliers_sample(dataset_name, column_key, limit=limit)
    return jsonify(result)


@app.route("/api/noise-series/<dataset_name>/<path:feature>")
def api_noise_series(dataset_name, feature):
    """Interactive rolling-mean / noise data for one variable (computed in analysis_engine)."""
    src = find_source_file(analysis_engine.clean_dataset_name(dataset_name)) or find_source_file(dataset_name)
    try:
        payload = analysis_engine.get_noise_series(
            dataset_name, feature,
            window=request.args.get("window", type=int),
            method="median" if request.args.get("method") == "median" else "mean",
            start=request.args.get("start") or None,
            end=request.args.get("end") or None,
            k=request.args.get("k", default=2.0, type=float),
            source_path=None,  # uses summary["source_file"]; falls back below
        )
    except FileNotFoundError:
        if src is None:
            return jsonify({"status": "error", "message": f"Source file of '{dataset_name}' not found"}), 404
        payload = analysis_engine.get_noise_series(
            dataset_name, feature,
            window=request.args.get("window", type=int),
            method="median" if request.args.get("method") == "median" else "mean",
            start=request.args.get("start") or None,
            end=request.args.get("end") or None,
            k=request.args.get("k", default=2.0, type=float),
            source_path=src,
        )
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    return jsonify(payload)


@app.route("/api/outliers-plot/<dataset_name>/<path:feature>")
def api_outliers_plot(dataset_name, feature):
    """Interactive boxplot / histogram data for one variable (computed in analysis_engine)."""
    kwargs = dict(
        k=request.args.get("k", default=1.5, type=float),
        points="all" if request.args.get("points") == "all" else "outliers",
    )
    try:
        payload = analysis_engine.get_outlier_series(dataset_name, feature, source_path=None, **kwargs)
    except FileNotFoundError:
        src = find_source_file(analysis_engine.clean_dataset_name(dataset_name)) or find_source_file(dataset_name)
        if src is None:
            return jsonify({"status": "error", "message": f"Source file of '{dataset_name}' not found"}), 404
        payload = analysis_engine.get_outlier_series(dataset_name, feature, source_path=src, **kwargs)
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    return jsonify(payload)


@app.route("/api/isolation-forest/<dataset_name>")
def api_isolation_forest(dataset_name):
    """Global (all-features) outlier detection with an Isolation Forest."""
    c = request.args.get("contamination", "auto")
    if c != "auto":
        try:
            c = float(c)
        except ValueError:
            c = "auto"
    kwargs = dict(contamination=c)
    try:
        payload = analysis_engine.get_isolation_forest(dataset_name, source_path=None, **kwargs)
    except FileNotFoundError:
        src = find_source_file(analysis_engine.clean_dataset_name(dataset_name)) or find_source_file(dataset_name)
        if src is None:
            return jsonify({"status": "error", "message": f"Source file of '{dataset_name}' not found"}), 404
        payload = analysis_engine.get_isolation_forest(dataset_name, source_path=src, **kwargs)
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    return jsonify(payload)


@app.route("/api/distribution-plot/<dataset_name>/<path:feature>")
def api_distribution_plot(dataset_name, feature):
    """Interactive data for the Distributions page: one variable, the correlation matrix or the target balance."""
    try:
        if feature == "__corr__":
            return jsonify(analysis_engine.get_correlation_matrix(dataset_name))
        if feature == "__target__":
            return jsonify(analysis_engine.get_target_balance(dataset_name))
        try:
            payload = analysis_engine.get_distribution_series(dataset_name, feature, source_path=None)
        except FileNotFoundError:
            src = find_source_file(analysis_engine.clean_dataset_name(dataset_name)) or find_source_file(dataset_name)
            if src is None:
                return jsonify({"status": "error", "message": f"Source file of '{dataset_name}' not found"}), 404
            payload = analysis_engine.get_distribution_series(dataset_name, feature, source_path=src)
    except KeyError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    return jsonify(payload)


@app.route("/api/anova/<dataset_name>/<path:target>")
def api_anova(dataset_name, target):
    """ANOVA (test F) de chaque variable numérique contre une cible catégorielle."""
    try:
        return jsonify(analysis_engine.get_anova_importance(dataset_name, target))
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("'\"")}), 404


def _with_split(dataset_name, body):
    """Le split (cible + part du test) vient de l'étape Split : il l'emporte sur ce que poste le prétraitement."""
    body = dict(body) if isinstance(body, dict) else {}
    sc = split.get_config(dataset_name)
    if sc:
        body["test_size"] = sc["test_size"]
        body["target"] = sc["target"]
        body["split_strategy"] = sc.get("strategy", "auto")
        if sc.get("excluded"):                                   # variables non retenues dans X
            body["drop_columns"] = sorted(set(body.get("drop_columns") or []) | set(sc["excluded"]))
    return body


@app.route("/api/split/<dataset_name>/preview", methods=["POST"])
def api_split_preview(dataset_name):
    """Aperçu du découpage train/test pour les réglages postés. Rien n'est sauvegardé."""
    try:
        return jsonify({"status": "ok", **split.preview(dataset_name, request.get_json(silent=True) or {})})
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("'\"")}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/split/<dataset_name>/save", methods=["POST"])
def api_split_save(dataset_name):
    """Sauvegarde le split ; le prétraitement et les features déjà calculés deviennent obsolètes."""
    try:
        result = split.save(dataset_name, request.get_json(silent=True) or {})
        preprocessing.delete_preprocessed(dataset_name)
        model_selection.delete_results(dataset_name)
        return jsonify({"status": "ok", **result})
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("'\"")}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/preprocessing/<dataset_name>/run", methods=["POST"])
def api_preprocessing_run(dataset_name):
    """Run the preprocessing pipeline with the posted config and save the cleaned dataset."""
    try:
        save = request.args.get("save", "1") != "0"          # ?save=0 : simple aperçu du pipeline complet, rien n'est enregistré
        result = preprocessing.process(dataset_name, _with_split(dataset_name, request.get_json(silent=True)), save=save,
                                       refresh=request.args.get("refresh") == "1")
        if save:
            model_selection.delete_results(dataset_name)      # les modèles dépendent des données prétraitées
        return jsonify({"status": "ok", **result})
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("'\"")}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/preprocessing/<dataset_name>/step/<step>", methods=["POST"])
def api_preprocessing_step(dataset_name, step):
    """Result of ONE preprocessing step (earlier enabled steps applied first). Nothing is saved."""
    try:
        return jsonify({"status": "ok", **preprocessing.step_result(dataset_name, step, _with_split(dataset_name, request.get_json(silent=True)),
                                                                      refresh=request.args.get("refresh") == "1")})
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("'\"")}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), 500


@app.route("/dataset/<dataset_name>/phase/model/<task>")
def view_model_task(dataset_name, task):
    """Page d'un type de problème (regression | classification) : sidebar Modèles individuels / Ensemble learning."""
    if task not in ("regression", "classification"):
        return redirect(url_for("view_phase_detail", dataset_name=dataset_name, phase_name="model"))
    data = analysis_engine.load_dataset_results(dataset_name)
    if not data:
        return f"Dataset '{dataset_name}' not found.", 404
    datasets, datasets_meta = get_datasets_meta()
    try:
        mp, error = model_selection.get_task_profile(dataset_name, task), None
    except (KeyError, FileNotFoundError) as e:
        mp, error = None, f"Fichier source introuvable : {str(e).strip(chr(34) + chr(39))}"
    except Exception as e:
        mp, error = None, f"{type(e).__name__}: {e}"
    return render_template(
        "model_task.html",
        datasets=datasets,
        datasets_meta=datasets_meta,
        active_dataset=dataset_name,
        current_phase=ML_PIPELINE_PHASES["model"],
        summary=data["summary"],
        task=task,
        mp=mp,
        error=error,
    )


_MODEL_JOBS = {}
_MODEL_JOBS_LOCK = threading.Lock()


def _model_worker(job_id, dataset_name, cfg, refresh=False):
    def progress(i, total, name):
        _MODEL_JOBS[job_id].update(done=i, total=total, current=name)
    try:
        res = model_selection.run(dataset_name, cfg, progress=progress, refresh=refresh)
        _MODEL_JOBS[job_id].update(state="done", result=res)
    except (KeyError, FileNotFoundError) as e:
        _MODEL_JOBS[job_id].update(state="error", message=str(e).strip("'\""))
    except ValueError as e:
        _MODEL_JOBS[job_id].update(state="error", message=str(e))
    except Exception as e:
        _MODEL_JOBS[job_id].update(state="error", message=f"{type(e).__name__}: {e}")


@app.route("/api/model/<dataset_name>/run", methods=["POST"])
def api_model_run(dataset_name):
    """Lance l'entraînement en arrière-plan et renvoie un job_id (suivi via /api/model/<ds>/status/<job_id>)."""
    cfg = request.get_json(silent=True) or {}
    try:
        model_selection.clean_config(cfg)                  # validation immédiate (erreur claire sans lancer de thread)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    job_id = uuid.uuid4().hex[:12]
    with _MODEL_JOBS_LOCK:
        for k in [k for k, v in _MODEL_JOBS.items() if time.time() - v["t0"] > 3600]:
            _MODEL_JOBS.pop(k, None)
        _MODEL_JOBS[job_id] = {"state": "running", "done": 0, "total": len(cfg.get("models") or []), "current": "", "t0": time.time()}
    threading.Thread(target=_model_worker, args=(job_id, dataset_name, cfg, request.args.get("refresh") == "1"), daemon=True).start()
    return jsonify({"status": "started", "job_id": job_id})


@app.route("/api/model/<dataset_name>/status/<job_id>")
def api_model_status(dataset_name, job_id):
    job = _MODEL_JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Tâche introuvable (serveur redémarré ?)."}), 404
    out = {"status": job["state"], "done": job["done"], "total": job["total"], "current": job["current"],
           "elapsed": round(time.time() - job["t0"], 1)}
    if job["state"] == "done":
        out["result"] = job["result"]
    elif job["state"] == "error":
        out["message"] = job["message"]
    return jsonify(out)


def _scenario_payload(res):
    out = dict(res)
    out["deltas"] = scenario_report.deltas(res)
    out["summary_md"] = scenario_report.auto_summary(res, "fr")
    return out


def _scenario_worker(job_id, dataset_name, task, window, refresh):
    def progress(i, total, name):
        _MODEL_JOBS[job_id].update(done=i, total=total, current=name)
    try:
        res = scenario_report.run(dataset_name, task=task, window=window, refresh=refresh, progress=progress)
        _MODEL_JOBS[job_id].update(state="done", result=_scenario_payload(res))
    except (KeyError, FileNotFoundError) as e:
        _MODEL_JOBS[job_id].update(state="error", message=str(e).strip("'\""))
    except ValueError as e:
        _MODEL_JOBS[job_id].update(state="error", message=str(e))
    except Exception as e:
        _MODEL_JOBS[job_id].update(state="error", message=f"{type(e).__name__}: {e}")


@app.route("/api/scenarios/<dataset_name>/run", methods=["POST"])
def api_scenarios_run(dataset_name):
    """Compare les modèles avec / sans outliers et avec / sans bruit (en arrière-plan ; suivi via /api/model/<ds>/status/<job_id>)."""
    body = request.get_json(silent=True) or {}
    task = body.get("task") if body.get("task") in ("regression", "classification") else None
    window = body.get("window")
    job_id = uuid.uuid4().hex[:12]
    with _MODEL_JOBS_LOCK:
        for k in [k for k, v in _MODEL_JOBS.items() if time.time() - v["t0"] > 3600]:
            _MODEL_JOBS.pop(k, None)
        _MODEL_JOBS[job_id] = {"state": "running", "done": 0, "total": 0, "current": "", "t0": time.time()}
    threading.Thread(target=_scenario_worker, args=(job_id, dataset_name, task, window, bool(body.get("refresh"))), daemon=True).start()
    return jsonify({"status": "started", "job_id": job_id})


@app.route("/api/scenarios/<dataset_name>/saved")
def api_scenarios_saved(dataset_name):
    """Dernière comparaison avec / sans outliers et bruit DÉJÀ enregistrée dans analysis.db (aucun entraînement)."""
    task = request.args.get("task") if request.args.get("task") in ("regression", "classification") else None
    try:
        res, exact = scenario_report.cached(dataset_name, task=task, window=request.args.get("window", type=int))
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("'\"")}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), 500
    if res is None:
        return jsonify({"status": "none"})
    return jsonify({"status": "ok", "exact": exact, "result": _scenario_payload(res)})


@app.route("/api/scenarios/<dataset_name>/pdf", methods=["POST"])
def api_scenarios_pdf(dataset_name):
    """Génère le rapport PDF à partir des résultats déjà enregistrés (aucun entraînement)."""
    body = request.get_json(silent=True) or {}
    task = body.get("task") if body.get("task") in ("regression", "classification") else None
    lang = body.get("language") if body.get("language") in ("fr", "en") else "fr"
    try:
        ds = analysis_engine.clean_dataset_name(dataset_name)
        res, _exact = scenario_report.cached(ds, task=task, window=body.get("window"))
        if res is None:
            return jsonify({"status": "error", "message": "Aucun résultat enregistré : lancez d'abord la comparaison."}), 404
        path = scenario_report.build_model_comparison_pdf(ds, language=lang, res=res)
        return jsonify({"status": "ok", "name": path.name, "url": url_for("serve_report", dataset_name=ds, filename=path.name)})
    except (KeyError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e).strip("'\"")}), 404
    except ai_interpret.AIError as e:
        return jsonify({"status": "error", "message": str(e)}), e.status
    except Exception as e:
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/correlation-target/<dataset_name>/<target_col>")
def api_correlation_target(dataset_name, target_col):
    """Compute Pearson |r| between a chosen target column and all other numeric features."""
    data = analysis_engine.load_dataset_results(dataset_name)
    if not data:
        return jsonify({"status": "error", "message": "Dataset not found"}), 404

    raw_tables = data.get("tables", {})
    corr_df = raw_tables.get("correlation")
    summary = data["summary"]
    numeric_cols = summary.get("numeric_columns", [])

    if corr_df is None or target_col not in corr_df.columns:
        return jsonify({"status": "error", "message": f"Target '{target_col}' not in correlation matrix"}), 400

    results = []
    for col in numeric_cols:
        if col == target_col:
            continue
        if col not in corr_df.index:
            continue
        val = corr_df.loc[col, target_col]
        r_val = abs(float(val)) if pd.notna(val) else 0.0
        # Strength assessment
        if r_val >= 0.7:
            strength = "Forte"
            color = "#34d399"
        elif r_val >= 0.4:
            strength = "Modérée"
            color = "#fbbf24"
        elif r_val >= 0.2:
            strength = "Faible"
            color = "#fb923c"
        else:
            strength = "Très faible"
            color = "#f87171"
        results.append({
            "name": col,
            "r": round(float(val), 4) if pd.notna(val) else 0.0,
            "corr_abs": round(r_val, 4),
            "corr_pct": round(r_val * 100.0, 1),
            "strength": strength,
            "color": color,
        })

    results.sort(key=lambda x: x["corr_abs"], reverse=True)
    for idx, item in enumerate(results, 1):
        item["rank"] = idx

    return jsonify({
        "status": "ok",
        "target": target_col,
        "features": results,
    })


def _fi_provider(dataset_name):
    """Données de l'onglet « Importance des variables » pour l'interprétation IA."""
    def provider(kind, target):
        data = analysis_engine.load_dataset_results(dataset_name)
        if kind == "fi_anova":
            res = analysis_engine.get_anova_importance(dataset_name, target)
            return {"test": "ANOVA à un facteur (F)", "cible": res["target"], "n_classes": res["n_classes"], "classes": res["classes"],
                    "variables": [{k: f[k] for k in ("rank", "name", "f", "p", "eta2", "score_pct")} for f in res["features"][:25]]}
        resp = api_correlation_target(dataset_name, target)
        payload = (resp[0] if isinstance(resp, tuple) else resp).get_json()
        if payload.get("status") != "ok":
            raise ai_interpret.AIError(payload.get("message", "Cible invalide"), 400)
        return {"cible": payload["target"],
                "variables": [{k: f[k] for k in ("rank", "name", "r", "strength")} for f in payload["features"][:25]]}
    return provider


@app.route("/api/agent/chat", methods=["POST"])
def api_agent_chat():
    """Agent conversationnel. Corps JSON : {dataset?, messages: [{role, content}, ...]}"""
    body = request.get_json(silent=True) or {}
    try:
        return jsonify({"status": "ok", **agent.chat(body.get("messages"), body.get("dataset"))})
    except ai_interpret.AIError as e:
        return jsonify({"status": "error", "message": str(e)}), e.status


@app.route("/results/<dataset_name>/reports/<path:filename>")
def serve_report(dataset_name, filename):
    """Télécharge un rapport PDF généré par l'agent."""
    reports_dir = Path("results") / analysis_engine.clean_dataset_name(dataset_name) / "reports"
    return send_from_directory(reports_dir, filename, mimetype="application/pdf", download_name=filename)


@app.route("/api/dataset/<dataset_name>", methods=["DELETE"])
def api_delete_dataset(dataset_name):
    """Supprime l'analyse d'un dataset (base SQLite + graphiques + rapports). Le fichier source d'origine est conservé."""
    name = analysis_engine.clean_dataset_name(dataset_name)
    if not analysis_engine.delete_dataset(name):
        return jsonify({"status": "error", "message": f"Dataset introuvable : {name}"}), 404
    preprocessing.delete_preprocessed(name)
    split.delete_config(name)
    step_cache.invalidate(name)
    cp = custom_cover_path(name)
    if cp is not None:
        cp.unlink()
    return jsonify({"status": "ok", "deleted": name, "remaining": analysis_engine.list_datasets()})


@app.route("/api/interpret/<dataset_name>", methods=["POST"])
def api_interpret(dataset_name):
    """Interprétation Groq d'un graphique. Corps JSON : {kind, feature?, target?}"""
    body = request.get_json(silent=True) or {}
    try:
        result = ai_interpret.interpret(
            dataset_name,
            kind=str(body.get("kind", "")),
            feature=str(body.get("feature", "") or ""),
            target=str(body.get("target", "") or ""),
            fi_provider=_fi_provider(dataset_name),
        )
        return jsonify({"status": "ok", **result})
    except ai_interpret.AIError as e:
        return jsonify({"status": "error", "message": str(e)}), e.status


def find_source_file(dataset_name):
    """Find <dataset_name>.<ext> in the project folder or the uploads folder."""
    folders = [Path(__file__).resolve().parent, Path.cwd(), Path(app.config["UPLOAD_FOLDER"]).resolve()]
    for folder in folders:
        for ext in sorted(analysis_engine.SUPPORTED_EXTENSIONS):
            candidate = folder / f"{dataset_name}{ext}"
            if candidate.exists():
                return candidate
    return None


@app.route("/api/run-analysis", methods=["POST"])
def api_run_analysis():
    # Empty fields = automatic detection
    target_class = request.form.get("target_class", "").strip() or None
    date_column = request.form.get("date_column", "").strip() or analysis_engine.AUTO
    dataset_name = request.form.get("dataset_name", "").strip()

    uploaded_file = request.files.get("csv_file")

    if uploaded_file and uploaded_file.filename != "":
        filename = secure_filename(uploaded_file.filename)
        ext = Path(filename).suffix.lower()
        if ext not in analysis_engine.SUPPORTED_EXTENSIONS:
            return jsonify({"status": "error",
                            "message": f"Unsupported file type '{ext}'. "
                                       f"Use: {', '.join(sorted(analysis_engine.SUPPORTED_EXTENSIONS))}"}), 400
        dataset_name = analysis_engine.clean_dataset_name(dataset_name or Path(filename).stem)
        file_to_run = Path(app.config["UPLOAD_FOLDER"]) / filename
        uploaded_file.save(file_to_run)
    else:
        # No new file: re-run an existing dataset from its original source file
        if not dataset_name:
            return jsonify({"status": "error", "message": "Choose a file or a dataset name."}), 400
        dataset_name = analysis_engine.clean_dataset_name(dataset_name)
        existing_summary = analysis_engine.load_summary(dataset_name)
        if existing_summary is not None:
            file_to_run = Path(existing_summary["source_file"])
        else:
            # First run (empty database): look for <dataset_name>.csv/.xlsx/... next to app.py or in uploads/
            file_to_run = find_source_file(dataset_name)
            if file_to_run is None:
                return jsonify({"status": "error",
                                "message": f"No file '{dataset_name}' found. Put {dataset_name}.csv next to app.py "
                                           f"or upload it from the /upload page."}), 404
        if not file_to_run.exists():
            return jsonify({"status": "error",
                            "message": f"Source file '{file_to_run}' not found. Upload it again."}), 404

    try:
        analysis_engine.run_full_analysis(
            file_to_run,
            dataset_name=dataset_name,
            date_column=date_column,
            target_class=target_class,
        )
        return jsonify({
            "status": "success",
            "message": f"Analysis completed for dataset '{dataset_name}'!",
            "redirect_url": url_for("view_dataset", dataset_name=dataset_name),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {e}"}), 500


@app.route("/export/<dataset_name>/<table_name>")
def export_table(dataset_name, table_name):
    dataset_name = analysis_engine.clean_dataset_name(dataset_name)
    table_name = Path(table_name).name
    csv_text = analysis_engine.export_table_csv(dataset_name, table_name)
    if csv_text is None:
        return "Table not found", 404
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{dataset_name}_{table_name}.csv"'},
    )


if __name__ == "__main__":
    print("Starting Analytics Dashboard...")
    app.run(host="127.0.0.1", port=5007, debug=True)
