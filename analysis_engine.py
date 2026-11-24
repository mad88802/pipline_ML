import os
import csv
import json
import re
import shutil
import sqlite3
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ======================================================================
# CONFIGURATION & CONSTANTS
# ======================================================================

RESULTS_DIR = Path("results")
DB_NAME = "analysis.db"          # SQLite file, created inside RESULTS_DIR

AUTO = "auto"

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}

MAX_NUMERIC_PLOTS = 30          # max numeric columns that get boxplot/distribution/smoothed plots
MAX_OUTLIER_TABLE = 5000        # max rows sent to the outliers table of the browser
MAX_OUTLIER_COLUMNS = 60        # max numeric columns that get an outlier file
MAX_CATEGORICAL_PLOTS = 15      # max categorical columns that get a bar chart
MAX_CATEGORY_CARDINALITY = 50   # above this, a text column is considered "high cardinality"
MAX_CLASSES_SHOWN = 30
MAX_PLOT_ROWS = 100_000         # plots use a random sample above this size
SMOOTH_ROWS = 200
SMOOTH_WINDOW = 5
HEATMAP_ANNOT_MAX = 15

TARGET_HINTS = (
    "target", "label", "class", "outcome", "survived", "species",
    "churn", "diagnosis", "status", "type", "category", "result"
)
ID_NAME_RE = re.compile(r"^(unnamed: ?\d+|id|index|row ?id|row_?number)$", re.IGNORECASE)

TABLES_WITH_INDEX = {
    "describe_numeric",
    "describe_categorical",
    "n_unique",
    "missing_report",
    "correlation",
    "class_balance",
}


# ======================================================================
# NAMES / SECURITY / HASHING
# ======================================================================

def safe_name(text):
    """'Temperature (C)' -> 'Temperature__C_' (safe for file names)."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))


def clean_dataset_name(name):
    """Dataset names are used as folder names: keep letters, digits, _ and -."""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")
    return cleaned or "dataset"


def calculate_file_hash(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_cache_key(file_hash, config):
    payload = {
        "file_hash": file_hash,
        "config": config
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# DATA LOADING

def _sniff_separator(path, encoding):
    with open(path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(20000)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_data(file_path):

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    if extension in {".csv", ".tsv", ".txt"}:
        last_error = None
        for encoding in ("utf-8-sig", "latin-1", "utf-8"):
            try:
                sep = _sniff_separator(file_path, encoding)
                return pd.read_csv(file_path, sep=sep, encoding=encoding, low_memory=False)
            except UnicodeDecodeError as e:
                last_error = e
        if last_error:
            raise last_error

    raise ValueError(f"Unsupported format: {extension}")


# Alias for compatibility with web app
read_table = load_data


# 2) DESCRIPTION

def describe_data(df):

    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(exclude=np.number)

    numerical_stats = numeric.describe().T if not numeric.empty else pd.DataFrame()
    categorical_stats = categorical.describe(include="all").T if not categorical.empty else pd.DataFrame()
    unique_counts = df.nunique(dropna=True).to_frame("n_unique")

    return {
        "numerical_stats": numerical_stats,
        "categorical_stats": categorical_stats,
        "unique_counts": unique_counts,
        "numeric_columns": numeric.columns.tolist(),
        "categorical_columns": categorical.columns.tolist()
    }


# 3) MISSING VALUES REPORT

def missing_values_report(df):

    counts = df.isna().sum()
    percentages = (df.isna().mean() * 100).round(2)

    report = pd.DataFrame({
        "Missing_Count": counts,
        "Missing_Percentage": percentages,
        "missing_count": counts,
        "missing_percentage": percentages
    })

    return report.sort_values("Missing_Count", ascending=False)


# 4) DUPLICATES REPORT

def duplicates_report(df):

    count = int(df.duplicated().sum())
    duplicate_rows = df.loc[df.duplicated(keep=False)].copy()

    return {
        "duplicate_count": count,
        "duplicate_rows": duplicate_rows
    }


# 5) DATETIME DETECTION & DATASET TYPE CLASSIFICATION

def to_datetime_safe(series):
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True, errors="coerce")


def detect_datetime_columns(df, threshold=0.80):

    candidates = []

    for col in df.columns:
        series = df[col]

        name_suggests_date = any(
            word in str(col).lower()
            for word in ["date", "datetime", "timestamp", "time", "heure", "fecha", "day"]
        )

        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(col)
            continue

        if not (name_suggests_date or series.dtype == "object" or pd.api.types.is_string_dtype(series)):
            continue

        values = series.dropna()
        if values.empty:
            continue

        # Avoid parsing entire dataset for large files
        sample = values.head(1000)

        try:
            parsed = to_datetime_safe(sample)
            success_rate = parsed.notna().mean()
            if success_rate >= threshold:
                candidates.append(col)
        except (ValueError, TypeError):
            continue

    return candidates


def detect_dataset_type(df, date_column=None, force_timeseries=False):

    datetime_columns = detect_datetime_columns(df)

    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found.")
        datetime_columns = list(dict.fromkeys([date_column] + datetime_columns))

    if force_timeseries:
        if date_column is None:
            if not datetime_columns:
                raise ValueError("Specify date_column for a time series.")
            date_column = datetime_columns[0]
        dataset_type = "time_series"
        reason = "Time-series analysis explicitly requested."

    elif datetime_columns:
        date_column = date_column or datetime_columns[0]
        is_unique = df[date_column].is_unique if hasattr(df[date_column], "is_unique") else False
        dataset_type = "time_series" if is_unique else "time_indexed_tabular"
        reason = (
            f"Automatically detected temporal column '{date_column}' "
            f"({'time series with unique timestamps' if is_unique else 'tabular observations over time'})."
        )
    else:
        dataset_type = "tabular"
        reason = "No datetime column was detected."

    return {
        "dataset_type": dataset_type,
        "datetime_candidates": datetime_columns,
        "date_column": date_column,
        "reason": reason
    }


# 6) TARGET VARIABLE DETECTION

def _is_integer_like(s):
    s = s.dropna()
    if s.empty:
        return False
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        return bool(np.all(np.mod(s.to_numpy(), 1) == 0))
    return False


def is_id_like(df, col):
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    if ID_NAME_RE.match(str(col).strip()):
        return True
    return bool(
        pd.api.types.is_integer_dtype(s)
        and len(s) > 20
        and s.nunique() == len(s)
        and s.is_monotonic_increasing
    )


def detect_target(df, exclude=()):

    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col].dropna()
        n = s.nunique()
        if n < 2 or n > 10:
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if not _is_integer_like(s) or is_id_like(df, col):
                continue
        candidates.append(col)

    if not candidates:
        return None

    for col in candidates:
        name = str(col).lower()
        if name == "y" or any(h in name for h in TARGET_HINTS):
            return col

    last = df.columns[-1]
    if last in candidates:
        return last

    non_numeric = [c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])]
    return (non_numeric or candidates)[0]


# 7) NOISE DETECTION: REFERENCE & TIME-SERIES RESIDUALS

def detect_reference_noise(df, reference_pairs):

    if not reference_pairs:
        return pd.DataFrame()

    rows = []
    for measurement_col, config in reference_pairs.items():
        reference_col = config.get("reference_col")
        tolerance = config.get("tolerance", 0.0)

        if measurement_col not in df.columns or reference_col not in df.columns:
            continue

        measurement = pd.to_numeric(df[measurement_col], errors="coerce")
        reference = pd.to_numeric(df[reference_col], errors="coerce")
        error = measurement - reference

        mask = measurement.notna() & reference.notna() & (error.abs() > tolerance)

        for idx in df.index[mask]:
            rows.append({
                "index": str(idx),
                "variable": measurement_col,
                "reference_variable": reference_col,
                "measurement": measurement.loc[idx],
                "reference": reference.loc[idx],
                "error": error.loc[idx],
                "absolute_error": abs(error.loc[idx]),
                "tolerance": tolerance
            })

    return pd.DataFrame(rows)


def auto_detect_reference_pairs(df, numeric_columns):

    if not numeric_columns or len(numeric_columns) < 2:
        return {}

    cols_lower = {str(c).lower(): c for c in numeric_columns}
    for col in numeric_columns:
        cl = str(col).lower()
        if "apparent" in cl:
            base_name = cl.replace("apparent", "").strip()
            for other_cl, orig in cols_lower.items():
                if other_cl != cl and base_name in other_cl:
                    return {orig: {"reference_col": col, "tolerance": 0.5}}
        elif "reference" in cl:
            base_name = cl.replace("reference", "").strip()
            for other_cl, orig in cols_lower.items():
                if other_cl != cl and base_name in other_cl:
                    return {orig: {"reference_col": col, "tolerance": 0.5}}

    num_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr().abs()
    best_pair = None
    best_r = 0.0
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            c1, c2 = numeric_columns[i], numeric_columns[j]
            r = corr.loc[c1, c2]
            if pd.notna(r) and r > best_r:
                best_r = r
                best_pair = (c1, c2)

    if best_pair and best_r >= 0.4:
        c1, c2 = best_pair
        diff = (num_df[c1] - num_df[c2]).dropna()
        std_diff = diff.std()
        tol = round(float(max(0.01, 0.25 * (std_diff if pd.notna(std_diff) and std_diff > 0 else 1.0))), 2)
        return {c1: {"reference_col": c2, "tolerance": tol}}

    c1, c2 = numeric_columns[0], numeric_columns[1]
    diff = (num_df[c1] - num_df[c2]).dropna()
    std_diff = diff.std()
    tol = round(float(max(0.01, 0.25 * (std_diff if pd.notna(std_diff) and std_diff > 0 else 1.0))), 2)
    return {c1: {"reference_col": c2, "tolerance": tol}}


def _plot_reference_noise(df, measurement_col, reference_col, tolerance, output_dir):

    if measurement_col not in df.columns or reference_col not in df.columns:
        return None

    measurement = pd.to_numeric(df[measurement_col], errors="coerce")
    reference = pd.to_numeric(df[reference_col], errors="coerce")

    valid = pd.DataFrame({"measurement": measurement, "reference": reference}).dropna()
    if len(valid) < 2:
        return None

    error = valid["measurement"] - valid["reference"]
    possible_noise = error.abs() > tolerance

    normal = valid[~possible_noise]
    flagged = valid[possible_noise]

    # Sample if large for snappy rendering
    if len(valid) > 12000:
        norm_sample = normal.sample(min(len(normal), 8000), random_state=42) if not normal.empty else normal
        flag_sample = flagged.sample(min(len(flagged), 4000), random_state=42) if not flagged.empty else flagged
    else:
        norm_sample = normal
        flag_sample = flagged

    min_val = min(valid["measurement"].min(), valid["reference"].min())
    max_val = max(valid["measurement"].max(), valid["reference"].max())
    padding = (max_val - min_val) * 0.05
    if padding == 0:
        padding = 1.0
    lower = min_val - padding
    upper = max_val + padding

    fig, ax = plt.subplots(figsize=(9, 7))
    if not norm_sample.empty:
        ax.scatter(
            norm_sample["reference"], norm_sample["measurement"],
            color="#10b981", alpha=0.55, s=36,
            label=f"Dans la tolérance : {len(normal):,} ({len(normal)/len(valid)*100:.1f}%)"
        )
    if not flag_sample.empty:
        ax.scatter(
            flag_sample["reference"], flag_sample["measurement"],
            color="#ef4444", alpha=0.85, s=50, marker="x",
            label=f"Écart > {tolerance} (Bruit : {len(flagged):,} - {len(flagged)/len(valid)*100:.1f}%)"
        )

    ax.plot([lower, upper], [lower, upper], color="#f8fafc", linestyle="--", linewidth=1.5, label="Accord Parfait (y = x)")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"Variable de Référence : {reference_col}", fontsize=11, color="#cbd5e1", fontweight="600")
    ax.set_ylabel(f"Variable de Mesure : {measurement_col}", fontsize=11, color="#cbd5e1", fontweight="600")
    ax.set_title(f"Détection du Bruit de Mesure (detect_reference_noise)\n{measurement_col} vs {reference_col} (Tolérance = ±{tolerance})", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    path = Path(output_dir) / "reference_noise.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return str(path)


def analyze_timeseries_noise(df, date_column, value_column, window=5):

    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column}")
    if value_column not in df.columns:
        raise ValueError(f"Missing value column: {value_column}")

    data = df[[date_column, value_column]].copy()
    data[date_column] = to_datetime_safe(data[date_column])
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")

    data = data.dropna().sort_values(date_column).reset_index(drop=True)
    if len(data) < window:
        raise ValueError("Not enough valid observations for this window.")

    data["estimated_signal"] = (
        data[value_column]
        .rolling(window=window, center=True, min_periods=1)
        .median()
    )
    data["residual"] = data[value_column] - data["estimated_signal"]
    return data


# 8) OUTLIERS (IQR METHOD)

def outliers_summary(df, numeric_columns):

    rows = []
    for col in numeric_columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(((values < lower) | (values > upper)).sum())
        pct = round(count / len(values) * 100, 2)

        rows.append({
            "column": col,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "lower_bound": round(lower, 3),
            "upper_bound": round(upper, 3),
            "outlier_count": count,
            "outlier_percent": pct,
            "outlier_percentage": pct
        })

    return pd.DataFrame(rows)


# 9) CLASS BALANCE

def class_balance(df, target_class, output_dir=None, save_plot=True):

    if target_class not in df.columns:
        return None

    counts = df[target_class].value_counts(dropna=False)
    percentages = (df[target_class].value_counts(dropna=False, normalize=True) * 100).round(2)

    balance = pd.DataFrame({
        "count": counts,
        "percent": percentages
    })

    if save_plot and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_data = df[target_class].astype("object").fillna("Missing")
        class_counts = plot_data.value_counts().head(MAX_CLASSES_SHOWN)

        fig, ax = plt.subplots(figsize=(8, 4))
        class_counts.plot(kind="bar", ax=ax, color="#2563eb", edgecolor="#1e3a8a")
        ax.set_title(f"Class Distribution — {target_class}", fontsize=12, fontweight="bold")
        ax.set_xlabel(target_class)
        ax.set_ylabel("Number of observations")
        ax.tick_params(axis="x", rotation=0 if len(class_counts) <= 6 else 45)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()

        fig.savefig(output_dir / "class_balance.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    return balance


# 10) CORRELATION MATRIX

def correlation_matrix(df):

    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr()


# 11) FEATURE SELECTION

def feature_selection(df, target_class=None, variance_threshold=0.0, correlation_threshold=0.95):

    numeric_df = df.select_dtypes(include="number").copy()

    if target_class in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_class])

    report = []
    for col in numeric_df.columns:
        variance = numeric_df[col].var()
        is_candidate = pd.notna(variance) and variance > variance_threshold
        report.append({
            "feature": col,
            "variance": variance,
            "selected": is_candidate,
            "reason": "Candidate" if is_candidate else "Zero or undefined variance"
        })

    result = pd.DataFrame(report)
    if result.empty:
        return result

    selected = result.loc[result["selected"], "feature"].tolist()
    if len(selected) > 1:
        corr = numeric_df[selected].corr().abs()
        for i, col in enumerate(selected):
            if not result.loc[result["feature"] == col, "selected"].iloc[0]:
                continue
            for other in selected[i + 1:]:
                if (
                    result.loc[result["feature"] == other, "selected"].iloc[0]
                    and corr.loc[col, other] > correlation_threshold
                ):
                    result.loc[result["feature"] == other, "selected"] = False
                    result.loc[result["feature"] == other, "reason"] = f"Highly correlated with {col}"

    result["reason"] = result.apply(
        lambda row: (
            "Selected"
            if row["selected"] and row["reason"] == "Candidate"
            else row["reason"]
        ),
        axis=1
    )
    return result


# 12) PLOT GENERATION

def plot_distribution(df, column, output_dir, bins=30):

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(values, bins=bins, ax=axes[0], color="#0284c7")
    axes[0].set_title(f"Histogram: {column}", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    if values.nunique() > 1:
        try:
            sns.kdeplot(values, fill=True, ax=axes[1], color="#38bdf8")
            axes[1].set_title(f"KDE Density: {column}", fontsize=11)
            axes[1].grid(True, linestyle="--", alpha=0.3)
        except Exception:
            axes[1].text(0.5, 0.5, "KDE unavailable", ha="center")
    else:
        axes[1].text(0.5, 0.5, "KDE unavailable: constant values", ha="center", va="center")
        axes[1].set_title(f"KDE: {column}", fontsize=11)

    fig.tight_layout()
    path = output_dir / f"distribution_{safe_name(column)}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_boxplot(plot_df, col, output_dir):
    plt.figure(figsize=(8, 3))
    sns.boxplot(data=plot_df, x=col, color="#38bdf8")
    plt.title(f"Boxplot - {col}", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / f"boxplot_{safe_name(col)}.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return str(path)


def _plot_smoothed(df, date_col, col, output_dir):
    sample = df.iloc[:SMOOTH_ROWS].copy()
    sample["Smoothed"] = sample[col].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
    plt.figure(figsize=(10, 4.5))
    plt.plot(sample[date_col], sample[col], label="Original", alpha=0.6, color="#0ea5e9")
    plt.plot(sample[date_col], sample["Smoothed"], label=f"Rolling mean ({SMOOTH_WINDOW} obs)",
             linewidth=2, color="#f59e0b")
    plt.title(f"{col}: Original vs Smoothed Trend", fontsize=12, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = Path(output_dir) / f"smoothed_{safe_name(col)}.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return str(path)


def _plot_categorical(df, col, output_dir):
    vc = df[col].fillna("Missing").astype(str).value_counts().head(15)
    plt.figure(figsize=(8, max(3, 0.35 * len(vc) + 1.2)))
    vc[::-1].plot(kind="barh", color="#2563eb", edgecolor="#1e3a8a")
    plt.title(f"Top values - {col}", fontsize=12, fontweight="bold")
    plt.xlabel("Number of observations")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / f"categorical_{safe_name(col)}.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return str(path)


def _plot_missing(missing_report, output_dir):
    data = missing_report[missing_report["Missing_Count"] > 0]["Missing_Percentage"]
    data = data.sort_values(ascending=False).head(30)
    if data.empty:
        return None
    plt.figure(figsize=(8, max(3, 0.3 * len(data) + 1.2)))
    data[::-1].plot(kind="barh", color="#ef4444", edgecolor="#7f1d1d")
    plt.title("Missing values (%)", fontsize=12, fontweight="bold")
    plt.xlabel("% of missing values")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "missing_values.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return str(path)


def _try(notes, label, func, *args):
    try:
        func(*args)
    except Exception as e:
        plt.close("all")
        notes.append(f"{label} skipped: {type(e).__name__}: {e}")


def _plot_correlation(corr, output_dir):
    if corr.empty:
        return None
    n = len(corr)
    size = min(max(7, 0.6 * n), 12)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(
        corr,
        annot=n <= HEATMAP_ANNOT_MAX,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        ax=ax
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = Path(output_dir) / "correlation_heatmap.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
    return str(path)



def generate_eda_plots(df, numeric_columns, output_dir):
 
    paths = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric_columns[:MAX_NUMERIC_PLOTS]:
        path = plot_distribution(df, col, output_dir)
        if path:
            paths.append(path)

    missing_rep = missing_values_report(df)
    miss_path = _plot_missing(missing_rep, output_dir)
    if miss_path:
        paths.append(miss_path)

    corr = correlation_matrix(df)
    corr_path = _plot_correlation(corr, output_dir)
    if corr_path:
        paths.append(corr_path)

    return paths


# 13) SAVE REPORT TABLES

def save_table(table, output_dir_or_path, filename=None):

    if filename is not None:
        target_path = Path(output_dir_or_path) / filename
    else:
        target_path = Path(output_dir_or_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(table, dict):
        for key in ["numerical_stats", "duplicate_rows", "table", "Missing_Count"]:
            if key in table and isinstance(table[key], pd.DataFrame):
                table = table[key]
                break
        else:
            try:
                table = pd.DataFrame(table)
            except Exception:
                table = pd.DataFrame([table])

    if hasattr(table, "to_csv"):
        table.to_csv(target_path, index=True)


# ======================================================================
# 14) SQLITE STORAGE  (replaces the old CSV / summary.json files)
# ======================================================================
#
# Layout of  results/analysis.db :
#   datasets     (name, summary JSON, created_at)           one row per analysed dataset
#   tables_meta  (dataset, kind, table_name, sql_table, ...) catalogue of the stored tables
#   t__<dataset>__<table>   result tables (head, describe_numeric, correlation, ...)
#   o__<dataset>__<column>  one table of IQR outliers per numeric column
# Plots (PNG) are still written to results/<dataset>/plots/.

INDEX_COL = "__index__"      # column used to store a DataFrame index
DUP_SUFFIX = "__dup"         # SQLite column names are case-insensitive ("Missing_Count" == "missing_count")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    name       TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tables_meta (
    dataset      TEXT NOT NULL,
    kind         TEXT NOT NULL,              -- 'table' or 'outlier'
    table_name   TEXT NOT NULL,
    sql_table    TEXT NOT NULL,
    has_index    INTEGER NOT NULL DEFAULT 0,
    index_name   TEXT,
    bool_columns TEXT NOT NULL DEFAULT '[]',
    n_rows       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (dataset, kind, table_name)
);
"""


def get_db_path(results_dir=RESULTS_DIR):
    return Path(results_dir) / DB_NAME


def _connect(results_dir=RESULTS_DIR):
    path = get_db_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # manual transactions
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.executescript(_SCHEMA)
    except sqlite3.OperationalError:
        pass
    return conn


def _connect_readonly(results_dir=RESULTS_DIR):
    """Connection for readers: returns None if the database does not exist yet."""
    if not get_db_path(results_dir).exists():
        return None
    return _connect(results_dir)


def _quote(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def _dedupe_columns(columns):
    seen, result = set(), []
    for c in (str(c) for c in columns):
        name = c
        while name.lower() in seen:
            name += DUP_SUFFIX
        seen.add(name.lower())
        result.append(name)
    return result


def _restore_column(name):
    while name.endswith(DUP_SUFFIX):
        name = name[: -len(DUP_SUFFIX)]
    return name


def _py_value(v):
    """Convert one cell to something sqlite3 can bind."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, int, str, bytes)):
        return v
    if isinstance(v, float):
        return None if v != v else v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if f != f else f
    return str(v)


def _store_table(conn, dataset, kind, name, df, has_index):
    """Create + fill one SQLite table from a DataFrame and register it in tables_meta."""
    df = df.copy()
    index_name = None
    if has_index:
        index_name = None if df.index.name is None else str(df.index.name)
        idx_values = list(df.index)
        df = df.reset_index(drop=True)
        df.insert(0, INDEX_COL, idx_values)

    df.columns = _dedupe_columns(df.columns)

    bool_cols, col_defs = [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            df[c] = s.map(lambda v: v.isoformat(sep=" ") if pd.notna(v) else None)
            s = df[c]
        if pd.api.types.is_bool_dtype(s):
            bool_cols.append(c)
            col_defs.append(f"{_quote(c)} INTEGER")
        elif pd.api.types.is_integer_dtype(s):
            col_defs.append(f"{_quote(c)} INTEGER")
        elif pd.api.types.is_float_dtype(s):
            col_defs.append(f"{_quote(c)} REAL")
        else:
            col_defs.append(_quote(c))      # no declared type: keeps numbers as numbers, text as text

    prefix = "t" if kind == "table" else "o"
    sql_table = f"{prefix}__{dataset}__{name}"
    # SQLite table names are case-insensitive: compare in lower case
    taken = {r[0].lower() for r in conn.execute("SELECT sql_table FROM tables_meta")}
    base, n = sql_table, 2
    while sql_table.lower() in taken:
        sql_table, n = f"{base}_{n}", n + 1

    conn.execute(f"CREATE TABLE {_quote(sql_table)} ({', '.join(col_defs)})")

    if len(df):
        cleaned = df.astype(object).where(df.notna(), None)
        rows = [tuple(_py_value(v) for v in row) for row in cleaned.itertuples(index=False, name=None)]
        placeholders = ",".join("?" * len(df.columns))
        conn.executemany(f"INSERT INTO {_quote(sql_table)} VALUES ({placeholders})", rows)

    conn.execute(
        "INSERT INTO tables_meta (dataset, kind, table_name, sql_table, has_index, index_name, bool_columns, n_rows) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (dataset, kind, name, sql_table, int(bool(has_index)), index_name,
         json.dumps(bool_cols), int(len(df))),
    )


def _delete_dataset(conn, dataset):
    for (sql_table,) in conn.execute("SELECT sql_table FROM tables_meta WHERE dataset=?", (dataset,)).fetchall():
        conn.execute(f"DROP TABLE IF EXISTS {_quote(sql_table)}")
    conn.execute("DELETE FROM tables_meta WHERE dataset=?", (dataset,))
    conn.execute("DELETE FROM datasets WHERE name=?", (dataset,))


def delete_dataset(dataset_name, results_dir=RESULTS_DIR):
    """
    Delete a dataset's analysis: its SQLite tables, plots, reports and cached files (results/<name>/).
    The original CSV/Excel file is NOT touched, so the dataset can be re-analysed later.
    Returns True if it existed, False otherwise.
    """
    dataset_name = clean_dataset_name(dataset_name)
    results_dir = Path(results_dir)
    if dataset_name not in list_datasets(results_dir):
        return False
    conn = _connect(results_dir)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _delete_dataset(conn, dataset_name)
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    root = results_dir.resolve()
    for folder in (results_dir / dataset_name, results_dir / f".tmp_{dataset_name}"):
        if folder.exists() and folder.resolve().parent == root:      # never delete outside results/
            shutil.rmtree(folder, ignore_errors=True)
    return True


def save_results_to_db(dataset_name, summary, tables, outlier_tables, results_dir=RESULTS_DIR):
    """
    Save a whole analysis in ONE transaction (all-or-nothing, replaces any previous run).

    tables         : {table_name: (DataFrame, keep_index)}
    outlier_tables : {column_key: DataFrame}
    """
    dataset_name = clean_dataset_name(dataset_name)
    conn = _connect(results_dir)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _delete_dataset(conn, dataset_name)
        for name, (frame, keep_index) in tables.items():
            _store_table(conn, dataset_name, "table", name, frame, keep_index)
        for key, frame in outlier_tables.items():
            _store_table(conn, dataset_name, "outlier", key, frame, False)
        conn.execute(
            "INSERT INTO datasets (name, summary, created_at) VALUES (?,?,datetime('now'))",
            (dataset_name, json.dumps(summary, ensure_ascii=False, default=str)),
        )
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _read_table(conn, sql_table, has_index, index_name, bool_cols, limit=None):
    query = f"SELECT * FROM {_quote(sql_table)}"
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)
    df = pd.read_sql_query(query, conn, params=params)
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    if has_index and INDEX_COL in df.columns:
        df = df.set_index(INDEX_COL)
        df.index.name = index_name
    df.columns = [_restore_column(c) for c in df.columns]
    return df


def _meta_rows(conn, dataset, kind):
    return conn.execute(
        "SELECT table_name, sql_table, has_index, index_name, bool_columns, n_rows "
        "FROM tables_meta WHERE dataset=? AND kind=? ORDER BY table_name",
        (dataset, kind),
    ).fetchall()


# ----------------------------------------------------------------------
# READING RESULTS (WEB APPLICATION INTEGRATION)
# ----------------------------------------------------------------------

def list_datasets(results_dir=RESULTS_DIR):
    conn = _connect_readonly(results_dir)
    if conn is None:
        return []
    try:
        return [r[0] for r in conn.execute("SELECT name FROM datasets ORDER BY name")]
    finally:
        conn.close()


def load_summary(dataset_name, results_dir=RESULTS_DIR):
    """The summary dict of a dataset, or None if it does not exist."""
    dataset_name = clean_dataset_name(dataset_name)
    conn = _connect_readonly(results_dir)
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT summary FROM datasets WHERE name=?", (dataset_name,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception as e:
        return {"dataset": dataset_name, "error": str(e)}


def load_dataset_results(dataset_name, results_dir=RESULTS_DIR):
    dataset_name = clean_dataset_name(dataset_name)
    summary = load_summary(dataset_name, results_dir)
    if summary is None:
        return None

    base = Path(results_dir) / dataset_name

    all_columns = list(summary.get("dtypes", {}).keys())
    pretty = {safe_name(c): c for c in all_columns}

    def pretty_name(key):
        return pretty.get(key, key.replace("__", " ").replace("_", " "))

    tables, outliers_columns = {}, []
    conn = _connect(results_dir)
    try:
        for name, sql_table, has_index, index_name, bool_cols, _n in _meta_rows(conn, dataset_name, "table"):
            try:
                tables[name] = _read_table(conn, sql_table, bool(has_index), index_name, json.loads(bool_cols))
            except Exception as e:
                print(f"Error reading table {name}: {e}")

        for key, _sql, _hi, _in, _bc, n_rows in _meta_rows(conn, dataset_name, "outlier"):
            outliers_columns.append({
                "key": key,
                "display": pretty_name(key),
                "n_rows": n_rows,
                "size_bytes": 0,        # no file any more; use n_rows
            })
    finally:
        conn.close()

    # Ensure n_missing_cells and missing_cell_percent are always present
    if "n_missing_cells" not in summary:
        total_missing = 0
        if "missing_report" in tables:
            m_df = tables["missing_report"]
            col_m = "Missing_Count" if "Missing_Count" in m_df.columns else ("missing_count" if "missing_count" in m_df.columns else None)
            if col_m:
                total_missing = int(m_df[col_m].sum())
        total_cells = max(1, int(summary.get("n_rows", 0) * summary.get("n_columns", 1)))
        summary["n_missing_cells"] = total_missing
        summary["missing_cell_percent"] = round((total_missing / total_cells) * 100, 2)

    plots = []
    plots_dir = base / "plots"
    if plots_dir.exists():
        for p in sorted(plots_dir.glob("*.png")):
            filename = p.name
            category, feature, title = "other", "", filename[:-4]

            if filename.startswith("distribution_"):
                category, feature = "distribution", filename[len("distribution_"):-4]
                title = f"Distribution (Hist + KDE) - {pretty_name(feature)}"
            elif filename.startswith("boxplot_"):
                category, feature = "boxplot", filename[len("boxplot_"):-4]
                title = f"Boxplot - {pretty_name(feature)}"
            elif filename.startswith("smoothed_"):
                category, feature = "smoothed", filename[len("smoothed_"):-4]
                title = f"Smoothed Trend - {pretty_name(feature)}"
            elif filename.startswith("categorical_"):
                category, feature = "categorical", filename[len("categorical_"):-4]
                title = f"Top values - {pretty_name(feature)}"
            elif filename in ("correlation_heatmap.png", "correlation_matrix.png"):
                category, title = "correlation", "Pearson Correlation Heatmap"
            elif filename == "class_balance.png":
                category, title = "class_balance", "Target Class Balance"
            elif filename == "missing_values.png":
                category, title = "missing", "Missing Values per Column"
            elif filename == "reference_noise.png" or filename.startswith("reference_noise_"):
                category, title = "reference_noise", "Analyse du Bruit de Référence (detect_reference_noise)"

            plots.append({
                "filename": filename,
                "url": f"/results/{dataset_name}/plots/{filename}",
                "category": category,
                "title": title,
                "feature": feature,
            })

    return {
        "dataset": dataset_name,
        "summary": summary,
        "tables": tables,
        "plots": plots,
        "outliers_columns": outliers_columns,
    }


def get_outliers_sample(dataset_name, column_key, limit=50, results_dir=RESULTS_DIR):
    dataset_name = clean_dataset_name(dataset_name)
    column_key = Path(column_key).name
    limit = max(1, min(int(limit), 1000))

    empty = {"error": "Column outliers not found", "rows": [], "total": 0}
    conn = _connect_readonly(results_dir)
    if conn is None:
        return empty
    try:
        meta = conn.execute(
            "SELECT sql_table, n_rows FROM tables_meta WHERE dataset=? AND kind='outlier' AND table_name=?",
            (dataset_name, column_key),
        ).fetchone()
        if meta is None:
            return empty
        sql_table, total = meta
        sample_df = _read_table(conn, sql_table, False, None, [], limit=limit)
    finally:
        conn.close()

    sample_df = sample_df.astype(object).where(sample_df.notna(), None)
    return {
        "column": column_key,
        "total": int(total),
        "limit": limit,
        "columns": list(sample_df.columns),
        "rows": sample_df.to_dict(orient="records"),
    }


def export_table_csv(dataset_name, table_name, results_dir=RESULTS_DIR):
    """Return a stored result table as CSV text (for the download button), or None."""
    dataset_name = clean_dataset_name(dataset_name)
    conn = _connect_readonly(results_dir)
    if conn is None:
        return None
    try:
        meta = conn.execute(
            "SELECT sql_table, has_index, index_name, bool_columns FROM tables_meta "
            "WHERE dataset=? AND kind='table' AND table_name=?",
            (dataset_name, table_name),
        ).fetchone()
        if meta is None:
            return None
        sql_table, has_index, index_name, bool_cols = meta
        df = _read_table(conn, sql_table, bool(has_index), index_name, json.loads(bool_cols))
    finally:
        conn.close()
    return df.to_csv(index=bool(has_index))


def migrate_legacy_results(results_dir=RESULTS_DIR):
    """
    One-off import of datasets analysed by the old CSV version (results/<name>/summary.json + *.csv)
    into the SQLite database. Datasets already in the database are skipped; the old files are left untouched.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    existing = set(list_datasets(results_dir))
    migrated = []
    for base in sorted(results_dir.iterdir()):
        if (not base.is_dir() or base.name.startswith(".") or base.name in existing
                or not (base / "summary.json").exists()):
            continue
        try:
            summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
            tables, outliers = {}, {}
            for f in sorted(base.glob("*.csv")):
                keep_index = f.stem in TABLES_WITH_INDEX
                try:
                    frame = pd.read_csv(f, index_col=0 if keep_index else None)
                except pd.errors.EmptyDataError:
                    continue
                if not keep_index and "Unnamed: 0" in frame.columns:
                    frame = frame.drop(columns=["Unnamed: 0"])
                tables[f.stem] = (frame, keep_index)
            if (base / "outliers").exists():
                for f in sorted((base / "outliers").glob("*.csv")):
                    try:
                        outliers[f.stem] = pd.read_csv(f)
                    except pd.errors.EmptyDataError:
                        continue
            save_results_to_db(base.name, summary, tables, outliers, results_dir)
            migrated.append(base.name)
        except Exception as e:
            print(f"Could not migrate '{base.name}': {type(e).__name__}: {e}")
    return migrated


# ======================================================================
# 15) MAIN PIPELINE & EXECUTION (SAVING PROCESS OF ANALYSIS_ENGINE)
# ======================================================================

def run_full_analysis(
    file_path,
    dataset_name=None,
    date_column=AUTO,
    target_class=AUTO,
    results_dir=RESULTS_DIR,
    force_timeseries=False,
    reference_pairs=None,
    config=None
):

    t0 = time.time()
    file_path = Path(file_path)
    dataset_name = clean_dataset_name(dataset_name or file_path.stem)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    final_out = results_dir / dataset_name
    out = results_dir / f".tmp_{dataset_name}"

    if out.exists():
        shutil.rmtree(out)

    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True)

    notes = []
    tables = {}           # table name -> (DataFrame, keep_index), written to SQLite at the end
    outlier_tables = {}   # column key  -> DataFrame

    def put(name, frame, keep_index=None):
        if keep_index is None:
            keep_index = name in TABLES_WITH_INDEX
        tables[name] = (frame, keep_index)

    try:
        # --------------------------------------------------
        # Step 1: Load dataset
        # --------------------------------------------------
        df = load_data(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty or df.shape[1] == 0:
            raise ValueError("The dataset file is empty.")

        # --------------------------------------------------
        # Step 2: Datetime detection & Dataset type classification
        # --------------------------------------------------
        resolved_date_col = None
        date_auto = False
        if date_column is not None and str(date_column).strip().lower() not in ("none", ""):
            req_date = str(date_column).strip()
            if req_date.lower() != AUTO:
                if req_date in df.columns:
                    resolved_date_col = req_date
                else:
                    notes.append(f"Specified date_column '{req_date}' not found. Automatic detection used.")
                    date_auto = True
            else:
                date_auto = True

        type_classification = detect_dataset_type(
            df,
            date_column=resolved_date_col,
            force_timeseries=force_timeseries
        )
        date_col = type_classification["date_column"]

        # Parse and sort by date column if applicable
        if date_col is not None and date_col in df.columns:
            parsed = to_datetime_safe(df[date_col])
            if parsed.notna().mean() >= 0.5:
                df[date_col] = parsed
                df = df.sort_values(date_col)
            else:
                notes.append(f"Date column '{date_col}' had low parse success (<50%). Not converted.")

        # --------------------------------------------------
        # Step 3: Target variable (automatic detection disabled)
        # --------------------------------------------------
        target = None
        if target_class is not None and str(target_class).strip().lower() not in ("none", "", AUTO):
            req_target = str(target_class).strip()
            if req_target in df.columns:
                target = req_target

        # --------------------------------------------------
        # Step 4: Dataset Description (Numerical & Categorical)
        # --------------------------------------------------
        desc = describe_data(df)
        numeric_columns = desc["numeric_columns"]
        categorical_columns = [c for c in desc["categorical_columns"] if c != date_col]

        # Save head sample
        put("head", df.head(20))

        # Save descriptive statistics tables
        if not desc["numerical_stats"].empty:
            put("describe_numeric", desc["numerical_stats"])
        if not desc["categorical_stats"].empty:
            put("describe_categorical", desc["categorical_stats"])
        put("n_unique", desc["unique_counts"])

        # --------------------------------------------------
        # Step 5: Missing values report
        # --------------------------------------------------
        missing_rep = missing_values_report(df)
        put("missing_report", missing_rep)
        _plot_missing(missing_rep, plots_dir)

        # --------------------------------------------------
        # Step 6: Duplicates report
        # --------------------------------------------------
        dup_rep = duplicates_report(df)
        n_duplicates = dup_rep["duplicate_count"]
        if not dup_rep["duplicate_rows"].empty:
            put("duplicates_sample", dup_rep["duplicate_rows"].head(50))
            put("duplicates", dup_rep["duplicate_rows"])
        else:
            put("duplicates_sample", pd.DataFrame(columns=df.columns))
            put("duplicates", pd.DataFrame(columns=df.columns))

        # --------------------------------------------------
        # Step 7: Column roles & Outlier summary (IQR)
        # --------------------------------------------------
        constant_columns = [c for c in numeric_columns if df[c].nunique(dropna=True) <= 1]
        id_like_columns = [c for c in numeric_columns if is_id_like(df, c)]
        usable_numeric = [c for c in numeric_columns if c not in constant_columns and c not in id_like_columns]
        plot_numeric = usable_numeric[:MAX_NUMERIC_PLOTS]

        outlier_rep = outliers_summary(df, usable_numeric[:MAX_OUTLIER_COLUMNS])
        if not outlier_rep.empty:
            put("outliers_summary", outlier_rep.sort_values("outlier_count", ascending=False))

        # Save individual outlier slices for deep-dive inspection
        for col in usable_numeric[:MAX_OUTLIER_COLUMNS]:
            values = pd.to_numeric(df[col], errors="coerce")
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (values < lower) | (values > upper)
            subset = df.loc[mask, [col] + ([date_col] if date_col and date_col in df.columns else [])].copy()
            subset.insert(0, "row", subset.index)
            outlier_tables[safe_name(col)] = subset

        # --------------------------------------------------
        # Step 8: Class balance
        # --------------------------------------------------
        balance_summary = None
        if target and target in df.columns:
            balance = class_balance(df, target, output_dir=plots_dir, save_plot=True)
            if balance is not None:
                put("class_balance", balance)
                balance_summary = balance.head(MAX_CLASSES_SHOWN).reset_index().to_dict(orient="records")

        # --------------------------------------------------
        # Step 9: Correlation matrix
        # --------------------------------------------------
        corr = correlation_matrix(df[plot_numeric]) if len(plot_numeric) >= 2 else pd.DataFrame()
        if not corr.empty:
            put("correlation", corr.round(3))
            _plot_correlation(corr, plots_dir)

        # --------------------------------------------------
        # Step 10: Feature selection (variance + correlation filtering)
        # --------------------------------------------------
        selection_report = feature_selection(
            df,
            target_class=target,
            variance_threshold=0.0,
            correlation_threshold=0.95
        )
        if not selection_report.empty:
            put("feature_selection", selection_report)
            selected_features = selection_report.loc[selection_report["selected"], "feature"].tolist()
        else:
            selected_features = []

        # --------------------------------------------------
        # Step 11: Noise analysis (reference / time-series)
        # --------------------------------------------------
        if not reference_pairs and len(numeric_columns) >= 2:
            reference_pairs = auto_detect_reference_pairs(df, numeric_columns)

        ref_noise_info = None
        if reference_pairs:
            ref_noise = detect_reference_noise(df, reference_pairs)
            if not ref_noise.empty:
                put("reference_noise", ref_noise.head(1000))

            for m_col, cfg in reference_pairs.items():
                r_col = cfg.get("reference_col")
                tol = cfg.get("tolerance", 0.5)
                try:
                    _plot_reference_noise(df, m_col, r_col, tol, plots_dir)
                    m_s = pd.to_numeric(df[m_col], errors="coerce")
                    r_s = pd.to_numeric(df[r_col], errors="coerce")
                    valid_mask = m_s.notna() & r_s.notna()
                    total_comp = int(valid_mask.sum())
                    n_flagged = int((valid_mask & ((m_s - r_s).abs() > tol)).sum())
                    ref_noise_info = {
                        "measurement_col": m_col,
                        "reference_col": r_col,
                        "tolerance": tol,
                        "total_comparisons": total_comp,
                        "normal_count": total_comp - n_flagged,
                        "noise_count": n_flagged,
                        "noise_percent": round(n_flagged / max(1, total_comp) * 100, 2)
                    }
                except Exception as e:
                    notes.append(f"Reference noise plot failed: {e}")
                break

        if type_classification["dataset_type"] == "time_series" and date_col and plot_numeric:
            try:
                ts_noise = analyze_timeseries_noise(df, date_col, plot_numeric[0])
                put("timeseries_noise_sample", ts_noise.head(500))
            except Exception:
                pass

        # --------------------------------------------------
        # Step 12: Visualizations (distributions, boxplots, smoothed)
        # --------------------------------------------------
        plot_df = df
        if len(df) > MAX_PLOT_ROWS:
            plot_df = df.sample(MAX_PLOT_ROWS, random_state=0)
            notes.append(f"Visual plots sampled {MAX_PLOT_ROWS:,} rows of {len(df):,} total.")

        for col in plot_numeric:
            try:
                plot_distribution(plot_df, col, plots_dir)
            except Exception as e:
                notes.append(f"Distribution plot failed for '{col}': {e}")

            try:
                _plot_boxplot(plot_df, col, plots_dir)
            except Exception as e:
                notes.append(f"Boxplot failed for '{col}': {e}")

            if date_col is not None and date_col in df.columns:
                try:
                    _plot_smoothed(df, date_col, col, plots_dir)
                except Exception as e:
                    notes.append(f"Smoothed trend failed for '{col}': {e}")

        # Top categorical plots
        high_cardinality = []
        cat_rows = []
        for col in categorical_columns:
            n_u = int(df[col].nunique(dropna=True))
            if n_u > MAX_CATEGORY_CARDINALITY:
                high_cardinality.append(col)
                continue
            if n_u == 0:
                continue
            vc = df[col].value_counts()
            cat_rows.append({
                "column": col,
                "n_unique": n_u,
                "top_value": str(vc.index[0]),
                "top_count": int(vc.iloc[0]),
                "top_percent": round(vc.iloc[0] / len(df) * 100, 2),
                "missing_percent": round(df[col].isna().mean() * 100, 2),
            })
            if col != target and len(cat_rows) <= MAX_CATEGORICAL_PLOTS:
                try:
                    _plot_categorical(df, col, plots_dir)
                except Exception:
                    pass

        if cat_rows:
            put("categorical_summary", pd.DataFrame(cat_rows))

        # --------------------------------------------------
        # Step 13: Summary + save everything to SQLite (written last)
        # --------------------------------------------------
        summary = {
            "dataset": dataset_name,
            "source_file": str(file_path),
            "file": Path(file_path).name,
            "shape": list(df.shape),
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "dataset_type": type_classification["dataset_type"],
            "dataset_type_details": type_classification,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "n_duplicates": n_duplicates,
            "n_missing_cells": int(df.isna().sum().sum()),
            "date_column": date_col,
            "date_column_auto_detected": bool(date_col and date_auto),
            "datetime_candidates": type_classification.get("datetime_candidates", []),
            "target_class": target,
            "class_balance": balance_summary,
            "selected_features": selected_features,
            "constant_columns": constant_columns,
            "id_like_columns": id_like_columns,
            "high_cardinality_columns": high_cardinality,
            "plotted_numeric_columns": plot_numeric,
            "reference_noise_info": ref_noise_info,
            "notes": notes,
            "analysis_seconds": round(time.time() - t0, 2),
        }

        # Everything (summary + tables + outliers) goes to SQLite in one transaction
        save_results_to_db(dataset_name, summary, tables, outlier_tables, results_dir)

        # --------------------------------------------------
        # Atomic swap to final directory
        # --------------------------------------------------
        if final_out.exists():
            shutil.rmtree(final_out)
        out.rename(final_out)

    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise

    return final_out


def run_pipeline(
    file_path,
    target_class=None,
    output_root=RESULTS_DIR,
    force_timeseries=False,
    date_column=None,
    config=None,
    reference_pairs=None
):
    """
    Direct alias / pipeline interface matching traitement.py conventions,
    saving into output_root via run_full_analysis.
    """
    clean_stem = Path(file_path).stem.replace(" ", "_")
    output_dir = run_full_analysis(
        file_path=file_path,
        dataset_name=clean_stem,
        date_column=date_column,
        target_class=target_class,
        results_dir=output_root,
        force_timeseries=force_timeseries,
        reference_pairs=reference_pairs,
        config=config
    )
    summary = load_summary(output_dir.name, output_root) or {}
    return {
        "cached": False,
        "output_dir": str(output_dir),
        "summary": summary
    }


# ==============================================================================
# INTERACTIVE NOISE SERIES (used by the Noise page: /api/noise-series/...)
# Computed on demand because the user changes year / bounds / window in the browser.
# ==============================================================================

_NOISE_CACHE = {}
_TZ_SUFFIX = re.compile(r"\s*(?:Z|[+-]\d{2}:?\d{2})$")


def _noise_dates(series):
    """Parse like to_datetime_safe but keep the local calendar time (drop '+0200' style offsets)."""
    if not pd.api.types.is_datetime64_any_dtype(series) and not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.replace(_TZ_SUFFIX, "", regex=True)
    parsed = None
    if not pd.api.types.is_datetime64_any_dtype(series) and not pd.api.types.is_numeric_dtype(series):
        # format="mixed" is very slow on 100k+ rows: use one explicit format when the column has one
        try:
            from pandas.tseries.api import guess_datetime_format
            first = series.dropna().astype(str).iloc[0]
            fmt = guess_datetime_format(first)
            if fmt:
                fast = pd.to_datetime(series, format=fmt, errors="coerce")
                if fast.notna().mean() >= 0.9:
                    parsed = fast
        except Exception:
            parsed = None
    if parsed is None:
        parsed = to_datetime_safe(series)
    try:
        return parsed.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return parsed


def _noise_frame(path, date_col):
    path = Path(path)
    key = (str(path), path.stat().st_mtime, date_col)
    if key in _NOISE_CACHE:
        return _NOISE_CACHE[key]
    df = load_data(path)
    df.columns = [str(c).strip() for c in df.columns]
    if date_col and date_col in df.columns:
        t = _noise_dates(df[date_col])
        if t.notna().mean() >= 0.5:
            df = df.copy()
            df["__t"] = t
            df = df[df["__t"].notna()].sort_values("__t").reset_index(drop=True)
    _NOISE_CACHE.clear()
    _NOISE_CACHE[key] = df
    return df


def _noise_list(a):
    out = np.asarray(a, dtype="float64")
    return [None if not np.isfinite(v) else round(float(v), 4) for v in out]


def get_noise_series(dataset_name, feature, window=None, method="mean", start=None, end=None,
                     k=2.0, max_points=4000, source_path=None, results_dir=RESULTS_DIR):
    """Raw signal, rolling estimate and residual (noise) of one column, ready for an interactive chart.

    feature  : the plot key (safe_name(column)) or the column name
    method   : "mean" (like the smoothed PNGs) or "median" (like analyze_timeseries_noise)
    start/end: dates (YYYY-MM-DD) for time series, row numbers otherwise
    Raises KeyError / FileNotFoundError when the dataset, file or column cannot be found.
    """
    summary = load_summary(dataset_name, results_dir) or {}
    src = source_path
    if not src and summary.get("source_file") and Path(summary["source_file"]).exists():
        src = summary["source_file"]
    if not src or not Path(src).exists():
        raise FileNotFoundError(f"Source file of '{dataset_name}' not found")

    date_col = summary.get("date_column")
    df = _noise_frame(src, date_col)

    col = feature if feature in df.columns else next((c for c in df.columns if c != "__t" and safe_name(c) == feature), None)
    if col is None:
        raise KeyError(f"Variable '{feature}' not found")

    has_time = "__t" in df.columns
    y_all = pd.to_numeric(df[col], errors="coerce")
    valid = y_all.notna()                      # analyze_timeseries_noise drops NaN rows too
    y = y_all[valid].reset_index(drop=True)
    t = df["__t"][valid].reset_index(drop=True) if has_time else None
    n = len(y)
    if n < 3:
        raise KeyError("Not enough valid observations")

    max_window = int(max(3, min(n // 2, 2000)))
    window = max(1, min(int(window) if window else SMOOTH_WINDOW, max_window))
    roll = y.rolling(window=window, center=True, min_periods=1)
    smooth = roll.median() if method == "median" else roll.mean()
    noise = y - smooth

    mask = np.ones(n, dtype=bool)
    if has_time:
        if start:
            mask &= (t >= pd.Timestamp(start)).to_numpy()
        if end:
            mask &= (t < pd.Timestamp(end) + pd.Timedelta(days=1)).to_numpy()
        years = sorted(int(v) for v in t.dt.year.unique())
        tmin, tmax = t.iloc[0].strftime("%Y-%m-%d"), t.iloc[-1].strftime("%Y-%m-%d")
    else:
        idx = np.arange(n)
        if start not in (None, ""):
            mask &= idx >= int(float(start))
        if end not in (None, ""):
            mask &= idx <= int(float(end))
        years, tmin, tmax = [], 0, n - 1

    full_sel = np.flatnonzero(mask)
    n_sel = int(len(full_sel))
    sel = full_sel if n_sel <= max_points else full_sel[np.linspace(0, n_sel - 1, max_points).astype(int)]

    nz = noise.iloc[full_sel].dropna()          # statistics use every point of the period
    sigma = float(nz.std()) if len(nz) > 1 else 0.0
    lo, hi = -k * sigma, k * sigma
    outside = int(((nz < lo) | (nz > hi)).sum())

    x = ([ts.strftime("%Y-%m-%d %H:%M:%S") for ts in t.iloc[sel]] if has_time else [int(i) for i in sel])
    return {
        "status": "ok", "column": str(col),
        "time_col": date_col if has_time else None,
        "time_kind": "datetime" if has_time else "index",
        "method": method, "window": window, "default_window": SMOOTH_WINDOW, "max_window": max_window,
        "k": k, "n_total": n, "n_selected": n_sel, "n_points": int(len(sel)),
        "years": years, "min": tmin, "max": tmax,
        "sigma": round(sigma, 4), "bound_lo": round(lo, 4), "bound_hi": round(hi, 4),
        "outside": outside, "outside_pct": round(100 * outside / max(1, len(nz)), 2),
        "x": x, "raw": _noise_list(y.iloc[sel]), "smooth": _noise_list(smooth.iloc[sel]), "noise": _noise_list(noise.iloc[sel]),
    }


# ==============================================================================
# INTERACTIVE OUTLIERS (used by the Outliers page: /api/outliers-plot/...)
# Computed on demand because the user changes the variable and the IQR multiplier in the browser.
# Same Tukey rule as outliers_summary(): [Q1 - k*IQR, Q3 + k*IQR], k = 1.5 by default.
# ==============================================================================

_OUTLIER_CACHE = {}


def _outlier_frame(path):
    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key in _OUTLIER_CACHE:
        return _OUTLIER_CACHE[key]
    df = load_data(path)
    df.columns = [str(c).strip() for c in df.columns]
    _OUTLIER_CACHE.clear()
    _OUTLIER_CACHE[key] = df
    return df


def _num(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def get_outlier_series(dataset_name, feature, k=1.5, points="outliers", bins=60,
                       max_points=3000, source_path=None, results_dir=RESULTS_DIR):
    """Everything needed to draw an interactive boxplot + histogram of one numeric column.

    feature : the plot key (safe_name(column)) or the column name
    k       : IQR multiplier of the fences (1.5 = Tukey's rule)
    points  : "outliers" (only the outliers) or "all" (outliers + a sample of the normal values)
    Raises KeyError / FileNotFoundError when the dataset, file or column cannot be found.
    """
    summary = load_summary(dataset_name, results_dir) or {}
    src = source_path
    if not src and summary.get("source_file") and Path(summary["source_file"]).exists():
        src = summary["source_file"]
    if not src or not Path(src).exists():
        raise FileNotFoundError(f"Source file of '{dataset_name}' not found")

    df = _outlier_frame(src)
    col = feature if feature in df.columns else next((c for c in df.columns if safe_name(c) == feature), None)
    if col is None:
        raise KeyError(f"Variable '{feature}' not found")
    date_col = summary.get("date_column")
    date_col = date_col if date_col in df.columns else None

    k = float(min(max(k, 0.5), 5.0))
    bins = int(min(max(bins, 10), 200))
    values = pd.to_numeric(df[col], errors="coerce")
    v = values.dropna()
    if len(v) < 3:
        raise KeyError("Not enough valid observations")

    q1, med, q3 = (float(v.quantile(q)) for q in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    is_out = (v < lower) | (v > upper)
    inside = v[~is_out]
    lower_fence = float(inside.min()) if len(inside) else q1      # Tukey whiskers stop at the last normal value
    upper_fence = float(inside.max()) if len(inside) else q3
    n, n_out = int(len(v)), int(is_out.sum())
    n_low, n_high = int((v < lower).sum()), int((v > upper).sum())
    pct = round(n_out / n * 100, 2)

    def pack(sub):
        rows = [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in sub.index]
        dates = ([None if pd.isna(d) else str(d) for d in df.loc[sub.index, date_col]] if date_col else [None] * len(sub))
        return {"x": [_num(x) for x in sub.to_numpy()], "row": rows, "date": dates}

    out = v[is_out]
    out_shown = out if len(out) <= max_points else out.sample(max_points, random_state=0)
    ins_shown = None
    if points == "all" and len(inside):
        ins_shown = inside if len(inside) <= 1500 else inside.sample(1500, random_state=0)

    # every outlier, most extreme first (distance beyond the fence), capped for the browser
    dist = pd.Series(np.where(out < lower, lower - out, out - upper), index=out.index)
    ordered = dist.sort_values(ascending=False).head(MAX_OUTLIER_TABLE)
    vals = out.loc[ordered.index]
    dates = df.loc[ordered.index, date_col] if date_col else None
    table = []
    for i, (idx, g) in enumerate(ordered.items()):
        val = vals.iloc[i]
        d_ = None if dates is None or pd.isna(dates.iloc[i]) else str(dates.iloc[i])
        table.append({"row": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "date": d_,
                      "value": _num(val), "side": "basse" if val < lower else "haute", "gap": _num(g)})

    return {
        "status": "ok", "column": str(col), "k": k, "points": points,
        "n": n, "n_outliers": n_out, "n_low": n_low, "n_high": n_high, "outlier_pct": pct,
        "q1": _num(q1), "median": _num(med), "q3": _num(q3), "iqr": _num(iqr),
        "lower": _num(lower), "upper": _num(upper),
        "lower_fence": _num(lower_fence), "upper_fence": _num(upper_fence),
        "mean": _num(v.mean()), "std": _num(v.std()), "min": _num(v.min()), "max": _num(v.max()),
        "outliers": pack(out_shown), "n_outliers_shown": int(len(out_shown)),
        "inliers": pack(ins_shown) if ins_shown is not None else None,
        "table": table, "table_total": n_out, "date_col": date_col,
    }


# ==============================================================================
# GLOBAL OUTLIERS BY ISOLATION FOREST (used by the Outliers page: /api/isolation-forest/...)
# ONE score per row, computed on ALL usable numeric columns together (not feature by feature).
# contamination = "auto"  -> sklearn's own threshold (offset_ = -0.5), the count is a result.
# contamination = 0.01-0.30 -> the top X % most isolated rows are flagged.
# ==============================================================================

IF_MAX_FIT_ROWS = 50_000       # the forest is fitted on a sample above this size, all rows are scored
IF_MAX_TABLE = 500             # rows sent to the "most anomalous rows" table
IF_MAX_SCATTER = 4000          # points drawn on the PCA projection
IF_MAX_FEATURES = 60


def get_isolation_forest(dataset_name, contamination="auto", n_estimators=200, source_path=None,
                         results_dir=RESULTS_DIR):
    """Global outlier detection with an Isolation Forest on every usable numeric column.

    Returns the number / percentage of outlier ROWS, the score distribution, a 2-D PCA view
    and the most anomalous rows (with the features that make each of them unusual).
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA

    summary, src = _resolve_source(dataset_name, source_path, results_dir)
    df = _outlier_frame(src)

    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    feats = [c for c in num_cols if df[c].nunique(dropna=True) > 1 and not is_id_like(df, c)][:IF_MAX_FEATURES]
    if not feats:
        raise KeyError("No usable numeric column for Isolation Forest")

    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]
    feats = list(X.columns)
    n_missing_cells = int(X.isna().sum().sum())
    X = X.fillna(X.median())
    if len(X) < 10:
        raise KeyError("Not enough rows for Isolation Forest")

    if contamination == "auto" or contamination is None:
        cont, auto = "auto", True
    else:
        cont, auto = float(min(max(float(contamination), 0.001), 0.5)), False

    fit_X = X if len(X) <= IF_MAX_FIT_ROWS else X.sample(IF_MAX_FIT_ROWS, random_state=42)
    n_estimators = int(min(max(n_estimators, 50), 500))
    model = IsolationForest(n_estimators=n_estimators, contamination=cont, random_state=42, n_jobs=-1)
    model.fit(fit_X)

    # anomaly score in (0, 1): the higher, the easier the row is to isolate.
    # outlier  <=>  anomaly >= threshold (= -offset_), exactly what model.predict() uses.
    anomaly = -model.score_samples(X)
    is_out = model.predict(X) == -1
    n, n_out = int(len(X)), int(is_out.sum())
    pct = round(n_out / n * 100, 2)

    # --- score histogram (inliers / outliers stacked) -----------------------------------------
    edges = np.histogram_bin_edges(anomaly, bins=50)
    h_in, _ = np.histogram(anomaly[~is_out], bins=edges)
    h_out, _ = np.histogram(anomaly[is_out], bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    # --- 2-D PCA view ---------------------------------------------------------------------------
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    scatter = None
    if len(feats) >= 2:
        pca = PCA(n_components=2, random_state=0)
        P = pca.fit_transform(Z)
        idx_out = np.flatnonzero(is_out)
        idx_in = np.flatnonzero(~is_out)
        rng = np.random.RandomState(0)
        if len(idx_out) > IF_MAX_SCATTER // 2:
            idx_out = rng.choice(idx_out, IF_MAX_SCATTER // 2, replace=False)
        if len(idx_in) > IF_MAX_SCATTER:
            idx_in = rng.choice(idx_in, IF_MAX_SCATTER, replace=False)

        def pk(ix):
            return {"x": [_num(v) for v in P[ix, 0]], "y": [_num(v) for v in P[ix, 1]],
                    "row": [int(X.index[i]) if isinstance(X.index[i], (int, np.integer)) else str(X.index[i]) for i in ix],
                    "score": [_num(anomaly[i], 4) for i in ix]}

        ev = pca.explained_variance_ratio_
        scatter = {"inliers": pk(idx_in), "outliers": pk(idx_out),
                   "var": [_num(ev[0] * 100, 1), _num(ev[1] * 100, 1)]}

    # --- most anomalous rows + the features that explain them -----------------------------------
    date_col = summary.get("date_column")
    date_col = date_col if date_col in df.columns else None
    top_idx = np.argsort(-anomaly)[: min(IF_MAX_TABLE, max(n_out, 15))]
    z_abs = Z.abs()
    table = []
    for i in top_idx:
        row_z = Z.iloc[i]
        worst = row_z.abs().sort_values(ascending=False).head(3).index
        idx = X.index[i]
        table.append({
            "row": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "date": None if not date_col or pd.isna(df.loc[idx, date_col]) else str(df.loc[idx, date_col]),
            "score": _num(anomaly[i], 4),
            "outlier": bool(is_out[i]),
            "why": [{"feature": str(c), "value": _num(X.iloc[i][c]), "z": _num(row_z[c], 2)} for c in worst],
        })

    # --- which features are most often extreme among the flagged rows -------------------------------
    drivers = []
    if n_out:
        share = (z_abs[is_out] > 3).mean().sort_values(ascending=False).head(10)
        mean_z = z_abs[is_out].mean()
        drivers = [{"feature": str(c), "pct_extreme": _num(share[c] * 100, 1), "mean_abs_z": _num(mean_z[c], 2)}
                   for c in share.index if share[c] > 0]

    return {
        "status": "ok", "method": "Isolation Forest",
        "n": n, "n_features": len(feats), "features": [str(c) for c in feats],
        "n_outliers": n_out, "outlier_pct": pct, "n_inliers": n - n_out,
        "contamination": "auto" if auto else cont, "n_estimators": n_estimators,
        "threshold": _num(-float(model.offset_), 4),   # anomaly >= threshold  => outlier
        "fitted_on": int(len(fit_X)), "n_missing_cells": n_missing_cells,
        "hist": {"x": [_num(v, 4) for v in centers], "inliers": [int(v) for v in h_in], "outliers": [int(v) for v in h_out]},
        "scatter": scatter, "table": table, "drivers": drivers, "date_col": date_col,
    }


# ==============================================================================
# INTERACTIVE DISTRIBUTIONS (used by the Distributions page: /api/distribution-plot/...)
# Histogram + Gaussian KDE (Scott bandwidth, like seaborn), correlation matrix, target balance.
# ==============================================================================

def _resolve_source(dataset_name, source_path, results_dir):
    summary = load_summary(dataset_name, results_dir) or {}
    src = source_path
    if not src and summary.get("source_file") and Path(summary["source_file"]).exists():
        src = summary["source_file"]
    if not src or not Path(src).exists():
        raise FileNotFoundError(f"Source file of '{dataset_name}' not found")
    return summary, src


def get_distribution_series(dataset_name, feature, bins=30, source_path=None, results_dir=RESULTS_DIR):
    """Histogram counts + KDE curve + shape statistics of one numeric column."""
    _summary, src = _resolve_source(dataset_name, source_path, results_dir)
    df = _outlier_frame(src)
    col = feature if feature in df.columns else next((c for c in df.columns if safe_name(c) == feature), None)
    if col is None:
        raise KeyError(f"Variable '{feature}' not found")
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(v) < 3:
        raise KeyError("Not enough valid observations")

    n = int(len(v))
    counts, edges = np.histogram(v.to_numpy(), bins=int(min(max(bins, 5), 200)))
    width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0

    kde_x = kde_y = None
    std = float(v.std())
    if v.nunique() > 1 and std > 0:
        h = std * n ** (-1 / 5)                                    # Scott's rule
        sample = v.to_numpy() if n <= 5000 else v.sample(5000, random_state=0).to_numpy()
        grid = np.linspace(float(v.min()) - 3 * h, float(v.max()) + 3 * h, 240)
        dens = np.zeros(len(grid))
        for chunk in np.array_split(sample, max(1, len(sample) // 1000)):
            dens += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / h) ** 2).sum(axis=1)
        dens /= len(sample) * h * np.sqrt(2 * np.pi)
        kde_x = [_num(x) for x in grid]
        kde_y = [_num(d * width * 100, 4) for d in dens]           # % of values per bin width

    return {
        "status": "ok", "kind": "variable", "column": str(col), "n": n, "n_unique": int(v.nunique()),
        "mean": _num(v.mean()), "median": _num(v.median()), "std": _num(std),
        "skew": _num(v.skew()), "kurtosis": _num(v.kurt()), "min": _num(v.min()), "max": _num(v.max()),
        "hist_x": [_num((edges[i] + edges[i + 1]) / 2, 4) for i in range(len(counts))],
        "hist_y": [int(c) for c in counts], "hist_w": _num(width, 4),
        "kde_x": kde_x, "kde_y": kde_y,
    }


def get_correlation_matrix(dataset_name, results_dir=RESULTS_DIR):
    data = load_dataset_results(dataset_name, results_dir)
    corr = (data or {}).get("tables", {}).get("correlation")
    if corr is None or corr.empty:
        raise KeyError("No correlation matrix for this dataset")
    labels = [str(c) for c in corr.columns]
    z = [[_num(corr.loc[r, c]) if r in corr.index else None for c in corr.columns] for r in corr.columns]
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if z[i][j] is not None:
                pairs.append({"a": labels[i], "b": labels[j], "r": z[i][j]})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"status": "ok", "kind": "correlation", "labels": labels, "z": z, "pairs": pairs[:200],
            "annotate": len(labels) <= HEATMAP_ANNOT_MAX}


def get_target_balance(dataset_name, results_dir=RESULTS_DIR):
    data = load_dataset_results(dataset_name, results_dir)
    if not data:
        raise KeyError("Dataset not found")
    bal = data["tables"].get("class_balance")
    target = data["summary"].get("target_class")
    if bal is None or bal.empty or not target:
        raise KeyError("No target variable for this dataset")
    if "count" not in bal.columns:
        raise KeyError("Unexpected class balance table")
    return {"status": "ok", "kind": "target", "target": str(target),
            "labels": [str(i) for i in bal.index], "counts": [int(c) for c in bal["count"]],
            "percents": [_num(p, 2) for p in bal["percent"]]}


# ==============================================================================
# FEATURE IMPORTANCE BY ANOVA (used by the Feature importance page: /api/anova/...)
# One-way ANOVA of every numeric column against a categorical target.
#   F  = (SSB / (k-1)) / (SSW / (N-k))   k = number of classes, N = valid rows
#   eta2 = SSB / SST                      share of the column's variance explained by the target
# The larger F, the better the column separates the target classes.
# ==============================================================================

try:                                    # exact p-value when scipy is installed (otherwise p = None)
    from scipy.stats import f as _F_DIST
except Exception:                       # pragma: no cover
    _F_DIST = None

ANOVA_MAX_CLASSES = 30


def _match_column(columns, name):
    return next((c for c in columns if c == name or safe_name(c) == name), None)


def get_anova_targets(dataset, data=None):
    """Colonnes utilisables comme cible : catégorielles (ou la cible détectée) avec 2 à 30 classes."""
    data = data or load_dataset_results(dataset) or {}
    summary, tables = data.get("summary", {}), data.get("tables", {})
    nun = tables.get("n_unique")
    counts = {}
    if nun is not None and not nun.empty:
        col = "n_unique" if "n_unique" in nun.columns else nun.columns[-1]
        counts = {str(i): int(v) for i, v in nun[col].items() if pd.notna(v)}
    cands = list(summary.get("categorical_columns") or [])
    tgt = summary.get("target_class")
    if tgt and tgt not in cands:
        cands.insert(0, tgt)
    out = [c for c in cands if 2 <= counts.get(c, 0) <= ANOVA_MAX_CLASSES]
    default = tgt if tgt in out else (out[0] if out else None)
    return out, default


def get_anova_importance(dataset, target, results_dir=RESULTS_DIR):
    summary, src = _resolve_source(dataset, None, results_dir)
    df = _outlier_frame(src)
    tcol = _match_column(df.columns, target)
    if tcol is None:
        raise KeyError(f"Cible '{target}' introuvable")

    results = []
    for c in summary.get("numeric_columns", []):
        if c == tcol or c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        ok = x.notna() & df[tcol].notna()
        xs = x[ok].to_numpy(float)
        codes, uniques = pd.factorize(df.loc[ok, tcol])
        k, n = len(uniques), len(xs)
        row = {"name": c, "f": 0.0, "p": None, "eta2": 0.0, "n": int(n)}
        if k >= 2 and n > k:
            cnt = np.bincount(codes)
            means = np.bincount(codes, weights=xs) / cnt
            grand = xs.mean()
            ssb = float((cnt * (means - grand) ** 2).sum())
            ssw = float(((xs - means[codes]) ** 2).sum())
            sst = ssb + ssw
            if ssw > 0 and sst > 0:
                F = (ssb / (k - 1)) / (ssw / (n - k))
                row["f"] = float(F)
                row["eta2"] = ssb / sst
                if _F_DIST is not None:
                    row["p"] = float(_F_DIST.sf(F, k - 1, n - k))
        results.append(row)

    results.sort(key=lambda r: r["f"], reverse=True)
    fmax = max((r["f"] for r in results), default=0.0) or 1.0
    for i, r in enumerate(results, 1):
        r["rank"] = i
        r["score_pct"] = round(r["f"] / fmax * 100, 1)
        r["f"] = round(r["f"], 2)
        r["eta2"] = round(r["eta2"], 4)
        r["significant"] = (r["p"] is not None and r["p"] < 0.05)
        r["p"] = None if r["p"] is None else float(f"{r['p']:.3e}")

    vc = df[tcol].value_counts()
    return {"status": "ok", "target": str(tcol), "n_classes": int(len(vc)),
            "classes": [{"label": str(k), "count": int(v)} for k, v in vc.head(10).items()],
            "has_p": _F_DIST is not None, "features": results}


# ==============================================================================
# DATA QUALITY (EDA step "Data Quality": /dataset/<name>/eda/quality)
# One verdict per indicator: missing values, duplicates, outliers, noise, class imbalance.
# status : ok (no treatment) | warning (worth treating / monitoring) | critical (treat before modelling) | na
# All thresholds live in DQ_THRESHOLDS so they are easy to tune.
# ==============================================================================

DQ_THRESHOLDS = {
    "missing_cells_warn": 1.0,       # % of missing cells in the whole dataset
    "missing_cells_crit": 15.0,
    "missing_col_warn": 5.0,         # % of missing values in one column
    "missing_col_crit": 40.0,        # a column above this is better dropped than imputed
    "dup_warn": 0.0,                 # any exact duplicate is worth removing
    "dup_crit": 5.0,                 # % of duplicated rows
    "out_col_warn": 5.0,             # % of IQR outliers in one column
    "out_col_crit": 10.0,
    "out_rows_warn": 5.0,            # % of rows with >= 1 IQR outlier
    "out_rows_crit": 15.0,
    "noise_nsr_warn": 0.15,          # time series: std(x - rolling mean) / std(x)
    "noise_nsr_crit": 0.35,
    "noise_label_warn": 10.0,        # classification: % of rows whose k-NN neighbours mostly disagree with their label
    "noise_label_crit": 25.0,
    "imb_ratio_warn": 1.5,           # majority class count / minority class count
    "imb_ratio_crit": 5.0,
    "imb_min_class_rows": 30,        # a class with fewer rows than this is critical on its own
    "skew_warn": 1.0,                # regression target: |skewness|
    "skew_crit": 2.0,
}
_DQ_LABEL = {"ok": "Aucun traitement", "warning": "À surveiller / traiter", "critical": "Traitement requis", "na": "Non applicable"}
_DQ_ORDER = {"ok": 0, "na": 0, "warning": 1, "critical": 2}


def _dq_level(value, warn, crit, strict=False):
    """ok / warning / critical from two thresholds (>= by default, > when strict)."""
    if value is None:
        return "na"
    if (value > crit) if strict else (value >= crit):
        return "critical"
    if (value > warn) if strict else (value >= warn):
        return "warning"
    return "ok"


def _dq_health(value, warn, crit):
    """0-100 health for the gauge (100 = perfect): 60 at the warning threshold, 20 at the critical one, 0 at twice the critical value."""
    if value is None:
        return None
    v = max(float(value), 0.0)
    if v <= 0:
        return 100
    if v <= warn:
        return round(100 - 40 * v / warn)
    if v <= crit:
        return round(60 - 40 * (v - warn) / max(crit - warn, 1e-9))
    return round(max(0.0, 20 - 20 * min(1.0, (v - crit) / max(crit, 1e-9))))


def _dq_indicator(key, title, icon, status, headline, metrics, advice, details=None, preprocessing_step=None, health=None, display=None, caption=None):
    return {"key": key, "health": None if status == "na" else health, "display": display or "–", "caption": caption or "",
            "title": title, "icon": icon, "status": status, "status_label": _DQ_LABEL[status],
            "needs_treatment": status in ("warning", "critical"), "headline": headline, "metrics": metrics,
            "advice": advice, "details": details or [], "preprocessing_step": preprocessing_step}


def _dq_missing(df):
    T = DQ_THRESHOLDS
    n, m = df.shape
    miss = df.isna().sum()
    total = int(miss.sum())
    cell_pct = round(total / max(n * m, 1) * 100, 2)
    row_pct = round(float(df.isna().any(axis=1).mean() * 100), 2) if n else 0.0
    col_pct = (df.isna().mean() * 100).round(2)
    bad = col_pct[col_pct > 0].sort_values(ascending=False)
    worst_col = str(bad.index[0]) if len(bad) else None
    worst_pct = float(bad.iloc[0]) if len(bad) else 0.0
    n_crit_cols = int((col_pct >= T["missing_col_crit"]).sum())
    levels = [_dq_level(cell_pct, T["missing_cells_warn"], T["missing_cells_crit"]),
              _dq_level(worst_pct, T["missing_col_warn"], T["missing_col_crit"])]
    status = max(levels, key=_DQ_ORDER.get)
    if total == 0:
        status, headline, advice = "ok", "Aucune valeur manquante.", "Rien à imputer."
    else:
        headline = f"{cell_pct}% de cellules manquantes · {len(bad)} colonne(s) concernée(s) · pire : {worst_col} ({worst_pct}%)"
        if n_crit_cols:
            advice = (f"{n_crit_cols} colonne(s) dépassent {T['missing_col_crit']:.0f}% de manquants : les supprimer plutôt que les imputer, "
                      "puis imputer le reste (médiane pour le numérique, mode pour le catégoriel).")
        elif status == "ok":
            advice = "Taux faible : une imputation simple est optionnelle (la plupart des modèles tolèrent ce niveau)."
        else:
            advice = "Imputer (médiane / mode, ou KNN si les manquants ne sont pas aléatoires) avant la modélisation."
    details = [{"column": str(c), "value": float(p), "count": int(miss[c]),
                "status": _dq_level(float(p), T["missing_col_warn"], T["missing_col_crit"])} for c, p in bad.head(15).items()]
    metrics = [("Cellules manquantes", f"{total:,}".replace(",", " ")), ("% cellules", f"{cell_pct}%"),
               ("Lignes avec ≥1 manquant", f"{row_pct}%"), ("Colonnes concernées", f"{len(bad)} / {m}")]
    health = min(_dq_health(cell_pct, T["missing_cells_warn"], T["missing_cells_crit"]), _dq_health(worst_pct, T["missing_col_warn"], T["missing_col_crit"]))
    return _dq_indicator("missing", "Valeurs manquantes", "🕳️", status, headline, metrics, advice, details, "missing",
                         health, f"{cell_pct}%", "cellules manquantes")


def _dq_duplicates(df, id_cols):
    T = DQ_THRESHOLDS
    n = len(df)
    try:
        dup = int(df.duplicated().sum())
    except TypeError:                                   # unhashable cells (lists...) -> compare as text
        dup = int(df.astype(str).duplicated().sum())
    pct = round(dup / max(n, 1) * 100, 2)
    dup_no_id = None
    if id_cols and len(id_cols) < df.shape[1]:
        try:
            dup_no_id = int(df.drop(columns=id_cols).duplicated().sum())
        except TypeError:
            dup_no_id = None
    status = "ok" if dup == 0 else _dq_level(pct, T["dup_warn"], T["dup_crit"], strict=True)
    if status == "ok" and dup:
        status = "warning"
    if dup == 0:
        headline, advice = "Aucune ligne dupliquée.", "Rien à supprimer."
    else:
        headline = f"{dup} ligne(s) dupliquée(s) ({pct}% du dataset)"
        advice = "Supprimer les doublons exacts (en gardant la 1re occurrence) : ils gonflent artificiellement certaines classes et fuient entre train et test."
    if dup_no_id and dup_no_id > dup:
        advice += f" Attention : sans la colonne identifiant, {dup_no_id} lignes sont identiques ({round(dup_no_id / n * 100, 2)}%)."
    metrics = [("Lignes dupliquées", f"{dup:,}".replace(",", " ")), ("% du dataset", f"{pct}%"), ("Lignes uniques", f"{n - dup:,}".replace(",", " "))]
    if dup_no_id is not None:
        metrics.append(("Doublons hors identifiant", f"{dup_no_id}"))
    return _dq_indicator("duplicates", "Doublons", "🔁", status, headline, metrics, advice, None, "duplicates",
                         _dq_health(pct, 0.0, T["dup_crit"]), f"{dup}", "lignes dupliquées")


def _dq_outliers(df, num_cols):
    T = DQ_THRESHOLDS
    cols = [c for c in num_cols if df[c].nunique(dropna=True) > 2]       # binary / constant columns have no outliers
    rows, flag = [], pd.Series(False, index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        mask = (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)
        extreme = (v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)
        flag |= mask.fillna(False)
        cnt = int(mask.sum())
        rows.append({"column": str(c), "count": cnt, "value": round(cnt / max(int(v.notna().sum()), 1) * 100, 2),
                     "extreme": int(extreme.sum()), "skew": _num(v.skew(), 2)})
    if not rows:
        return _dq_indicator("outliers", "Outliers (aberrants)", "📦", "na", "Pas de variable numérique exploitable.", [],
                             "Aucune colonne numérique continue à analyser.", None, "outliers")
    rows.sort(key=lambda r: -r["value"])
    rows_pct = round(float(flag.mean() * 100), 2)
    worst = rows[0]
    for r in rows:
        r["status"] = _dq_level(r["value"], T["out_col_warn"], T["out_col_crit"])
    n_bad = sum(1 for r in rows if r["value"] >= T["out_col_warn"])
    status = max([_dq_level(rows_pct, T["out_rows_warn"], T["out_rows_crit"]), _dq_level(worst["value"], T["out_col_warn"], T["out_col_crit"])],
                 key=_DQ_ORDER.get)
    headline = f"{rows_pct}% des lignes ont ≥1 outlier IQR · pire colonne : {worst['column']} ({worst['value']}%)"
    skewed = [r["column"] for r in rows if r["value"] >= T["out_col_warn"] and r["skew"] is not None and abs(r["skew"]) > 1][:3]
    if status == "ok":
        advice = "Outliers rares : aucun traitement nécessaire (laisser tels quels, surtout pour les modèles à base d'arbres)."
    else:
        advice = ("Vérifier d'abord si ce sont des erreurs de mesure ou des valeurs réelles extrêmes. Si ce sont des erreurs : capping (winsorisation) ou suppression ; "
                  "sinon garder et utiliser un modèle robuste / un scaler robuste.")
        if skewed:
            advice += f" Les colonnes {', '.join(skewed)} sont très asymétriques : une transformation log peut suffire."
    metrics = [("Lignes avec ≥1 outlier", f"{rows_pct}%"), ("Colonnes ≥ 5% d'outliers", f"{n_bad} / {len(rows)}"),
               ("Valeurs extrêmes (3·IQR)", f"{sum(r['extreme'] for r in rows):,}".replace(",", " ")), ("Méthode", "IQR × 1.5")]
    health = min(_dq_health(rows_pct, T["out_rows_warn"], T["out_rows_crit"]), _dq_health(worst["value"], T["out_col_warn"], T["out_col_crit"]))
    return _dq_indicator("outliers", "Outliers (aberrants)", "📦", status, headline, metrics, advice, rows[:15], "outliers",
                         health, f"{rows_pct}%", "lignes avec outlier")


def _dq_noise(df, num_cols, summary, target):
    T = DQ_THRESHOLDS
    date_col = summary.get("date_column")
    is_ts = summary.get("dataset_type") == "time_series" and date_col in df.columns
    cols = [c for c in num_cols if df[c].nunique(dropna=True) > 5]
    if is_ts and cols:
        try:
            d = _noise_frame(summary["source_file"], date_col) if summary.get("source_file") and Path(summary["source_file"]).exists() else df
        except Exception:
            d = df
        w = max(3, min(11, len(d) // 50 * 2 + 1))
        rows = []
        for c in cols[:40]:
            x = pd.to_numeric(d[c], errors="coerce").dropna()
            if len(x) < 30 or x.std() == 0:
                continue
            resid = x - x.rolling(w, center=True, min_periods=1).mean()
            nsr = float(resid.std() / x.std())
            rows.append({"column": str(c), "value": round(nsr, 3), "status": _dq_level(nsr, T["noise_nsr_warn"], T["noise_nsr_crit"])})
        if rows:
            rows.sort(key=lambda r: -r["value"])
            med = float(np.median([r["value"] for r in rows]))
            n_bad = sum(1 for r in rows if r["value"] >= T["noise_nsr_warn"])
            status = max([_dq_level(med, T["noise_nsr_warn"], T["noise_nsr_crit"]),
                          "warning" if n_bad and n_bad >= len(rows) / 3 else "ok"], key=_DQ_ORDER.get)
            headline = f"Ratio bruit/signal médian {med:.2f} · {n_bad}/{len(rows)} signaux bruités · pire : {rows[0]['column']} ({rows[0]['value']:.2f})"
            advice = ("Signaux propres : pas de lissage nécessaire." if status == "ok" else
                      f"Lisser les signaux bruités (moyenne mobile fenêtre {w}, ou médiane mobile si le bruit est impulsif) et ne pas lisser la cible à prédire.")
            metrics = [("Ratio bruit/signal médian", f"{med:.2f}"), ("Signaux bruités (≥0.15)", f"{n_bad} / {len(rows)}"),
                       ("Fenêtre de lissage", str(w)), ("Mesure", "σ(x − moyenne mobile) / σ(x)")]
            return _dq_indicator("noise", "Bruit (noise)", "〰️", status, headline, metrics, advice, rows[:15], "noise",
                                 _dq_health(med, T["noise_nsr_warn"], T["noise_nsr_crit"]), f"{med:.2f}", "ratio bruit/signal")
    # tabular classification: label-noise / class-overlap indicator (k-NN disagreement)
    if target and target in df.columns and df[target].nunique(dropna=True) <= 30 and len(num_cols) >= 2:
        try:
            from sklearn.neighbors import NearestNeighbors
            feats = [c for c in num_cols if c != target and df[c].nunique(dropna=True) > 1][:30]
            d = df[feats + [target]].dropna(subset=[target])
            if len(d) > 5000:
                d = d.sample(5000, random_state=0)
            if len(d) >= 50 and feats:
                X = d[feats].apply(pd.to_numeric, errors="coerce")
                X = X.fillna(X.median()).fillna(0)
                X = (X - X.mean()) / X.std().replace(0, 1)
                y = pd.factorize(d[target])[0]
                k = 5
                idx = NearestNeighbors(n_neighbors=k + 1).fit(X.to_numpy()).kneighbors(X.to_numpy(), return_distance=False)[:, 1:]
                neigh = y[idx]
                maj = np.array([np.bincount(r).argmax() for r in neigh])
                noisy = float((maj != y).mean() * 100)
                status = _dq_level(noisy, T["noise_label_warn"], T["noise_label_crit"])
                headline = f"{noisy:.1f}% des lignes ressemblent surtout à une autre classe que la leur (k-NN, k={k})"
                advice = ("Classes bien séparées : pas de nettoyage nécessaire." if status == "ok" else
                          "Classes qui se chevauchent / étiquettes possiblement erronées : vérifier ces lignes, filtrer les points ambigus (ex. ENN / Tomek links) "
                          "ou utiliser un modèle régularisé. Ce n'est pas forcément une erreur : certaines classes sont naturellement proches.")
                metrics = [("Lignes en désaccord k-NN", f"{noisy:.1f}%"), ("Voisins (k)", str(k)), ("Variables utilisées", str(len(feats))),
                           ("Mesure", "bruit d'étiquette (proxy)")]
                return _dq_indicator("noise", "Bruit (noise)", "〰️", status, headline, metrics, advice, None, "noise",
                                     _dq_health(noisy, T["noise_label_warn"], T["noise_label_crit"]), f"{noisy:.1f}%", "bruit d'étiquette")
        except Exception:
            pass
    return _dq_indicator("noise", "Bruit (noise)", "〰️", "na", "Mesure non applicable pour ce dataset.", [],
                         "Le bruit se mesure sur des signaux ordonnés (série temporelle) ou via une cible de classification ; ni l'un ni l'autre n'est disponible ici.",
                         None, "noise")


def _dq_imbalance(df, num_cols, target):
    T = DQ_THRESHOLDS
    if not target or target not in df.columns:
        return _dq_indicator("imbalance", "Déséquilibre des classes", "⚖️", "na", "Pas de variable cible détectée.", [],
                             "Choisir une cible dans l'étape Split pour évaluer le déséquilibre.", None, "smote")
    y = df[target].dropna()
    k = y.nunique()
    if k > 30 or (pd.api.types.is_float_dtype(y) and k > 10):          # continuous target -> regression: check skew instead
        sk = _num(pd.to_numeric(y, errors="coerce").skew(), 2)
        status = _dq_level(abs(sk), T["skew_warn"], T["skew_crit"]) if sk is not None else "na"
        headline = f"Cible continue « {target} » · asymétrie {sk}" if sk is not None else f"Cible continue « {target} »"
        advice = ("Cible à peu près symétrique : rien à faire." if status in ("ok", "na") else
                  "Cible très asymétrique : essayer une transformation log / Box-Cox de la cible, ou un échantillonnage stratifié par quantiles.")
        return _dq_indicator("imbalance", "Déséquilibre de la cible", "⚖️", status, headline,
                             [("Type de cible", "continue (régression)"), ("Asymétrie", str(sk))], advice, None, None,
                             _dq_health(abs(sk), T["skew_warn"], T["skew_crit"]) if sk is not None else None, f"{sk}", "asymétrie de la cible")
    vc = y.value_counts()
    if k < 2:
        return _dq_indicator("imbalance", "Déséquilibre des classes", "⚖️", "na", "Une seule classe dans la cible.", [], "Cible constante : inutilisable.", None, None)
    ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 2)
    minority_pct = round(float(vc.iloc[-1] / vc.sum() * 100), 2)
    probs = (vc / vc.sum()).to_numpy()
    entropy = round(float(-(probs * np.log(probs)).sum() / np.log(k)), 3)
    status = _dq_level(ratio, T["imb_ratio_warn"], T["imb_ratio_crit"])
    if int(vc.iloc[-1]) < T["imb_min_class_rows"]:
        status = "critical"
    details = [{"column": str(lab), "value": round(float(c / vc.sum() * 100), 2), "count": int(c), "status": "ok"} for lab, c in vc.head(15).items()]
    headline = f"Ratio majoritaire/minoritaire {ratio}:1 · classe minoritaire {minority_pct}% · {k} classes"
    if status == "ok":
        advice = "Classes équilibrées : aucun rééquilibrage nécessaire."
    elif status == "warning":
        advice = "Déséquilibre modéré : utiliser un split stratifié et les métriques F1 / AUC ; class_weight='balanced' suffit souvent."
    else:
        advice = ("Fort déséquilibre : split stratifié obligatoire, class_weight='balanced' et/ou SMOTE (appliqué uniquement sur le train), "
                  "et évaluer avec F1 / AUC plutôt que l'accuracy.")
    if int(vc.iloc[-1]) < T["imb_min_class_rows"]:
        advice += f" La classe minoritaire n'a que {int(vc.iloc[-1])} lignes : envisager de la fusionner ou de collecter plus de données."
    metrics = [("Ratio max/min", f"{ratio}:1"), ("Classe minoritaire", f"{minority_pct}%"), ("Nb de classes", str(k)),
               ("Entropie normalisée", f"{entropy} (1 = équilibré)")]
    health = _dq_health(ratio - 1, T["imb_ratio_warn"] - 1, T["imb_ratio_crit"] - 1)
    if int(vc.iloc[-1]) < T["imb_min_class_rows"]:
        health = min(health, 20)
    return _dq_indicator("imbalance", "Déséquilibre des classes", "⚖️", status, headline, metrics, advice, details, "smote",
                         health, f"{ratio}:1", "majoritaire / minoritaire")


def get_data_quality(dataset_name, target=None, source_path=None, results_dir=RESULTS_DIR):
    """Five data-quality indicators (missing, duplicates, outliers, noise, imbalance) with a treatment verdict each, plus a global score."""
    summary, src = _resolve_source(dataset_name, source_path, results_dir)
    df = _outlier_frame(src)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                and not is_id_like(df, c)]
    id_cols = [c for c in df.columns if is_id_like(df, c)]
    # target: the one asked for (?target=) > the one of the analysis > automatic detection
    target = _match_column(df.columns, target) if target else None
    target = target or _match_column(df.columns, summary.get("target_class") or "") or detect_target(df, exclude=id_cols)
    target_candidates = [str(c) for c in df.columns if c not in id_cols and 2 <= df[c].nunique(dropna=True) <= 30][:25]
    if target in num_cols:
        num_cols = [c for c in num_cols if c != target]

    indicators = [_dq_missing(df), _dq_duplicates(df, id_cols), _dq_outliers(df, num_cols),
                  _dq_noise(df, num_cols, summary, target), _dq_imbalance(df, num_cols, target)]

    scored = [i for i in indicators if i["status"] != "na"]
    hs = [i["health"] for i in scored if i["health"] is not None]
    score = int(round(sum(hs) / len(hs))) if hs else None
    n_crit = sum(1 for i in indicators if i["status"] == "critical")
    n_warn = sum(1 for i in indicators if i["status"] == "warning")
    if n_crit:
        verdict, level = f"{n_crit} indicateur(s) critique(s) : traitement requis avant la modélisation.", "critical"
    elif n_warn:
        verdict, level = f"{n_warn} indicateur(s) à surveiller : un traitement léger est recommandé.", "warning"
    else:
        verdict, level = "Données propres : aucun traitement nécessaire.", "ok"
    return {"status": "ok", "n_rows": int(len(df)), "n_cols": int(df.shape[1]), "target": target, "score": score,
            "target_candidates": target_candidates, "level": level, "verdict": verdict, "indicators": indicators, "thresholds": DQ_THRESHOLDS,
            "to_treat": [i["title"] for i in indicators if i["needs_treatment"]]}


if __name__ == "__main__":
    import sys
    test_file = "weatherHistory.csv" if os.path.exists("weatherHistory.csv") else None
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if test_file and os.path.exists(test_file):
        print(f"Running pipeline on {test_file}...")
        res = run_pipeline(test_file)
        print("Analysis completed successfully!")
        print("Summary keys:", list(res["summary"].keys()))
        print("Dataset Type:", res["summary"].get("dataset_type"))
        print("Selected Features:", res["summary"].get("selected_features"))
    else:
        print("No test file found. Specify a CSV file path as argument.")
