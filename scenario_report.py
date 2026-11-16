"""
Comparaison des modèles selon 4 scénarios de données, puis rapport PDF dédié.

    A. avec outliers  + avec bruit   (données brutes : aucun traitement d'outliers, aucun lissage)
    B. sans outliers  + avec bruit   (One-Class SVM : les lignes d'entraînement aberrantes sont retirées)
    C. avec outliers  + sans bruit   (moyenne mobile sur les variables, séries temporelles seulement)
    D. sans outliers  + sans bruit   (les deux traitements)

Principe d'équité :
  * mêmes modèles ET mêmes hyperparamètres dans les 4 scénarios (ceux de la dernière sélection de modèles
    enregistrée, sinon les valeurs par défaut) -> seule la donnée change ;
  * mêmes réglages de prétraitement que ceux enregistrés (imputation, encodage, mise à l'échelle...), seuls
    `outliers` et `smooth_window` sont forcés ;
  * le split train / test est celui de l'étape Split ; les outliers ne sont retirés que du TRAIN (le test n'est
    jamais touché) et le lissage n'est jamais appliqué à la cible ;
  * rien n'est écrit dans le prétraitement ni dans les résultats de modèles enregistrés : seul le cache
    (step_cache, type « scenario_cmp ») est alimenté, pour qu'un second rapport soit instantané.

Le « bruit » est ici celui de l'étape Prétraitement : variations à haute fréquence retirées par moyenne mobile.
Il n'a de sens que pour une série temporelle ; sinon seuls les scénarios « avec / sans outliers » sont calculés.
"""
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import analysis_engine as ae
import model_selection as ms
import preprocessing as pp
import split as sp
import step_cache

DEFAULT_WINDOW = 5
NEGLIGIBLE = 0.005            # écart de score (R² ou F1) en dessous duquel on parle d'effet négligeable
KIND = "scenario_cmp"

SCENARIO_DEFS = [
    {"id": "raw", "outliers": "none", "noise": False},
    {"id": "no_outliers", "outliers": "ocsvm", "noise": False},
    {"id": "no_noise", "outliers": "none", "noise": True},
    {"id": "clean", "outliers": "ocsvm", "noise": True},
]

TXT = {
    "fr": {
        "raw": "Avec outliers, avec bruit", "no_outliers": "Sans outliers, avec bruit",
        "no_noise": "Avec outliers, sans bruit", "clean": "Sans outliers, sans bruit",
        "raw_s": "A. Brut", "no_outliers_s": "B. Sans outliers", "no_noise_s": "C. Sans bruit", "clean_s": "D. Sans les deux",
    },
    "en": {
        "raw": "With outliers, with noise", "no_outliers": "Without outliers, with noise",
        "no_noise": "With outliers, without noise", "clean": "Without outliers, without noise",
        "raw_s": "A. Raw", "no_outliers_s": "B. No outliers", "no_noise_s": "C. No noise", "clean_s": "D. Neither",
    },
}


# ======================================================================
# Calcul
# ======================================================================
def _base_config(ds, summary):
    """Réglages de prétraitement enregistrés (sinon défauts + réglages de l'étape Split)."""
    rep = pp.load_report(ds) or {}
    cfg = dict(rep.get("config") or {})
    sc = sp.get_config(ds) or {}
    if not cfg.get("target"):
        cfg["target"] = sc.get("target") or summary.get("target_class")
    if not rep:
        if sc.get("test_size") is not None:
            cfg["test_size"] = sc["test_size"]
        if sc.get("strategy"):
            cfg["split_strategy"] = sc["strategy"]
    return cfg, rep.get("created_at")


def saved_models(ds, task=None):
    """Modèles déjà entraînés dans l'étape Modèles, avec leurs hyperparamètres finaux : (task, [(id, params)])."""
    found = {}
    for t in ("classification", "regression"):
        for g in ms.GROUPS:
            r = ms.get_results(ds, t, g)
            for m in (r or {}).get("results", []):
                if m.get("id") in ms.MODELS and ms.available(m["id"]):
                    found.setdefault(t, {})[m["id"]] = m.get("params") or {}
    if not found:
        return task, []
    if task not in found:
        task = max(found, key=lambda t: len(found[t]))
    order = [k for g in ms.GROUPS.values() for k in g["models"]]
    return task, [(k, found[task][k]) for k in order if k in found[task]]


def _scenarios(is_ts):
    return [s for s in SCENARIO_DEFS if is_ts or not s["noise"]]


def _row(r, task):
    t = r["test"]
    row = {"id": r["id"], "name": r["name"], "cv": r["cv_mean"], "cv_std": r["cv_std"], "train": r["train_score"],
           "test": r["test_score"], "seconds": r["seconds"], "overfit": r["overfit"]}
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc") if task == "classification" else ("r2", "mae", "rmse")
    row.update({k: t.get(k) for k in keys})
    return row


def _step_detail(steps, key):
    return next((s.get("detail") for s in steps if s.get("key") == key), None)


def _compute(ds, task, models, window, cv, progress):
    summary, raw = pp._load(ds)
    dtype, date_col = pp._dtype(summary), summary.get("date_column")
    is_ts = dtype == "time_series" and date_col is not None
    base, _ = _base_config(ds, summary)
    target = base.get("target")
    if not target:
        raise ValueError("Aucune variable cible : choisissez y dans l'étape Split.")
    scen = _scenarios(is_ts)
    total = len(scen) * len(models)
    done = 0
    out_sc, notes, used_task, chrono = [], [], task, False
    for sd in scen:
        cfg = dict(base, outliers=sd["outliers"], smooth_window=window if sd["noise"] else 0)
        clean, steps, _extra = pp.run(raw, cfg, dtype, date_col)
        d = ms._dataset(ds, data=(clean, True, {"target": target}))
        used_task = ms._task_of(d["y_raw"], used_task or "auto")
        chrono = d["chronological"]
        Xtr, Xte, ytr, yte, classes = ms._split_xy(d, used_task)
        rows = []
        for key, params in models:
            done += 1
            if progress:
                progress(done - 1, total, f"{sd['id']} · {ms.MODELS[key]['name']}")
            c = {"params": {key: params}, "tune": False}
            try:
                r, _pred = ms._fit_one(key, used_task, Xtr, ytr, Xte, yte, c, cv, chrono, classes)
                rows.append(_row(r, used_task))
            except Exception as e:                      # un modèle qui échoue ne bloque pas les autres
                rows.append({"id": key, "name": ms.MODELS[key]["name"], "error": f"{type(e).__name__}: {e}"[:160]})
        ok = [r for r in rows if "error" not in r and r.get("test") is not None]
        out_sc.append({
            "id": sd["id"], "outliers": sd["outliers"], "noise": sd["noise"], "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "outlier_note": _step_detail(steps, "outliers"), "noise_note": _step_detail(steps, "noise"),
            "rows": rows, "best": max(ok, key=lambda r: r["test"])["id"] if ok else None,
        })
    if progress:
        progress(total, total, "")
    base_n = out_sc[0]["n_train"]
    for s in out_sc:
        s["n_removed"] = max(0, base_n - s["n_train"])
    if not is_ts:
        notes.append("not_ts")
    return {
        "dataset": ae.clean_dataset_name(ds), "task": used_task, "target": target, "is_time_series": is_ts, "window": window if is_ts else None,
        "metric": "F1 pondéré" if used_task == "classification" else "R²", "cv": cv, "nu": base.get("ocsvm_nu", pp.DEFAULTS["ocsvm_nu"]),
        "chronological": chrono, "models": [k for k, _ in models], "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenarios": out_sc, "notes": notes,
    }


def run(dataset, task=None, window=None, cv=3, models=None, refresh=False, progress=None):
    """Résultats des modèles dans les 4 scénarios (2 si ce n'est pas une série temporelle). Mis en cache."""
    ds = ae.clean_dataset_name(dataset)
    summary, _ = pp._load(ds)
    base, pp_at = _base_config(ds, summary)
    task = task if task in ("classification", "regression") else None
    saved_task, saved = saved_models(ds, task)
    if models:
        chosen = [(k, dict(p)) for k, p in models if k in ms.MODELS and ms.available(k)]
    elif saved:
        chosen, task = saved, saved_task
    else:
        chosen = [(k, ms._clean_params(None)[k]) for g in ms.GROUPS.values() for k in g["models"] if ms.available(k)]
    if not chosen:
        raise ValueError("Aucun modèle utilisable.")
    try:
        window = int(window) if window else 0
    except (TypeError, ValueError):
        window = 0
    window = window if window >= 2 else (int(base.get("smooth_window") or 0) if int(base.get("smooth_window") or 0) >= 2 else DEFAULT_WINDOW)
    cv = cv if cv in ms.CV_CHOICES else 3
    params = {"task": task, "window": window, "cv": cv, "models": chosen, "pp_at": pp_at,
              "pp": {k: base.get(k) for k in sorted(base) if k not in ("outliers", "smooth_window")}}
    res = step_cache.get_or_compute(ds, KIND, params, lambda: _compute(ds, task, chosen, window, cv, progress), refresh=refresh)
    res["models_source"] = "saved" if (saved and not models) else "default" if not models else "custom"
    return res


# ======================================================================
# Lecture des résultats
# ======================================================================
def label(sid, lang="fr", short=False):
    return TXT.get(lang, TXT["fr"])[sid + ("_s" if short else "")]


def deltas(res):
    """{scénario: {modèle: score_test - score_test du scénario brut}} pour les scénarios autres que le brut."""
    sc = {s["id"]: s for s in res["scenarios"]}
    base = {r["id"]: r for r in sc["raw"]["rows"] if "error" not in r and r.get("test") is not None}
    out = {}
    for sid, s in sc.items():
        if sid == "raw":
            continue
        out[sid] = {r["id"]: r["test"] - base[r["id"]]["test"] for r in s["rows"]
                    if "error" not in r and r.get("test") is not None and r["id"] in base}
    return out


def _verdict(mean, lang):
    if abs(mean) < NEGLIGIBLE:
        return "effet négligeable" if lang == "fr" else "negligible effect"
    return ("amélioration" if mean > 0 else "dégradation") if lang == "fr" else ("improvement" if mean > 0 else "degradation")


def auto_summary(res, lang="fr"):
    """Synthèse rédigée par règles (aucun appel IA), uniquement à partir des chiffres calculés."""
    fr = lang != "en"
    T = (lambda a, b: a if fr else b)
    name = {k: ms.MODELS[k]["name"] for k in res["models"]}
    sc = {s["id"]: s for s in res["scenarios"]}
    m = res["metric"]
    L = []
    best = []
    for s in res["scenarios"]:
        ok = [r for r in s["rows"] if "error" not in r and r.get("test") is not None]
        if ok:
            b = max(ok, key=lambda r: r["test"])
            best.append((s["id"], b))
    if best:
        L.append(T("Meilleur modèle par scénario (" + m + " sur le test) : ", "Best model per scenario (" + m + " on the test set): ")
                 + " ; ".join(f"{label(i, lang, True)} → **{b['name']}** ({b['test']:.3f})" for i, b in best) + ".")
        top = max(best, key=lambda x: x[1]["test"])
        L.append(T(f"Meilleur résultat global : **{top[1]['name']}** dans le scénario « {label(top[0], lang)} » ({top[1]['test']:.3f}).",
                   f"Best overall result: **{top[1]['name']}** in the scenario “{label(top[0], lang)}” ({top[1]['test']:.3f})."))
    dl = deltas(res)
    for sid, fr_txt, en_txt in (("no_outliers", "Retirer les outliers", "Removing outliers"), ("no_noise", "Retirer le bruit (lissage)", "Removing noise (smoothing)"),
                                ("clean", "Retirer les deux", "Removing both")):
        d = dl.get(sid)
        if not d:
            continue
        mean = float(np.mean(list(d.values())))
        up = sum(1 for v in d.values() if v > NEGLIGIBLE)
        down = sum(1 for v in d.values() if v < -NEGLIGIBLE)
        flat = len(d) - up - down
        bk, wk = max(d, key=d.get), min(d, key=d.get)
        extra = ""
        if sid in ("no_outliers", "clean") and sc[sid]["n_removed"]:
            extra = T(f" ({sc[sid]['n_removed']:,} lignes retirées du train)", f" ({sc[sid]['n_removed']:,} rows removed from train)").replace(",", " ")
        L.append(T(f"{fr_txt}{extra} : écart moyen de {m} **{mean:+.3f}** ({_verdict(mean, lang)}) ; {up} modèle(s) en hausse, {down} en baisse, {flat} stable(s). "
                   f"Écart le plus favorable : {name[bk]} ({d[bk]:+.3f}) ; le plus défavorable : {name[wk]} ({d[wk]:+.3f}).",
                   f"{en_txt}{extra} : mean {m} change **{mean:+.3f}** ({_verdict(mean, lang)}); {up} model(s) up, {down} down, {flat} stable. "
                   f"Most favourable change: {name[bk]} ({d[bk]:+.3f}); least favourable: {name[wk]} ({d[wk]:+.3f})."))
    if "not_ts" in res.get("notes", []):
        L.append(T("Le bruit (lissage par moyenne mobile) ne s'applique qu'aux séries temporelles : ce dataset n'en est pas une, seuls les scénarios avec / sans outliers sont comparés.",
                   "Noise (moving-average smoothing) only applies to time series: this dataset is not one, so only the with / without outliers scenarios are compared."))
    L.append(T(f"Écarts inférieurs à {NEGLIGIBLE} : à considérer comme du hasard (un seul entraînement par modèle et par scénario).",
               f"Differences below {NEGLIGIBLE} should be treated as chance (one training per model and scenario)."))
    return "\n".join("- " + x for x in L)


def compact(res):
    """Version courte pour l'agent (chiffres clés uniquement)."""
    dl = deltas(res)
    return {
        "dataset": res["dataset"], "tache": res["task"], "cible": res["target"], "metrique": res["metric"], "serie_temporelle": res["is_time_series"],
        "scenarios": [{"id": s["id"], "outliers_retires": s["outliers"] != "none", "bruit_retire": s["noise"], "lignes_train": s["n_train"],
                       "lignes_retirees": s["n_removed"], "meilleur": s["best"],
                       "scores_test": {r["id"]: r.get("test") for r in s["rows"] if "error" not in r}} for s in res["scenarios"]],
        "ecart_moyen_vs_brut": {k: round(float(np.mean(list(v.values()))), 4) for k, v in dl.items() if v},
    }


# ======================================================================
# PDF
# ======================================================================
PL = {
    "fr": {"title": "Comparaison des modèles : outliers et bruit", "gen": "Généré le", "synth": "Synthèse", "info": "Cadre de l'expérience",
           "ds": "Dataset", "target": "Cible", "task": "Problème", "classification": "Classification", "regression": "Régression", "metric": "Métrique principale",
           "models": "Modèles", "src_saved": "ceux de l'étape Modèles (hyperparamètres enregistrés)", "src_default": "valeurs par défaut (aucun entraînement enregistré)",
           "src_custom": "sélection personnalisée", "split": "Train / Test", "cv": "Validation croisée", "win": "Fenêtre de lissage", "none": "-",
           "sc_title": "Scénarios comparés", "scen": "Scénario", "outl": "Outliers", "noise": "Bruit", "ntr": "Lignes train", "rem": "Retirées", "detail": "Détail",
           "kept": "conservés", "removed": "retirés (One-Class SVM)", "kept_n": "conservé", "smoothed": "lissé (moyenne mobile)",
           "res_title": "Résultats par scénario", "model": "Modèle", "cvm": "CV", "train": "Train", "test": "Test", "sec": "Durée (s)",
           "cmp_title": "Comparaison globale (score test)", "fx_title": "Effet de chaque traitement (écart vs scénario A)", "best": "meilleur",
           "chart1": "Score test par modèle et par scénario", "chart2": "Écart de score vs scénario A (brut)",
           "meth_title": "Méthode et limites", "page": "Page", "err": "échec", "ovf": "sur-apprentissage possible",
           "meth": ["Mêmes modèles et mêmes hyperparamètres dans tous les scénarios : seule la donnée change.",
                    "Même split train / test que l'étape Split. Les outliers (One-Class SVM, ν = {nu:g}) ne sont retirés que du train ; le test n'est jamais modifié.",
                    "Bruit = variations à haute fréquence retirées par moyenne mobile (fenêtre {win}) sur les variables numériques, jamais sur la cible ; réservé aux séries temporelles.",
                    "Un seul entraînement par modèle et par scénario : les écarts très faibles ne sont pas significatifs.",
                    "Les autres étapes (imputation, encodage, mise à l'échelle) sont celles du prétraitement enregistré."]},
    "en": {"title": "Model comparison: outliers and noise", "gen": "Generated on", "synth": "Summary", "info": "Experiment setup",
           "ds": "Dataset", "target": "Target", "task": "Problem", "classification": "Classification", "regression": "Regression", "metric": "Main metric",
           "models": "Models", "src_saved": "those of the Models step (saved hyperparameters)", "src_default": "default values (no saved training)",
           "src_custom": "custom selection", "split": "Train / Test", "cv": "Cross-validation", "win": "Smoothing window", "none": "-",
           "sc_title": "Compared scenarios", "scen": "Scenario", "outl": "Outliers", "noise": "Noise", "ntr": "Train rows", "rem": "Removed", "detail": "Detail",
           "kept": "kept", "removed": "removed (One-Class SVM)", "kept_n": "kept", "smoothed": "smoothed (moving average)",
           "res_title": "Results per scenario", "model": "Model", "cvm": "CV", "train": "Train", "test": "Test", "sec": "Time (s)",
           "cmp_title": "Overall comparison (test score)", "fx_title": "Effect of each treatment (change vs scenario A)", "best": "best",
           "chart1": "Test score per model and scenario", "chart2": "Score change vs scenario A (raw)",
           "meth_title": "Method and limits", "page": "Page", "err": "failed", "ovf": "possible overfitting",
           "meth": ["Same models and hyperparameters in every scenario: only the data changes.",
                    "Same train / test split as the Split step. Outliers (One-Class SVM, nu = {nu:g}) are removed from train only; the test set is never modified.",
                    "Noise = high-frequency variations removed with a moving average (window {win}) on numeric features, never on the target; time series only.",
                    "One training per model and scenario: very small differences are not significant.",
                    "Other steps (imputation, encoding, scaling) are those of the saved preprocessing."]},
}


def _charts(res, lang, tmp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = PL[lang]
    cols = ["#1d4ed8", "#0ea5e9", "#f59e0b", "#10b981"]
    names = [ms.MODELS[k]["name"] for k in res["models"]]
    x = np.arange(len(names))
    scen = res["scenarios"]
    w = 0.8 / max(1, len(scen))
    out = {}

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, s in enumerate(scen):
        by = {r["id"]: r for r in s["rows"]}
        vals = [by[k].get("test") if k in by and "error" not in by[k] and by[k].get("test") is not None else np.nan for k in res["models"]]
        ax.bar(x + (i - (len(scen) - 1) / 2) * w, vals, w, label=label(s["id"], lang, True), color=cols[i % 4])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(res["metric"], fontsize=9)
    ax.set_title(P["chart1"], fontsize=10)
    ax.legend(fontsize=7, ncol=len(scen), loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    out["scores"] = Path(tmp) / "scores.png"
    fig.savefig(out["scores"], dpi=160)
    plt.close(fig)

    dl = deltas(res)
    if dl:
        fig, ax = plt.subplots(figsize=(8.2, 3.2))
        sids = [s["id"] for s in scen if s["id"] != "raw"]
        w = 0.8 / len(sids)
        for i, sid in enumerate(sids):
            ax.bar(x + (i - (len(sids) - 1) / 2) * w, [dl[sid].get(k, np.nan) for k in res["models"]], w, label=label(sid, lang, True), color=cols[(i + 1) % 4])
        ax.axhline(0, color="#475569", lw=.8)
        ax.axhspan(-NEGLIGIBLE, NEGLIGIBLE, color="#94a3b8", alpha=.15)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Δ " + res["metric"], fontsize=9)
        ax.set_title(P["chart2"], fontsize=10)
        ax.legend(fontsize=7, ncol=len(sids), loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        out["delta"] = Path(tmp) / "delta.png"
        fig.savefig(out["delta"], dpi=160)
        plt.close(fig)
    return out


def build_model_comparison_pdf(dataset, narrative="", language="fr", res=None, **run_kw):
    """PDF : résultats des modèles avec / sans outliers et avec / sans bruit. Retourne le chemin du fichier."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        from ai_interpret import AIError
        raise AIError("La génération PDF nécessite reportlab : pip install reportlab", 500)
    from agent import _fmt, _md_to_flowables, _txt

    lang = language if language in PL else "fr"
    P = PL[lang]
    res = res or run(dataset, **run_kw)
    ds = res["dataset"]
    out_dir = Path(ae.RESULTS_DIR) / ds / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"comparaison_modeles_{ds}_{datetime.now():%Y%m%d_%H%M%S}.pdf"

    ACC = colors.HexColor("#1d4ed8")
    base = getSampleStyleSheet()
    st = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=ACC, alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5, textColor=colors.HexColor("#64748b"), spaceAfter=10),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, textColor=colors.HexColor("#0f172a"), spaceBefore=14, spaceAfter=6, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, textColor=ACC, spaceBefore=8, spaceAfter=4, keepWithNext=1),
        "p": ParagraphStyle("p", parent=base["Normal"], fontSize=10, leading=14.5, spaceAfter=4),
        "b": ParagraphStyle("b", parent=base["Normal"], fontSize=10, leading=14.5, leftIndent=14, bulletIndent=3, spaceAfter=2),
        "cap": ParagraphStyle("c", parent=base["Normal"], fontSize=8.5, textColor=colors.HexColor("#64748b"), spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8, leading=10),
    }
    W = A4[0] - 4 * cm

    def table(rows, widths=None, bold_cells=(), header=True):
        cells = [[Paragraph(_txt(c), st["cell"]) for c in r] for r in rows]
        if header:
            for i in range(len(cells[0])):
                cells[0][i] = Paragraph(f"<font color='white'><b>{_txt(rows[0][i])}</b></font>", st["cell"])
        for (ri, ci) in bold_cells:
            cells[ri][ci] = Paragraph(f"<b>{_txt(rows[ri][ci])}</b>", st["cell"])
        tb = Table(cells, colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                 ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), ACC)]
        tb.setStyle(TableStyle(style))
        return tb

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(2 * cm, 1.1 * cm, _txt(f"{ds} - {P['title']}"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{P['page']} {doc.page}")
        canvas.restoreState()

    def f3(v):
        return "-" if v is None else f"{v:.3f}"

    scen = res["scenarios"]
    first = scen[0]
    clf = res["task"] == "classification"
    src = P["src_" + res.get("models_source", "saved")]
    names = ", ".join(ms.MODELS[k]["name"] for k in res["models"])

    story = [Paragraph(_txt(P["title"]), st["title"]),
             Paragraph(_txt(f"{ds}  |  {P['gen']} {datetime.now():%d/%m/%Y %H:%M}"), st["sub"])]

    story += [Paragraph(P["info"], st["h"]), table([
        [P["ds"], ds, P["target"], str(res["target"])],
        [P["task"], P["classification"] if clf else P["regression"], P["metric"], res["metric"]],
        [P["split"], f"{_fmt(first['n_train'])} / {_fmt(first['n_test'])}", P["cv"], f"{res['cv']} plis"],
        [P["models"], names, P["win"], str(res["window"]) if res.get("window") else P["none"]],
        ["", src, "", ""]], [3 * cm, 6.2 * cm, 3.4 * cm, W - 12.6 * cm], header=False)]

    story.append(Paragraph(P["synth"], st["h"]))
    if str(narrative or "").strip():
        story += _md_to_flowables(narrative, st["p"], st["b"])
        story.append(Spacer(1, 4))
    story += _md_to_flowables(auto_summary(res, lang), st["p"], st["b"])

    # scénarios
    rows = [[P["scen"], P["outl"], P["noise"], P["ntr"], P["rem"], P["detail"]]]
    for s in scen:
        det = " ".join(x for x in (s.get("outlier_note") if s["outliers"] != "none" else None, s.get("noise_note") if s["noise"] else None) if x) or "-"
        rows.append([label(s["id"], lang), P["removed"] if s["outliers"] != "none" else P["kept"], P["smoothed"] if s["noise"] else P["kept_n"],
                     _fmt(s["n_train"]), _fmt(s["n_removed"]), det])
    story += [Paragraph(P["sc_title"], st["h"]), table(rows, [4 * cm, 2.7 * cm, 2.5 * cm, 1.7 * cm, 1.9 * cm, W - 12.8 * cm])]

    # comparaison globale + graphiques
    pivot = [[P["model"]] + [label(s["id"], lang, True) for s in scen]]
    bold = []
    for i, k in enumerate(res["models"], start=1):
        line = [ms.MODELS[k]["name"]]
        for j, s in enumerate(scen, start=1):
            r = next((x for x in s["rows"] if x["id"] == k), None)
            line.append(P["err"] if (r is None or "error" in r) else f3(r.get("test")))
            if s.get("best") == k:
                bold.append((i, j))
        pivot.append(line)
    n = len(scen)
    story += [Paragraph(P["cmp_title"], st["h"]), table(pivot, [W - n * 3.2 * cm] + [3.2 * cm] * n, bold_cells=bold),
              Paragraph(_txt(f"{P['best']} = gras"), st["cap"])]

    tmp = tempfile.TemporaryDirectory()
    try:
        ch = _charts(res, lang, tmp.name)

        def img(f, h=8 * cm):
            iw, ih = ImageReader(str(f)).getSize()
            w = W
            hh = w * ih / iw
            if hh > h:
                w, hh = h * iw / ih, h
            return Image(str(f), width=w, height=hh)
        story.append(img(ch["scores"]))

        dl = deltas(res)
        if dl:
            sids = [s["id"] for s in scen if s["id"] != "raw"]
            fx = [[P["model"]] + [label(i, lang, True) for i in sids]]
            for k in res["models"]:
                fx.append([ms.MODELS[k]["name"]] + [(f"{dl[i][k]:+.3f}" if k in dl[i] else "-") for i in sids])
            fx.append(["Moyenne" if lang == "fr" else "Mean"] + [(f"{np.mean(list(dl[i].values())):+.3f}" if dl[i] else "-") for i in sids])
            story += [Paragraph(P["fx_title"], st["h"]), table(fx, [W - len(sids) * 3.6 * cm] + [3.6 * cm] * len(sids)), Spacer(1, 6), img(ch["delta"], 7 * cm)]

        # détail par scénario
        head = Paragraph(P["res_title"], st["h"])
        for s in scen:
            hdr = [P["model"], P["cvm"], P["train"], P["test"]] + (["Accuracy", "Precision", "Recall", "ROC AUC"] if clf else ["MAE", "RMSE"]) + [P["sec"]]
            body = [hdr]
            for r in s["rows"]:
                if "error" in r:
                    body.append([r["name"], P["err"]] + [""] * (len(hdr) - 2))
                    continue
                m = [f3(r.get("accuracy")), f3(r.get("precision")), f3(r.get("recall")), f3(r.get("roc_auc"))] if clf else [f3(r.get("mae")), f3(r.get("rmse"))]
                body.append([r["name"] + (" *" if r.get("overfit") else ""), f3(r.get("cv")), f3(r.get("train")), f3(r.get("test"))] + m + [f"{r['seconds']:.1f}"])
            first_w = 3.4 * cm
            rest = (W - first_w) / (len(hdr) - 1)
            block = [Paragraph(_txt(label(s["id"], lang)), st["h3"]), table(body, [first_w] + [rest] * (len(hdr) - 1))]
            story.append(KeepTogether(([head] if head is not None else []) + block))
            head = None
        story.append(Paragraph(_txt("* " + P["ovf"]), st["cap"]))

        story.append(Paragraph(P["meth_title"], st["h"]))
        for line in P["meth"]:
            story += _md_to_flowables("- " + line.format(nu=res.get("nu") or 0.05, win=res.get("window") or "-"), st["p"], st["b"])

        SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                          title=f"{P['title']} - {ds}", author="EDA Studio").build(story, onFirstPage=footer, onLaterPages=footer)
    finally:
        tmp.cleanup()
    return path
