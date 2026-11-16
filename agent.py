
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

import analysis_engine
from ai_interpret import AIError, GROQ_URL, DEFAULT_MODEL, _api_key

MAX_ROUNDS = 8               # appels d'outils enchaînés au maximum pour une question
MAX_HISTORY = 8              # derniers messages conservés
MAX_TOOL_CHARS = 3500        # taille max d'un résultat d'outil renvoyé au modèle
MAX_MSG_CHARS = 4000
MAX_AUTO_WAIT = 20           # attente max (s) avant de réessayer automatiquement après un 429


# ======================================================================
# Utilitaires
# ======================================================================
def _records(df, limit=25):
    if df is None or df.empty:
        return []
    out = df.head(limit)
    if not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    return json.loads(out.to_json(orient="records", double_precision=4))


def _resolve_dataset(name, default=None):
    """Valide un nom de dataset (on ne fait jamais confiance au modèle)."""
    available = analysis_engine.list_datasets()
    if not name:                                   # pas de nom donné -> dataset actif
        if default and default in available:
            return default
        raise ValueError(f"Précise le dataset. Datasets disponibles : {available}")
    name = analysis_engine.clean_dataset_name(name)
    if name in available:
        return name
    raise ValueError(f"Dataset inconnu : {name!r}. Datasets disponibles : {available}")


def _resolve_column(summary, name):
    cols = list((summary.get("dtypes") or {}).keys())
    low = str(name).strip().lower()
    for c in cols:
        if c == name or c.lower() == low or analysis_engine.safe_name(c) == name:
            return c
    close = [c for c in cols if low in c.lower()]
    if len(close) == 1:
        return close[0]
    raise ValueError(f"Colonne inconnue : {name!r}. Colonnes : {cols}")


def _missing_table(tables):
    """Normalise missing_report -> [{column, count, percent}] (colonnes avec manquants, tri décroissant)."""
    rep = tables.get("missing_report")
    if rep is None or rep.empty:
        return []
    cnt = next((c for c in ("Missing_Count", "missing_count") if c in rep.columns), None)
    pct = next((c for c in ("Missing_Percentage", "missing_percentage") if c in rep.columns), None)
    rows = []
    for col, r in rep.iterrows():
        c = int(r[cnt]) if cnt and pd.notna(r[cnt]) else 0
        if c > 0:
            rows.append({"column": str(col), "count": c, "percent": round(float(r[pct]), 2) if pct and pd.notna(r[pct]) else None})
    rows.sort(key=lambda x: -x["count"])
    return rows


def _class_balance(summary):
    cb = summary.get("class_balance")
    return cb if isinstance(cb, list) else []


# ======================================================================
# Outils
# ======================================================================
def tool_list_datasets(ctx):
    out = []
    for name in analysis_engine.list_datasets():
        s = analysis_engine.load_summary(name) or {}
        out.append({"dataset": name, "lignes": s.get("n_rows"), "colonnes": s.get("n_columns"), "type": s.get("dataset_type"),
                    "cible": s.get("target_class")})
    return {"datasets": out}


def tool_dataset_overview(ctx, dataset=None):
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    s = analysis_engine.load_summary(ds) or {}
    cells = max(1, (s.get("n_rows") or 0) * (s.get("n_columns") or 1))
    return {
        "dataset": ds, "fichier": s.get("file"), "lignes": s.get("n_rows"), "colonnes": s.get("n_columns"),
        "type": s.get("dataset_type"), "colonne_date": s.get("date_column"), "cible_detectee": s.get("target_class"),
        "colonnes_numeriques": s.get("numeric_columns"), "colonnes_categorielles": s.get("categorical_columns"),
        "types": s.get("dtypes"), "doublons": s.get("n_duplicates"),
        "cellules_manquantes": s.get("n_missing_cells"),
        "pct_cellules_manquantes": round(100 * (s.get("n_missing_cells") or 0) / cells, 2),
        "colonnes_constantes": s.get("constant_columns"), "colonnes_type_id": s.get("id_like_columns"),
        "haute_cardinalite": s.get("high_cardinality_columns"), "equilibre_cible": _class_balance(s),
        "notes": s.get("notes"),
    }


def tool_column_details(ctx, column, dataset=None):
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    data = analysis_engine.load_dataset_results(ds)
    s, t = data["summary"], data["tables"]
    col = _resolve_column(s, column)
    out = {"dataset": ds, "colonne": col, "type": (s.get("dtypes") or {}).get(col)}

    nun = t.get("n_unique")
    if nun is not None and col in nun.index:
        out["valeurs_uniques"] = int(nun.loc[col].iloc[0])
    out["manquants"] = next((m for m in _missing_table(t) if m["column"] == col), {"count": 0, "percent": 0})

    if col in (s.get("numeric_columns") or []):
        desc = t.get("describe_numeric")
        if desc is not None and col in desc.index:
            out["stats"] = {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in desc.loc[col].items()}
        try:
            d = analysis_engine.get_distribution_series(ds, col)
            out["forme"] = {"asymetrie_skew": d["skew"], "kurtosis": d["kurtosis"], "mediane": d["median"]}
        except Exception:
            pass
        try:
            o = analysis_engine.get_outlier_series(ds, col, k=1.5, points="outliers")
            out["outliers_iqr"] = {k: o.get(k) for k in ("n_outliers", "outlier_pct", "n_low", "n_high", "lower", "upper")}
        except Exception:
            pass
    else:
        cs = t.get("categorical_summary")
        if cs is not None and "column" in cs.columns:
            r = cs[cs["column"] == col]
            if not r.empty:
                out["resume_categoriel"] = _records(r, 1)[0]
        try:
            _summ, src = analysis_engine._resolve_source(ds, None, analysis_engine.RESULTS_DIR)
            df = analysis_engine._outlier_frame(src)
            vc = df[col].value_counts(dropna=True).head(10)
            out["valeurs_les_plus_frequentes"] = [{"valeur": str(k), "effectif": int(v)} for k, v in vc.items()]
        except Exception:
            pass
    return out


def tool_missing_values(ctx, dataset=None):
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    data = analysis_engine.load_dataset_results(ds)
    rows = _missing_table(data["tables"])
    return {"dataset": ds, "cellules_manquantes": data["summary"].get("n_missing_cells"), "colonnes_concernees": rows[:30],
            "n_colonnes_concernees": len(rows)}


def tool_outliers_overview(ctx, dataset=None):
    """Global outliers: ONE Isolation Forest on all numeric columns (not feature by feature)."""
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    try:
        r = analysis_engine.get_isolation_forest(ds)
    except Exception as e:
        return {"dataset": ds, "outliers": [], "erreur": str(e)}
    return {"dataset": ds, "methode": "Isolation Forest (toutes variables numériques, contamination auto)",
            "n_lignes": r["n"], "n_variables": r["n_features"], "n_lignes_outliers": r["n_outliers"],
            "pourcentage_outliers": r["outlier_pct"], "variables_les_plus_extremes": r["drivers"][:6],
            "lignes_les_plus_anormales": [{"ligne": x["row"], "score": x["score"], "pourquoi": x["why"]} for x in r["table"][:5]]}


def tool_correlations(ctx, dataset=None, top=10):
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    top = max(1, min(int(top or 10), 25))
    try:
        m = analysis_engine.get_correlation_matrix(ds)
    except KeyError as e:
        return {"dataset": ds, "message": str(e)}
    return {"dataset": ds, "variables": m["labels"], "paires_les_plus_correlees": m["pairs"][:top]}


def tool_target_analysis(ctx, dataset=None, target=None, top=10):
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    s = analysis_engine.load_summary(ds) or {}
    target = target or s.get("target_class")
    if not target:
        return {"dataset": ds, "message": "Aucune cible catégorielle détectée pour ce dataset."}
    out = {"dataset": ds, "cible": target, "equilibre_classes": _class_balance(s) if target == s.get("target_class") else None}
    try:
        a = analysis_engine.get_anova_importance(ds, target)
        out["anova"] = {"n_classes": a["n_classes"], "classes": a["classes"],
                        "variables": [{k: f[k] for k in ("rank", "name", "f", "p", "eta2", "score_pct")} for f in a["features"][:max(1, min(int(top or 10), 25))]]}
    except Exception as e:
        out["anova_erreur"] = str(e)
    return out


def tool_sample_rows(ctx, dataset=None, n=5):
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    head = analysis_engine.load_dataset_results(ds)["tables"].get("head")
    return {"dataset": ds, "lignes": _records(head, max(1, min(int(n or 5), 10)))}


def tool_create_pdf_report(ctx, narrative="", dataset=None, language="fr"):
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    path = build_pdf_report(ds, narrative=narrative, language=language if language in ("fr", "en") else "fr")
    ctx.setdefault("files", []).append({"name": path.name, "url": f"/results/{ds}/reports/{path.name}", "dataset": ds})
    return {"status": "ok", "fichier": path.name, "message": "Le PDF est prêt, un bouton de téléchargement est affiché à l'utilisateur."}


def _pipeline_info(ds):
    """Ce qui a été fait dans le pipeline (prétraitement, split, modèles), en chiffres : vide si l'étape n'a pas été faite."""
    info = {}
    try:
        import model_selection as ms_
        import preprocessing as pp_
        import split as sp_
    except Exception:
        return info
    try:
        rep = pp_.load_report(ds)
        if rep:
            cfg = rep.get("config") or {}
            info["pretraitement"] = {
                "lignes_avant": (rep.get("before") or {}).get("rows"), "lignes_apres": (rep.get("after") or {}).get("rows"),
                "outliers": cfg.get("outliers"), "lissage_bruit_fenetre": cfg.get("smooth_window"),
                "etapes": [f"{x.get('label')}: {x.get('detail')}"[:140] for x in rep.get("steps", []) if x.get("key") != "raw"][:8]}
    except Exception:
        pass
    try:
        sc = sp_.get_config(ds)
        if sc:
            info["split"] = {"cible": sc.get("target"), "part_test": sc.get("test_size"), "strategie": sc.get("strategy")}
    except Exception:
        pass
    try:
        runs = []
        for task in ("classification", "regression"):
            for gid, g in ms_.GROUPS.items():
                r = ms_.get_results(ds, task, gid)
                if not r:
                    continue
                res = sorted(r.get("results", []), key=lambda x: -(x.get("cv_mean") or -1e9))[:5]
                runs.append({"tache": task, "famille": g["title"], "metrique": r.get("metric"), "meilleur": next((x["name"] for x in r["results"] if x["id"] == r.get("best")), None),
                             "modeles": [{"nom": x["name"], "cv": x.get("cv_mean"), "test": x.get("test_score")} for x in res]})
        if runs:
            info["modeles"] = runs
    except Exception:
        pass
    return info


def tool_create_model_comparison_pdf(ctx, narrative="", dataset=None, language="fr", task=None, window=None):
    """2e PDF : résultats des modèles avec / sans outliers et avec / sans bruit (calcul mis en cache)."""
    import scenario_report
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    lang = language if language in ("fr", "en") else "fr"
    res = scenario_report.run(ds, task=task, window=window)
    path = scenario_report.build_model_comparison_pdf(ds, narrative=narrative, language=lang, res=res)
    ctx.setdefault("files", []).append({"name": path.name, "url": f"/results/{ds}/reports/{path.name}", "dataset": ds})
    return {"status": "ok", "fichier": path.name, "resume": scenario_report.compact(res),
            "message": "Le PDF de comparaison est prêt, un bouton de téléchargement est affiché. Commente les chiffres du résumé sans en inventer."}


def tool_dataset_brief(ctx, dataset=None):
    """Synthèse compacte en UN appel : dimensions, qualité, manquants, outliers, corrélations, cible."""
    ds = _resolve_dataset(dataset, ctx.get("dataset"))
    data = analysis_engine.load_dataset_results(ds)
    s, t = data["summary"], data["tables"]
    cells = max(1, (s.get("n_rows") or 0) * (s.get("n_columns") or 1))
    out = {
        "dataset": ds, "fichier": s.get("file"), "lignes": s.get("n_rows"), "colonnes": s.get("n_columns"),
        "type": s.get("dataset_type"), "colonne_date": s.get("date_column"), "cible": s.get("target_class"),
        "n_numeriques": len(s.get("numeric_columns") or []), "n_categorielles": len(s.get("categorical_columns") or []),
        "doublons": s.get("n_duplicates"), "cellules_manquantes": s.get("n_missing_cells"),
        "pct_manquant": round(100 * (s.get("n_missing_cells") or 0) / cells, 2),
        "colonnes_constantes": (s.get("constant_columns") or [])[:8], "colonnes_type_id": (s.get("id_like_columns") or [])[:8],
    }
    out["manquants_top"] = _missing_table(t)[:8]
    pl = _pipeline_info(ds)
    if pl:
        out["pipeline"] = pl
    o = t.get("outliers_summary")
    if o is not None and not o.empty and "outlier_count" in o.columns:
        cols = [c for c in ("column", "outlier_count", "outlier_percent") if c in o.columns]
        out["outliers_top"] = _records(o[cols].sort_values("outlier_count", ascending=False), 6)
    try:
        out["correlations_top"] = analysis_engine.get_correlation_matrix(ds)["pairs"][:6]
    except Exception:
        pass
    if s.get("target_class"):
        out["equilibre_cible"] = _class_balance(s)[:6]
        try:
            a = analysis_engine.get_anova_importance(ds, s["target_class"])
            out["variables_importantes"] = [{k: f[k] for k in ("name", "f", "eta2")} for f in a["features"][:5]]
        except Exception:
            pass
    return out


def auto_narrative(dataset, language="fr"):
    """Synthèse rédigée par règles (aucun appel IA) : sert de secours quand Groq est indisponible."""
    b = tool_dataset_brief({"dataset": dataset}, dataset)
    fr = language != "en"
    T = (lambda a, b_: a if fr else b_)
    L = [T(f"**{b['lignes']:,} lignes** et **{b['colonnes']} colonnes** ({b['n_numeriques']} numériques, {b['n_categorielles']} catégorielles).",
           f"**{b['lignes']:,} rows** and **{b['colonnes']} columns** ({b['n_numeriques']} numeric, {b['n_categorielles']} categorical).")]
    if b.get("cellules_manquantes"):
        top = ", ".join(f"{m['column']} ({m['percent']}%)" for m in b["manquants_top"][:3] if m.get("percent") is not None)
        L.append(T(f"Valeurs manquantes : {b['cellules_manquantes']:,} cellules ({b['pct_manquant']} %). Principales colonnes : {top}.",
                   f"Missing values: {b['cellules_manquantes']:,} cells ({b['pct_manquant']}%). Main columns: {top}."))
    else:
        L.append(T("Aucune valeur manquante.", "No missing values."))
    if b.get("doublons"):
        L.append(T(f"{b['doublons']:,} lignes dupliquées.", f"{b['doublons']:,} duplicated rows."))
    if b.get("colonnes_constantes"):
        L.append(T("Colonnes constantes (sans information) : ", "Constant columns (no information): ") + ", ".join(b["colonnes_constantes"]) + ".")
    if b.get("colonnes_type_id"):
        L.append(T("Colonnes de type identifiant : ", "Identifier-like columns: ") + ", ".join(b["colonnes_type_id"]) + ".")
    ot = [r for r in b.get("outliers_top", []) if (r.get("outlier_count") or 0) > 0][:3]
    if ot:
        L.append(T("Outliers (IQR) : ", "Outliers (IQR): ") + ", ".join(f"{r['column']} ({r['outlier_count']}, {r['outlier_percent']:.1f}%)" for r in ot) + ".")
    ct = [p for p in b.get("correlations_top", []) if abs(p.get("r", 0)) >= 0.5][:3]
    if ct:
        L.append(T("Corrélations fortes : ", "Strong correlations: ") + ", ".join(f"{p['a']} / {p['b']} (r = {p['r']:+.2f})" for p in ct) + ".")
    if b.get("cible") and b.get("variables_importantes"):
        L.append(T(f"Cible détectée : {b['cible']}. Variables les plus discriminantes : ", f"Detected target: {b['cible']}. Most discriminating variables: ")
                 + ", ".join(v["name"] for v in b["variables_importantes"][:4]) + ".")
    pl = (b.get("pipeline") or {})
    if pl.get("pretraitement"):
        p_ = pl["pretraitement"]
        if isinstance(p_.get("lignes_avant"), int) and isinstance(p_.get("lignes_apres"), int):
            L.append(T(f"Prétraitement fait : {p_['lignes_avant']:,} → {p_['lignes_apres']:,} lignes.", f"Preprocessing done: {p_['lignes_avant']:,} → {p_['lignes_apres']:,} rows."))
    for r_ in pl.get("modeles", []):
        if r_.get("meilleur"):
            top = r_["modeles"][0] if r_.get("modeles") else {}
            L.append(T(f"Modèles ({r_['tache']}, {r_['famille']}) : meilleur = **{r_['meilleur']}** (score test {top.get('test')}).",
                       f"Models ({r_['tache']}, {r_['famille']}): best = **{r_['meilleur']}** (test score {top.get('test')})."))
    return "\n".join("- " + x for x in L)


TOOLS = {
    "dataset_brief": (tool_dataset_brief,
                      "Synthèse compacte d'un dataset EN UN SEUL APPEL : dimensions, qualité, manquants, outliers, corrélations, cible. "
                      "À utiliser en premier pour décrire un dataset ou préparer un rapport.",
                      {"dataset": {"type": "string", "description": "Nom du dataset (défaut : dataset actif)"}}, []),
    "list_datasets": (tool_list_datasets, "Liste tous les datasets analysés disponibles (nom, taille, type, cible).", {}, []),
    "dataset_overview": (tool_dataset_overview,
                         "Vue d'ensemble d'un dataset : dimensions, types, colonnes numériques/catégorielles, date, cible, doublons, manquants.",
                         {"dataset": {"type": "string", "description": "Nom du dataset (défaut : dataset actif)"}}, []),
    "column_details": (tool_column_details,
                       "Détails d'une colonne : type, manquants, statistiques, asymétrie, outliers (numérique) ou valeurs fréquentes (catégorielle).",
                       {"column": {"type": "string", "description": "Nom de la colonne"}, "dataset": {"type": "string"}}, ["column"]),
    "missing_values": (tool_missing_values, "Colonnes avec valeurs manquantes (effectif et pourcentage).",
                       {"dataset": {"type": "string"}}, []),
    "outliers_overview": (tool_outliers_overview, "Nombre global de lignes outliers du dataset (Isolation Forest sur toutes les variables numériques), avec les variables les plus extrêmes et les lignes les plus anormales.",
                          {"dataset": {"type": "string"}}, []),
    "correlations": (tool_correlations, "Paires de variables numériques les plus corrélées (Pearson).",
                     {"dataset": {"type": "string"}, "top": {"type": "integer", "description": "Nombre de paires (max 25)"}}, []),
    "target_analysis": (tool_target_analysis,
                        "Équilibre des classes de la cible et importance des variables par ANOVA (test F, p-value, eta²).",
                        {"dataset": {"type": "string"}, "target": {"type": "string", "description": "Cible catégorielle (défaut : cible détectée)"},
                         "top": {"type": "integer"}}, []),
    "sample_rows": (tool_sample_rows, "Premières lignes du dataset.",
                    {"dataset": {"type": "string"}, "n": {"type": "integer", "description": "1 à 10"}}, []),
    "create_pdf_report": (tool_create_pdf_report,
                          "Génère le RAPPORT GÉNÉRAL PDF du dataset : EDA (chiffres, tableaux, graphiques) + prétraitement, split et résultats des modèles "
                          "quand ces étapes ont été faites. "
                          "Appelle d'abord les outils d'analyse, puis passe dans `narrative` une synthèse rédigée (6 à 12 lignes, Markdown simple) "
                          "basée uniquement sur les résultats obtenus.",
                          {"narrative": {"type": "string", "description": "Synthèse rédigée à inclure en tête du rapport"},
                           "dataset": {"type": "string"}, "language": {"type": "string", "enum": ["fr", "en"]}}, ["narrative"]),
}


TOOLS["create_model_comparison_pdf"] = (
    tool_create_model_comparison_pdf,
    "Génère le 2e PDF : résultats des modèles utilisés (ceux de l'étape Modèles, mêmes hyperparamètres) AVEC / SANS outliers et AVEC / SANS bruit "
    "(4 scénarios ; 2 seulement si le dataset n'est pas une série temporelle). Peut prendre de quelques secondes à quelques minutes la première fois "
    "(les modèles sont ré-entraînés), instantané ensuite. Renvoie un résumé chiffré à commenter. `narrative` est facultatif.",
    {"narrative": {"type": "string", "description": "Texte facultatif placé en tête du PDF"}, "dataset": {"type": "string"},
     "language": {"type": "string", "enum": ["fr", "en"]},
     "task": {"type": "string", "enum": ["classification", "regression"], "description": "Par défaut : celui des modèles déjà entraînés"},
     "window": {"type": "integer", "description": "Fenêtre de la moyenne mobile pour retirer le bruit (défaut : réglage du prétraitement, sinon 5)"}}, [])


def _tool_specs():
    return [{"type": "function", "function": {"name": n, "description": d,
                                              "parameters": {"type": "object", "properties": p, "required": r}}}
            for n, (_f, d, p, r) in TOOLS.items()]


def _run_tool(ctx, name, raw_args):
    if name not in TOOLS:
        return {"error": f"Outil inconnu : {name}"}
    try:
        args = json.loads(raw_args) if raw_args else {}
        if not isinstance(args, dict):
            args = {}
    except ValueError:
        args = {}
    allowed = set(TOOLS[name][2])
    args = {k: v for k, v in args.items() if k in allowed}
    try:
        return TOOLS[name][0](ctx, **args)
    except Exception as e:                 # l'erreur est renvoyée au modèle, qui peut se corriger
        return {"error": str(e)[:400]}


# ======================================================================
# Boucle de l'agent
# ======================================================================
def _system_prompt(dataset):
    return (
        "Tu es l'assistant data d'une application de pipeline ML (EDA, prétraitement, split, sélection de modèles). Tu aides à comprendre n'importe quel "
        "dataset déjà analysé et tu peux générer DEUX types de rapports PDF.\n"
        f"Dataset actuellement ouvert par l'utilisateur : {dataset or 'aucun'}. Sans précision, c'est celui dont il parle.\n"
        "Règles :\n"
        "- Réponds dans la langue de l'utilisateur (français par défaut), de façon claire et concise, en Markdown simple.\n"
        "- N'invente JAMAIS un chiffre : appuie-toi uniquement sur les résultats des outils. Si une info manque, dis-le.\n"
        "- Pour décrire un dataset, appelle dataset_brief (un seul appel donne l'essentiel, y compris l'état du pipeline) ; les autres outils servent à approfondir une question précise.\n"
        "- Rapport GÉNÉRAL (EDA + prétraitement + split + modèles) : appelle dataset_brief, rédige une synthèse fidèle, puis appelle create_pdf_report UNE seule fois.\n"
        "- Rapport de COMPARAISON DES MODÈLES (avec / sans bruit, avec / sans outliers) : appelle create_model_comparison_pdf UNE seule fois, puis commente "
        "les chiffres du résumé renvoyé. Si l'utilisateur demande les deux rapports, appelle les deux outils.\n"
        "- Ne recopie pas de lien : un bouton de téléchargement apparaît tout seul.\n"
        "- Tu ne peux ni modifier ni supprimer des données ; tu lis les résultats d'analyse (le seul calcul que tu lances est la comparaison des modèles).\n"
    )


def _retry_wait(r):
    """Secondes à attendre avant de réessayer (en-tête Retry-After ou message Groq « try again in 7m12.5s »)."""
    try:
        if r.headers.get("retry-after"):
            return float(r.headers["retry-after"])
    except ValueError:
        pass
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", r.text or "")
    if not m:
        return None
    mult = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * mult[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)))


def _human(sec):
    sec = int(sec + 0.999)
    return f"{sec} s" if sec < 60 else f"{round(sec / 60)} min" if sec < 3600 else f"{sec / 3600:.1f} h"


def _groq(messages, tools=None, retry=True, retry_429=2):
    payload = {"model": os.environ.get("GROQ_MODEL", DEFAULT_MODEL), "messages": messages, "temperature": 0.2, "max_tokens": 1200}
    if tools:
        payload.update(tools=tools, tool_choice="auto")
    try:
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
                          json=payload, timeout=60)
    except requests.Timeout:
        raise AIError("Groq a mis trop de temps à répondre, réessaie.", 504)
    except requests.RequestException as e:
        raise AIError(f"Impossible de joindre Groq : {e}", 502)

    if r.status_code == 401:
        raise AIError("Clé Groq refusée (401). Vérifie GROQ_API_KEY.", 401)
    if r.status_code == 429:
        wait = _retry_wait(r)
        if retry_429 > 0 and wait is not None and wait <= MAX_AUTO_WAIT:
            time.sleep(wait + 0.5)                                   # quota par minute : on attend puis on réessaie
            return _groq(messages, tools, retry=retry, retry_429=retry_429 - 1)
        hint = f" Réessaie dans environ {_human(wait)}." if wait else " Patiente quelques secondes."
        raise AIError("Limite Groq atteinte (429)." + hint, 429)
    if r.status_code == 400 and retry and "tool_use_failed" in r.text:
        return _groq(messages, tools, retry=False, retry_429=retry_429)          # le modèle a mal formé un appel d'outil : on réessaie une fois
    if not r.ok:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:200])
        except ValueError:
            msg = r.text[:200]
        raise AIError(f"Erreur Groq {r.status_code} : {msg}", 502)
    return r.json()["choices"][0]["message"]


def _chat_llm(history, dataset=None, ctx=None):
    """Boucle d'agent avec outils (Groq)."""
    clean = []
    for m in (history or [])[-MAX_HISTORY:]:
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip():
            clean.append({"role": m["role"], "content": str(m["content"])[:MAX_MSG_CHARS]})
    if not clean or clean[-1]["role"] != "user":
        raise AIError("Message vide.", 400)

    try:
        dataset = _resolve_dataset(dataset) if dataset else None
    except ValueError:
        dataset = None

    ctx = ctx if ctx is not None else {"dataset": dataset, "files": []}
    messages = [{"role": "system", "content": _system_prompt(dataset)}] + clean
    steps = []

    for _ in range(MAX_ROUNDS):
        msg = _groq(messages, _tool_specs())
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"reply": (msg.get("content") or "").strip() or "Je n'ai pas de réponse à te donner.",
                    "files": ctx["files"], "steps": steps}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c["function"]
            steps.append(fn["name"])
            result = json.dumps(_run_tool(ctx, fn["name"], fn.get("arguments")), ensure_ascii=False, default=str)
            if len(result) > MAX_TOOL_CHARS:
                result = result[:MAX_TOOL_CHARS] + "…(tronqué)"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    # trop d'enchaînements : on demande une réponse finale sans outils
    messages.append({"role": "user", "content": "Réponds maintenant avec les informations déjà obtenues."})
    msg = _groq(messages)
    return {"reply": (msg.get("content") or "").strip(), "files": ctx["files"], "steps": steps}


def _fallback(dataset, user_text, ctx, err):
    """Groq indisponible (quota, réseau) : réponse et PDF générés par règles, sans IA."""
    lang = "en" if re.search(r"\b(please|describe|generate|summary|report|the)\b", user_text, re.I) else "fr"
    wants_pdf = bool(re.search(r"\b(pdf|rapport|report)\b", user_text, re.I))
    wants_cmp = bool(re.search(r"(bruit|noise|outlier|aberrant|compar|mod[eè]les?\b|models?\b)", user_text, re.I))
    wants_gen = bool(re.search(r"(g[ée]n[ée]ral|complet|full|global|overall)", user_text, re.I))
    text = auto_narrative(dataset, lang)
    files = list(ctx.get("files") or [])
    extra = ""
    if wants_pdf and not files:
        if not wants_cmp or wants_gen:
            path = build_pdf_report(dataset, narrative=text, language=lang)
            files.append({"name": path.name, "url": f"/results/{dataset}/reports/{path.name}", "dataset": dataset})
        if wants_cmp:
            try:
                import scenario_report
                path = scenario_report.build_model_comparison_pdf(dataset, language=lang)
                files.append({"name": path.name, "url": f"/results/{dataset}/reports/{path.name}", "dataset": dataset})
            except Exception as e:
                extra = (f"Le PDF de comparaison des modèles n'a pas pu être généré : {e}\n\n" if lang == "fr"
                         else f"The model comparison PDF could not be generated: {e}\n\n")
    head = (f"⚠️ {err} Réponse générée sans IA (synthèse automatique).\n\n" if lang == "fr"
            else f"⚠️ {err} Answer generated without AI (automatic summary).\n\n")
    if files:
        head += ("Le(s) rapport(s) PDF est/sont prêt(s).\n\n" if lang == "fr" else "The PDF report(s) is/are ready.\n\n")
    return {"reply": head + extra + text, "files": files, "steps": ["mode_sans_ia"], "degraded": True}


def chat(history, dataset=None):
    """history : [{role: 'user'|'assistant', content: str}, ...]  ->  {reply, files, steps}"""
    try:
        ds = _resolve_dataset(dataset) if dataset else None
    except ValueError:
        ds = None
    ctx = {"dataset": ds, "files": []}
    try:
        return _chat_llm(history, ds, ctx)
    except AIError as e:
        if e.status in (429, 502, 504) and ds:
            last = next((m.get("content", "") for m in reversed(history or []) if m.get("role") == "user"), "")
            return _fallback(ds, str(last), ctx, e)
        raise


# ======================================================================
# Rapport PDF (reportlab)
# ======================================================================
LABELS = {
    "fr": {
        "title": "Rapport général du dataset", "generated": "Généré le", "file": "Fichier", "key": "Chiffres clés",
        "rows": "Lignes", "cols": "Colonnes", "type": "Type de dataset", "dups": "Doublons", "missing": "Cellules manquantes",
        "date": "Colonne date", "target": "Cible détectée", "none": "aucune", "synth": "Synthèse", "columns": "Colonnes",
        "col": "Colonne", "dtype": "Type", "miss": "Manq. %", "uniq": "Uniques", "mean": "Moyenne", "std": "Écart-type",
        "min": "Min", "max": "Max", "miss_title": "Valeurs manquantes", "no_miss": "Aucune valeur manquante.",
        "count": "Nombre", "pct": "%", "out_title": "Valeurs aberrantes (IQR, k = 1,5)", "lower": "Borne basse", "upper": "Borne haute",
        "corr_title": "Corrélations (Pearson)", "pair": "Paire", "tgt_title": "Cible et importance des variables (ANOVA)",
        "class": "Classe", "var": "Variable", "dist_title": "Distributions", "others": "autres colonnes non affichées",
        "pp_title": "Prétraitement", "step": "Étape", "detail": "Détail", "split_title": "Split train / test", "test_size": "Part de test", "strategy": "Stratégie",
        "excluded": "Variables exclues", "models_title": "Modèles", "model": "Modèle", "cvm": "CV", "train": "Train", "test": "Test", "best": "Meilleur modèle", "task": "Problème",
        "page": "Page", "no_corr": "Pas de matrice de corrélation disponible.", "anova_note": "F élevé = la variable sépare bien les classes de la cible ; p < 0,05 = écart significatif ; eta2 = part de variance expliquée.",
    },
    "en": {
        "title": "General dataset report", "generated": "Generated on", "file": "File", "key": "Key figures",
        "rows": "Rows", "cols": "Columns", "type": "Dataset type", "dups": "Duplicates", "missing": "Missing cells",
        "date": "Date column", "target": "Detected target", "none": "none", "synth": "Summary", "columns": "Columns",
        "col": "Column", "dtype": "Type", "miss": "Miss. %", "uniq": "Unique", "mean": "Mean", "std": "Std dev",
        "min": "Min", "max": "Max", "miss_title": "Missing values", "no_miss": "No missing values.",
        "count": "Count", "pct": "%", "out_title": "Outliers (IQR, k = 1.5)", "lower": "Lower bound", "upper": "Upper bound",
        "corr_title": "Correlations (Pearson)", "pair": "Pair", "tgt_title": "Target and feature importance (ANOVA)",
        "class": "Class", "var": "Variable", "dist_title": "Distributions", "others": "more columns not shown",
        "pp_title": "Preprocessing", "step": "Step", "detail": "Detail", "split_title": "Train / test split", "test_size": "Test share", "strategy": "Strategy",
        "excluded": "Excluded variables", "models_title": "Models", "model": "Model", "cvm": "CV", "train": "Train", "test": "Test", "best": "Best model", "task": "Problem",
        "page": "Page", "no_corr": "No correlation matrix available.", "anova_note": "High F = the variable separates the target classes well; p < 0.05 = significant difference; eta2 = share of variance explained.",
    },
}


def _txt(v):
    """Texte sûr pour reportlab (police Helvetica = Latin-1) : on échappe le XML et on remplace les glyphes absents."""
    s = "" if v is None else str(v)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for a, b in (("→", "->"), ("←", "<-"), ("ν", "nu"), ("≤", "&lt;="), ("≥", "&gt;="), ("×", "x"), ("η", "eta"), ("Δ", "delta "), ("−", "-"), ("\u202f", " "), ("\u00a0", " ")):
        s = s.replace(a, b)
    return s.encode("cp1252", "replace").decode("cp1252")


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer() and abs(v) < 1e15):
        return f"{int(v):,}".replace(",", " ")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return _txt(v)
    return f"{x:,.0f}".replace(",", " ") if abs(x) >= 1000 else f"{x:.{max(nd, 4)}g}"


def _md_to_flowables(md, style, bullet_style):
    from reportlab.platypus import Paragraph
    out = []
    for line in str(md or "").splitlines():
        t = line.strip()
        if not t:
            continue
        bullet = re.match(r"^([-*•]|\d+[.)])\s+", t)
        body = _txt(re.sub(r"^([-*•]|\d+[.)])\s+", "", t) if bullet else re.sub(r"^#+\s*", "", t))
        body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
        out.append(Paragraph(body, bullet_style, bulletText="•") if bullet else Paragraph(body, style))
    return out


def build_pdf_report(dataset, narrative="", language="fr"):
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
    except ImportError:
        raise AIError("La génération PDF nécessite reportlab : pip install reportlab", 500)

    L = LABELS.get(language, LABELS["fr"])
    data = analysis_engine.load_dataset_results(dataset)
    if not data:
        raise AIError("Dataset introuvable", 404)
    s, t = data["summary"], data["tables"]
    plots_dir = Path(analysis_engine.RESULTS_DIR) / analysis_engine.clean_dataset_name(dataset) / "plots"
    out_dir = plots_dir.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rapport_{analysis_engine.clean_dataset_name(dataset)}_{datetime.now():%Y%m%d_%H%M%S}.pdf"

    # ---- styles
    ACC = colors.HexColor("#1d4ed8")
    base = getSampleStyleSheet()
    st = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, textColor=ACC, alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5, textColor=colors.HexColor("#64748b"), spaceAfter=10),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, textColor=colors.HexColor("#0f172a"),
                            spaceBefore=14, spaceAfter=6, keepWithNext=1),
        "p": ParagraphStyle("p", parent=base["Normal"], fontSize=10, leading=14.5, spaceAfter=4),
        "b": ParagraphStyle("b", parent=base["Normal"], fontSize=10, leading=14.5, leftIndent=14, bulletIndent=3, spaceAfter=2),
        "cap": ParagraphStyle("c", parent=base["Normal"], fontSize=8.5, textColor=colors.HexColor("#64748b"), spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8, leading=10),
    }
    W = A4[0] - 4 * cm

    def table(rows, widths=None, header=True):
        cells = [[Paragraph(_txt(c), st["cell"]) if isinstance(c, str) else Paragraph(_txt(c), st["cell"]) for c in r] for r in rows]
        tb = Table(cells, colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                 ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), ACC), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
            for i in range(len(cells[0])):                      # texte blanc en en-tête
                cells[0][i] = Paragraph(f"<font color='white'><b>{_txt(rows[0][i])}</b></font>", st["cell"])
            tb = Table(cells, colWidths=widths, repeatRows=1)
        tb.setStyle(TableStyle(style))
        return tb

    def image(name, caption, max_h=9.5 * cm):
        f = plots_dir / name
        if not f.exists():
            return None
        iw, ih = ImageReader(str(f)).getSize()
        w = W
        h = w * ih / iw
        if h > max_h:
            w, h = max_h * iw / ih, max_h
        return KeepTogether([Image(str(f), width=w, height=h), Paragraph(_txt(caption), st["cap"])])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(2 * cm, 1.1 * cm, _txt(f"{dataset} - {L['title']}"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{L['page']} {doc.page}")
        canvas.restoreState()

    story = [Paragraph(_txt(L["title"]), st["title"]),
             Paragraph(_txt(f"{dataset}  |  {L['generated']} {datetime.now():%d/%m/%Y %H:%M}  |  {L['file']} : {s.get('file') or '-'}"), st["sub"])]

    # ---- chiffres clés
    cells = max(1, (s.get("n_rows") or 0) * (s.get("n_columns") or 1))
    miss_pct = 100 * (s.get("n_missing_cells") or 0) / cells
    story += [Paragraph(L["key"], st["h"]), table([
        [L["rows"], _fmt(s.get("n_rows")), L["cols"], _fmt(s.get("n_columns"))],
        [L["type"], s.get("dataset_type") or "-", L["dups"], _fmt(s.get("n_duplicates"))],
        [L["missing"], f"{_fmt(s.get('n_missing_cells'))} ({miss_pct:.2f} %)", L["date"], s.get("date_column") or L["none"]],
        [L["target"], s.get("target_class") or L["none"], "", ""]], [3.6 * cm, 5 * cm, 3.6 * cm, W - 12.2 * cm], header=False)]

    # ---- synthèse rédigée par l'agent
    if str(narrative or "").strip():
        story += [Paragraph(L["synth"], st["h"])] + _md_to_flowables(narrative, st["p"], st["b"])

    # ---- colonnes
    dtypes = s.get("dtypes") or {}
    desc = t.get("describe_numeric")
    nun = t.get("n_unique")
    miss = {m["column"]: m for m in _missing_table(t)}
    rows = [[L["col"], L["dtype"], L["miss"], L["uniq"], L["mean"], L["std"], L["min"], L["max"]]]
    for c, dt in list(dtypes.items())[:40]:
        d = desc.loc[c] if (desc is not None and c in desc.index) else None
        rows.append([c, str(dt).split("[")[0], f"{(miss.get(c) or {}).get('percent') or 0:.1f}",
                     _fmt(int(nun.loc[c].iloc[0])) if nun is not None and c in nun.index else "-",
                     _fmt(d["mean"]) if d is not None else "-", _fmt(d["std"]) if d is not None else "-",
                     _fmt(d["min"]) if d is not None else "-", _fmt(d["max"]) if d is not None else "-"])
    story += [Paragraph(L["columns"], st["h"]), table(rows, [3.8 * cm, 2.2 * cm, 1.5 * cm, 1.6 * cm, 2.1 * cm, 2.1 * cm, 1.9 * cm, W - 15.2 * cm])]
    if len(dtypes) > 40:
        story.append(Paragraph(_txt(f"+ {len(dtypes) - 40} {L['others']}"), st["cap"]))

    # ---- manquants
    story.append(Paragraph(L["miss_title"], st["h"]))
    if miss:
        story.append(table([[L["col"], L["count"], L["pct"]]] + [[m["column"], _fmt(m["count"]), f"{m['percent']:.2f}"] for m in list(miss.values())[:12]],
                           [W - 7 * cm, 3.5 * cm, 3.5 * cm]))
        img = image("missing_values.png", L["miss_title"], 8 * cm)
        story += [Spacer(1, 6), img] if img else []
    else:
        story.append(Paragraph(L["no_miss"], st["p"]))

    # ---- outliers
    o = t.get("outliers_summary")
    if o is not None and not o.empty and "outlier_count" in o.columns:
        o = o.sort_values("outlier_count", ascending=False).head(12)
        story += [Paragraph(L["out_title"], st["h"]),
                  table([[L["col"], L["lower"], L["upper"], L["count"], L["pct"]]] +
                        [[r["column"], _fmt(r.get("lower_bound")), _fmt(r.get("upper_bound")), _fmt(r["outlier_count"]), f"{float(r['outlier_percent']):.2f}"]
                         for _, r in o.iterrows()], [W - 13.5 * cm, 3 * cm, 3 * cm, 3.5 * cm, 4 * cm])]

    # ---- corrélations
    story.append(Paragraph(L["corr_title"], st["h"]))
    try:
        pairs = analysis_engine.get_correlation_matrix(dataset)["pairs"][:10]
        story.append(table([[L["pair"], "r"]] + [[f"{p['a']}  x  {p['b']}", f"{p['r']:+.3f}"] for p in pairs], [W - 3 * cm, 3 * cm]))
        img = image("correlation_heatmap.png", L["corr_title"], 10 * cm)
        story += [Spacer(1, 6), img] if img else []
    except KeyError:
        story.append(Paragraph(L["no_corr"], st["p"]))

    # ---- cible + ANOVA
    target = s.get("target_class")
    if target:
        story.append(Paragraph(f"{L['tgt_title']} : {_txt(target)}", st["h"]))
        cb = _class_balance(s)
        if cb:
            story.append(table([[L["class"], L["count"], L["pct"]]] + [[str(r.get(target, list(r.values())[0])), _fmt(r.get("count")), f"{float(r.get('percent', 0)):.2f}"] for r in cb[:12]],
                               [W - 7 * cm, 3.5 * cm, 3.5 * cm]))
        try:
            a = analysis_engine.get_anova_importance(dataset, target)
            story += [Spacer(1, 8), table([["#", L["var"], "F", "p-value", "eta2"]] +
                                          [[str(f["rank"]), f["name"], _fmt(f["f"], 4), "n/a" if f["p"] is None else ("<0.0001" if f["p"] < 1e-4 else f"{f['p']:.4f}"), f"{f['eta2']:.3f}"]
                                           for f in a["features"][:10]], [1 * cm, W - 11.5 * cm, 3 * cm, 3.5 * cm, 4 * cm]),
                      Paragraph(_txt(L["anova_note"]), st["cap"])]
        except Exception:
            pass
        img = image("class_balance.png", L["tgt_title"], 8 * cm)
        story += [Spacer(1, 6), img] if img else []

    # ---- distributions (4 premières variables)
    dist = sorted(plots_dir.glob("distribution_*.png"))[:4] if plots_dir.exists() else []
    if dist:
        imgs = [image(f.name, f.stem.replace("distribution_", "").replace("_", " ").strip(), 6.5 * cm) for f in dist]
        imgs = [i for i in imgs if i]
        if imgs:                                               # le titre reste collé à la première image
            story.append(KeepTogether([Paragraph(L["dist_title"], st["h"]), imgs[0]]))
            story += imgs[1:]

    # ---- suite du pipeline : prétraitement, split, modèles (seulement ce qui a été fait ; l'EDA ci-dessus n'en dépend jamais)
    try:
        import model_selection as ms_
        import preprocessing as pp_
        import split as sp_
        rep = pp_.load_report(dataset)
        if rep:
            b, a = rep.get("before") or {}, rep.get("after") or {}
            story.append(Paragraph(L["pp_title"], st["h"]))
            story.append(Paragraph(_txt(f"{L['rows']} : {_fmt(b.get('rows'))} -> {_fmt(a.get('rows'))}   |   {L['cols']} : {_fmt(b.get('cols'))} -> {_fmt(a.get('cols'))}"
                                        f"   |   {L['missing']} : {_fmt(b.get('missing'))} -> {_fmt(a.get('missing'))}"), st["p"]))
            steps = [x for x in rep.get("steps", []) if x.get("key") != "raw"]
            if steps:
                story.append(table([[L["step"], L["detail"], L["rows"], L["cols"]]] +
                                   [[x.get("label") or "", x.get("detail") or "-", _fmt(x.get("rows")), _fmt(x.get("cols"))] for x in steps],
                                   [3.4 * cm, W - 3.4 * cm - 4.4 * cm, 2.2 * cm, 2.2 * cm]))
        sc = sp_.get_config(dataset)
        if sc:
            story += [Paragraph(L["split_title"], st["h"]), table([
                [L["target"], str(sc.get("target") or "-"), L["test_size"], f"{float(sc.get('test_size', 0)) * 100:.0f} %"],
                [L["strategy"], str(sc.get("strategy") or "auto"), L["excluded"], ", ".join(map(str, sc.get("excluded") or [])) or "-"]],
                [3 * cm, 5.5 * cm, 3.4 * cm, W - 11.9 * cm], header=False)]
        first = True
        for task in ("classification", "regression"):
            for gid, g in ms_.GROUPS.items():
                r = ms_.get_results(dataset, task, gid)
                if not r:
                    continue
                if first:
                    story.append(Paragraph(L["models_title"], st["h"]))
                    first = False
                res = sorted(r.get("results", []), key=lambda x: -(x.get("cv_mean") or -1e9))
                bestname = next((x["name"] for x in res if x["id"] == r.get("best")), "-")
                head = Paragraph(_txt(f"{task} - {g['title']}  ({L['best']} : {bestname}, {r.get('metric')})"), st["p"])
                rows = [[L["model"], L["cvm"], L["train"], L["test"]]] + [
                    [x["name"], _fmt(x.get("cv_mean")), _fmt(x.get("train_score")), _fmt(x.get("test_score"))] for x in res]
                story.append(KeepTogether([head, table(rows, [W - 9 * cm, 3 * cm, 3 * cm, 3 * cm]), Spacer(1, 6)]))
    except Exception:
        pass

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title=f"{L['title']} - {dataset}", author="EDA Studio").build(story, onFirstPage=footer, onLaterPages=footer)
    return path
