"""
Cahier des charges + résumé du pipeline, exportés en PDF (reportlab).

Le document fusionne deux choses :
  * le CAHIER DES CHARGES : contexte, données, exigences (fonctionnelles / non fonctionnelles), livrables, risques ;
  * le RÉSUMÉ DU PIPELINE : ce qui a réellement été fait (prétraitement, split, modèles), avec les chiffres.

Chaque exigence reçoit un statut calculé à partir des résultats déjà présents (Réalisé / À faire) :
rien n'est inventé, tout vient d'analysis_engine, preprocessing, split et model_selection.
Fonctionne pour n'importe quel dataset déjà analysé.
"""
from datetime import datetime
from pathlib import Path

import analysis_engine

L10N = {
    "fr": {
        "title": "Cahier des charges et résumé du pipeline", "generated": "Généré le", "file": "Fichier",
        "ctx": "1. Contexte et objectif", "data": "2. Description des données", "req": "3. Exigences fonctionnelles",
        "pipe": "4. Résumé du pipeline", "nfr": "5. Exigences non fonctionnelles et contraintes",
        "deliv": "6. Livrables", "risk": "7. Risques et points d'attention", "valid": "8. Critères de validation",
        "rows": "Lignes", "cols": "Colonnes", "type": "Type de dataset", "target": "Cible détectée", "date": "Colonne date",
        "dups": "Doublons", "missing": "Cellules manquantes", "none": "aucune", "num": "Variables numériques", "cat": "Variables catégorielles",
        "id": "ID", "need": "Exigence", "status": "Statut", "done": "Réalisé", "todo": "À faire", "demand": "Sur demande",
        "step": "Étape", "detail": "Détail", "model": "Modèle", "cv": "Val", "test": "Test", "best": "Meilleur modèle",
        "test_size": "Part de test", "strategy": "Stratégie", "excluded": "Variables exclues", "auto": "auto",
        "no_pp": "Prétraitement non encore réalisé.", "no_split": "Split train / test non encore configuré.",
        "no_models": "Aucun modèle n'a encore été entraîné.", "no_risk": "Aucun risque majeur détecté automatiquement.",
        "ctx_auto": "Ce document décrit le besoin et l'état d'avancement du pipeline de machine learning appliqué au dataset « {ds} » "
                    "({rows} lignes, {cols} colonnes). L'objectif est de passer de l'analyse exploratoire à la sélection de modèles "
                    "de façon reproductible, avec des rapports exportables.",
        "page": "Page", "item": "Point", "check": "Contrôle",
        "r_eda": "Analyser le dataset : dimensions, types, valeurs manquantes, aberrantes, corrélations, cible.",
        "r_pp": "Prétraiter les données (valeurs aberrantes, bruit, imputation, encodage selon la configuration choisie).",
        "r_split": "Séparer les données en jeu d'entraînement et de test avec une stratégie adaptée au dataset.",
        "r_model": "Comparer plusieurs familles de modèles sur une validation (20 % du train) et retenir le meilleur selon la métrique : F1 et AUC (classification), R² (régression).",
        "r_cmp": "Comparer les modèles avec / sans valeurs aberrantes et avec / sans bruit.",
        "r_pdf": "Exporter les résultats en rapports PDF via l'agent.",
        "nf1": "Application web (Flask) utilisable sur n'importe quel dataset importé, sans code spécifique à un jeu de données.",
        "nf2": "Résultats d'étapes mis en cache pour éviter de recalculer ; configuration du prétraitement et du split sauvegardée (reproductibilité).",
        "nf3": "Aucune donnée n'est modifiée ni supprimée par l'agent : il lit les résultats d'analyse.",
        "nf4": "Rapports PDF lisibles, paginés, avec tableaux chiffrés ; aucun chiffre inventé par l'agent.",
        "d1": "Rapport général PDF : EDA, prétraitement, split et modèles.",
        "d2": "Rapport PDF de comparaison des modèles (avec / sans outliers, avec / sans bruit).",
        "d3": "Le présent cahier des charges avec résumé du pipeline.",
        "v1": "Aucune valeur manquante non traitée avant l'entraînement.",
        "v2": "Le jeu de test n'est jamais utilisé pour choisir le modèle (sélection sur une validation issue du train).",
        "v3": "Écart entre score d'entraînement et de test du meilleur modèle examiné (surapprentissage).",
        "v4": "Résultats reproductibles à partir de la configuration sauvegardée.",
        "rk_miss": "{pct:.1f} % de cellules manquantes : prévoir une imputation.", "rk_dup": "{n} lignes dupliquées à examiner.",
        "rk_const": "Colonnes constantes à écarter : {c}.", "rk_id": "Colonnes de type identifiant (à exclure du modèle) : {c}.",
        "rk_imb": "Cible déséquilibrée (classe majoritaire {mx:.0f} %, minoritaire {mn:.0f} %) : privilégier F1 / AUC plutôt que l'accuracy.",
        "rk_out": "Valeurs aberrantes importantes sur « {c} » ({p:.1f} %).", "rk_nopp": "Prétraitement non réalisé : les modèles ne doivent pas être comparés avant.",
        "rk_over": "Écart train/test élevé pour {m} ({a:.3f} vs {b:.3f}) : risque de surapprentissage.",
    },
    "en": {
        "title": "Specification and pipeline summary", "generated": "Generated on", "file": "File",
        "ctx": "1. Context and objective", "data": "2. Data description", "req": "3. Functional requirements",
        "pipe": "4. Pipeline summary", "nfr": "5. Non-functional requirements and constraints",
        "deliv": "6. Deliverables", "risk": "7. Risks and watch points", "valid": "8. Acceptance criteria",
        "rows": "Rows", "cols": "Columns", "type": "Dataset type", "target": "Detected target", "date": "Date column",
        "dups": "Duplicates", "missing": "Missing cells", "none": "none", "num": "Numeric variables", "cat": "Categorical variables",
        "id": "ID", "need": "Requirement", "status": "Status", "done": "Done", "todo": "To do", "demand": "On demand",
        "step": "Step", "detail": "Detail", "model": "Model", "cv": "Val", "test": "Test", "best": "Best model",
        "test_size": "Test share", "strategy": "Strategy", "excluded": "Excluded variables", "auto": "auto",
        "no_pp": "Preprocessing not done yet.", "no_split": "Train / test split not configured yet.",
        "no_models": "No model has been trained yet.", "no_risk": "No major risk detected automatically.",
        "ctx_auto": "This document describes the requirements and the progress of the machine learning pipeline applied to dataset \"{ds}\" "
                    "({rows} rows, {cols} columns). The goal is to go from exploratory analysis to model selection "
                    "in a reproducible way, with exportable reports.",
        "page": "Page", "item": "Item", "check": "Check",
        "r_eda": "Analyse the dataset: dimensions, types, missing and outlier values, correlations, target.",
        "r_pp": "Preprocess the data (outliers, noise, imputation, encoding according to the chosen configuration).",
        "r_split": "Split the data into train and test sets with a strategy suited to the dataset.",
        "r_model": "Compare several model families on a validation set (20 % of train) and keep the best one for the metric: F1 and AUC (classification), R² (regression).",
        "r_cmp": "Compare models with / without outliers and with / without noise.",
        "r_pdf": "Export the results as PDF reports through the agent.",
        "nf1": "Web application (Flask) usable on any imported dataset, with no dataset-specific code.",
        "nf2": "Step results cached to avoid recomputation; preprocessing and split configuration saved (reproducibility).",
        "nf3": "The agent never modifies or deletes data: it reads analysis results.",
        "nf4": "Readable, paginated PDF reports with numeric tables; no figure invented by the agent.",
        "d1": "General PDF report: EDA, preprocessing, split and models.",
        "d2": "Model comparison PDF report (with / without outliers, with / without noise).",
        "d3": "This specification with pipeline summary.",
        "v1": "No untreated missing value before training.",
        "v2": "The test set is never used to choose the model (selection on a validation set taken from train).",
        "v3": "Gap between train and test score of the best model reviewed (overfitting).",
        "v4": "Results reproducible from the saved configuration.",
        "rk_miss": "{pct:.1f} % missing cells: plan an imputation.", "rk_dup": "{n} duplicated rows to review.",
        "rk_const": "Constant columns to drop: {c}.", "rk_id": "Identifier-like columns (exclude from the model): {c}.",
        "rk_imb": "Imbalanced target (majority class {mx:.0f} %, minority {mn:.0f} %): prefer F1 / AUC over accuracy.",
        "rk_out": "Significant outliers in \"{c}\" ({p:.1f} %).", "rk_nopp": "Preprocessing not done: models should not be compared before.",
        "rk_over": "Large train/test gap for {m} ({a:.3f} vs {b:.3f}): overfitting risk.",
    },
}


def _risks(s, t, info, L):
    out = []
    cells = max(1, (s.get("n_rows") or 0) * (s.get("n_columns") or 1))
    pct = 100 * (s.get("n_missing_cells") or 0) / cells
    if pct > 0.5:
        out.append(L["rk_miss"].format(pct=pct))
    if (s.get("n_duplicates") or 0) > 0:
        out.append(L["rk_dup"].format(n=s["n_duplicates"]))
    if s.get("constant_columns"):
        out.append(L["rk_const"].format(c=", ".join(map(str, s["constant_columns"][:6]))))
    if s.get("id_like_columns"):
        out.append(L["rk_id"].format(c=", ".join(map(str, s["id_like_columns"][:6]))))
    cb = s.get("class_balance")
    if isinstance(cb, list) and len(cb) > 1:
        try:
            pcts = [float(r.get("percent")) for r in cb if r.get("percent") is not None]
            if pcts and min(pcts) > 0 and max(pcts) / min(pcts) > 3:
                out.append(L["rk_imb"].format(mx=max(pcts), mn=min(pcts)))
        except (TypeError, ValueError):
            pass
    o = t.get("outliers_summary")
    try:
        if o is not None and not o.empty and "outlier_percent" in o.columns:
            row = o.sort_values("outlier_percent", ascending=False).iloc[0]
            if float(row["outlier_percent"]) > 5:
                out.append(L["rk_out"].format(c=row["column"], p=float(row["outlier_percent"])))
    except Exception:
        pass
    if "pretraitement" not in info:
        out.append(L["rk_nopp"])
    for run in info.get("modeles", []):
        for m in run.get("modeles", [])[:1]:
            tr, te = m.get("val"), m.get("test")
            if isinstance(tr, (int, float)) and isinstance(te, (int, float)) and tr - te > 0.15:
                out.append(L["rk_over"].format(m=m["nom"], a=tr, b=te))
    return out


def build_specification_pdf(dataset, narrative="", language="fr"):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    import agent as ag                                   # import tardif : agent importe ce module à la demande

    L = L10N.get(language, L10N["fr"])
    data = analysis_engine.load_dataset_results(dataset)
    if not data:
        raise ag.AIError("Dataset introuvable", 404)
    s, t = data["summary"], data["tables"]
    info = ag._pipeline_info(dataset)
    out_dir = Path(analysis_engine.RESULTS_DIR) / analysis_engine.clean_dataset_name(dataset) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cahier_des_charges_{analysis_engine.clean_dataset_name(dataset)}_{datetime.now():%Y%m%d_%H%M%S}.pdf"

    ACC = colors.HexColor("#1d4ed8")
    base = getSampleStyleSheet()
    st = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=21, textColor=ACC, alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5, textColor=colors.HexColor("#64748b"), spaceAfter=10),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#0f172a"),
                            spaceBefore=14, spaceAfter=6, keepWithNext=1),
        "p": ParagraphStyle("p", parent=base["Normal"], fontSize=10, leading=14.5, spaceAfter=4),
        "b": ParagraphStyle("b", parent=base["Normal"], fontSize=10, leading=14.5, leftIndent=14, bulletIndent=3, spaceAfter=2),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.5, leading=11),
    }
    W = A4[0] - 4 * cm

    def table(rows, widths, header=True):
        cells = [[Paragraph(ag._txt(c), st["cell"]) for c in r] for r in rows]
        if header:
            cells[0] = [Paragraph(f"<font color='white'><b>{ag._txt(c)}</b></font>", st["cell"]) for c in rows[0]]
        style = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                 ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), ACC))
        tb = Table(cells, colWidths=widths, repeatRows=1 if header else 0)
        tb.setStyle(TableStyle(style))
        return tb

    def bullets(items):
        return [Paragraph(ag._txt(i), st["b"], bulletText="•") for i in items]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(2 * cm, 1.1 * cm, ag._txt(f"{dataset} - {L['title']}"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{L['page']} {doc.page}")
        canvas.restoreState()

    story = [Paragraph(ag._txt(L["title"]), st["title"]),
             Paragraph(ag._txt(f"{dataset}  |  {L['generated']} {datetime.now():%d/%m/%Y %H:%M}  |  {L['file']} : {s.get('file') or '-'}"), st["sub"])]

    # 1. contexte et objectif
    story.append(Paragraph(L["ctx"], st["h"]))
    if str(narrative or "").strip():
        story += ag._md_to_flowables(narrative, st["p"], st["b"])
    else:
        story.append(Paragraph(ag._txt(L["ctx_auto"].format(ds=dataset, rows=ag._fmt(s.get("n_rows")), cols=ag._fmt(s.get("n_columns")))), st["p"]))

    # 2. données
    cells = max(1, (s.get("n_rows") or 0) * (s.get("n_columns") or 1))
    miss_pct = 100 * (s.get("n_missing_cells") or 0) / cells
    story += [Paragraph(L["data"], st["h"]), table([
        [L["rows"], ag._fmt(s.get("n_rows")), L["cols"], ag._fmt(s.get("n_columns"))],
        [L["type"], s.get("dataset_type") or "-", L["target"], s.get("target_class") or L["none"]],
        [L["num"], ag._fmt(len(s.get("numeric_columns") or [])), L["cat"], ag._fmt(len(s.get("categorical_columns") or []))],
        [L["missing"], f"{ag._fmt(s.get('n_missing_cells'))} ({miss_pct:.2f} %)", L["dups"], ag._fmt(s.get("n_duplicates"))],
        [L["date"], s.get("date_column") or L["none"], "", ""]],
        [3.8 * cm, 4.6 * cm, 3.8 * cm, W - 12.2 * cm], header=False)]

    # 3. exigences fonctionnelles (statut calculé)
    pp, sp_, md = info.get("pretraitement"), info.get("split"), info.get("modeles")
    yes = lambda ok: L["done"] if ok else L["todo"]
    story += [Paragraph(L["req"], st["h"]), table([[L["id"], L["need"], L["status"]],
        ["F1", L["r_eda"], L["done"]], ["F2", L["r_pp"], yes(pp)], ["F3", L["r_split"], yes(sp_)],
        ["F4", L["r_model"], yes(md)], ["F5", L["r_cmp"], L["demand"]], ["F6", L["r_pdf"], L["done"]]],
        [1.2 * cm, W - 4.2 * cm, 3 * cm])]

    # 4. résumé du pipeline
    story.append(Paragraph(L["pipe"], st["h"]))
    if pp:
        story.append(Paragraph(ag._txt(f"{L['rows']} : {ag._fmt(pp.get('lignes_avant'))} -> {ag._fmt(pp.get('lignes_apres'))}"), st["p"]))
        rows = [[L["step"], L["detail"]]]
        for e in pp.get("etapes", []):
            lab, _, det = e.partition(": ")
            rows.append([lab, det or "-"])
        if len(rows) > 1:
            story.append(table(rows, [4 * cm, W - 4 * cm]))
    else:
        story.append(Paragraph(ag._txt(L["no_pp"]), st["p"]))
    if sp_:
        pt = sp_.get("part_test")
        story += [Spacer(1, 6), table([[L["target"], str(sp_.get("cible") or "-"), L["test_size"], f"{float(pt) * 100:.0f} %" if pt is not None else "-"],
                                       [L["strategy"], str(sp_.get("strategie") or L["auto"]), "", ""]], [3 * cm, 5.5 * cm, 3.4 * cm, W - 11.9 * cm], header=False)]
    else:
        story.append(Paragraph(ag._txt(L["no_split"]), st["p"]))
    if md:
        for run in md:
            rows = [[L["model"], L["cv"], L["test"]]] + [[m["nom"], ag._fmt(m.get("val")), ag._fmt(m.get("test"))] for m in run["modeles"]]
            head = Paragraph(ag._txt(f"{run['tache']} - {run['famille']}  ({L['best']} : {run.get('meilleur') or '-'}, {run.get('metrique')})"), st["p"])
            story.append(KeepTogether([Spacer(1, 4), head, table(rows, [W - 6 * cm, 3 * cm, 3 * cm])]))
    else:
        story.append(Paragraph(ag._txt(L["no_models"]), st["p"]))

    # 5 à 8
    story += [Paragraph(L["nfr"], st["h"])] + bullets([L["nf1"], L["nf2"], L["nf3"], L["nf4"]])
    story += [Paragraph(L["deliv"], st["h"])] + bullets([L["d1"], L["d2"], L["d3"]])
    story += [Paragraph(L["risk"], st["h"])] + (bullets(_risks(s, t, info, L)) or [Paragraph(ag._txt(L["no_risk"]), st["p"])])
    story += [Paragraph(L["valid"], st["h"])] + bullets([L["v1"], L["v2"], L["v3"], L["v4"]])

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title=f"{L['title']} - {dataset}", author="EDA Studio").build(story, onFirstPage=footer, onLaterPages=footer)
    return path


# ======================================================================
# Version 100 % schémas (pas de paragraphes) : flux du pipeline, architecture, graphiques, statuts
# ======================================================================
SCH = {
    "fr": {"title": "Cahier des charges - schémas du pipeline", "gen": "Généré le", "rows": "Lignes", "cols": "Colonnes", "miss": "Manquants",
           "target": "Cible", "none": "aucune", "pipe": "Pipeline", "arch": "Architecture", "charts": "Données et résultats", "req": "Exigences et risques",
           "eda": "EDA", "pp": "Prétraitement", "split": "Split train / test", "models": "Modèles", "pdf": "Rapports PDF",
           "done": "Réalisé", "todo": "À faire", "demand": "Sur demande", "test": "test", "strategy": "stratégie", "best": "meilleur",
           "pdf_gen": "Général", "pdf_cmp": "Comparaison", "pdf_spec": "Cahier des charges", "user": "Utilisateur", "web": "Interface web (Flask)",
           "chat": "Chat agent", "agent": "Agent IA", "tools": "Outils", "mods": "Modules d'analyse", "store": "Résultats (SQLite, graphiques)",
           "types": "Types de variables", "num": "Numériques", "cat": "Catégorielles", "missing": "Valeurs manquantes (%)", "balance": "Équilibre de la cible (%)",
           "outl": "Valeurs aberrantes (%)", "scores": "Scores des modèles", "nomiss": "Aucune valeur manquante", "cv": "Val", "risks": "Risques détectés",
           "norisk": "Aucun risque majeur", "r_miss": "Manquants", "r_dup": "Doublons", "r_imb": "Cible déséquilibrée", "r_out": "Outliers", "r_nopp": "Sans prétraitement",
           "r_over": "Surapprentissage", "r_const": "Colonnes constantes", "r_id": "Colonnes ID",
           "F1": "Analyse EDA", "F2": "Prétraitement", "F3": "Split", "F4": "Sélection de modèles", "F5": "Avec / sans bruit", "F6": "Export PDF", "page": "Page"},
    "en": {"title": "Specification - pipeline diagrams", "gen": "Generated on", "rows": "Rows", "cols": "Columns", "miss": "Missing",
           "target": "Target", "none": "none", "pipe": "Pipeline", "arch": "Architecture", "charts": "Data and results", "req": "Requirements and risks",
           "eda": "EDA", "pp": "Preprocessing", "split": "Train / test split", "models": "Models", "pdf": "PDF reports",
           "done": "Done", "todo": "To do", "demand": "On demand", "test": "test", "strategy": "strategy", "best": "best",
           "pdf_gen": "General", "pdf_cmp": "Comparison", "pdf_spec": "Specification", "user": "User", "web": "Web interface (Flask)",
           "chat": "Agent chat", "agent": "AI agent", "tools": "Tools", "mods": "Analysis modules", "store": "Results (SQLite, plots)",
           "types": "Variable types", "num": "Numeric", "cat": "Categorical", "missing": "Missing values (%)", "balance": "Target balance (%)",
           "outl": "Outliers (%)", "scores": "Model scores", "nomiss": "No missing values", "cv": "Val", "risks": "Detected risks",
           "norisk": "No major risk", "r_miss": "Missing", "r_dup": "Duplicates", "r_imb": "Imbalanced target", "r_out": "Outliers", "r_nopp": "No preprocessing",
           "r_over": "Overfitting", "r_const": "Constant columns", "r_id": "ID columns",
           "F1": "EDA analysis", "F2": "Preprocessing", "F3": "Split", "F4": "Model selection", "F5": "With / without noise", "F6": "PDF export", "page": "Page"},
}


def _short_risks(s, t, info, K):
    out = []
    cells = max(1, (s.get("n_rows") or 0) * (s.get("n_columns") or 1))
    pct = 100 * (s.get("n_missing_cells") or 0) / cells
    if pct > 0.5:
        out.append((f"{K['r_miss']} {pct:.1f} %", "o"))
    if (s.get("n_duplicates") or 0) > 0:
        out.append((f"{K['r_dup']} {s['n_duplicates']}", "o"))
    if s.get("constant_columns"):
        out.append((f"{K['r_const']} {len(s['constant_columns'])}", "o"))
    if s.get("id_like_columns"):
        out.append((f"{K['r_id']} {len(s['id_like_columns'])}", "o"))
    cb = s.get("class_balance")
    try:
        pcts = [float(r.get("percent")) for r in cb if r.get("percent") is not None] if isinstance(cb, list) else []
        if len(pcts) > 1 and min(pcts) > 0 and max(pcts) / min(pcts) > 3:
            out.append((f"{K['r_imb']} {max(pcts):.0f}/{min(pcts):.0f} %", "r"))
    except (TypeError, ValueError):
        pass
    try:
        o = t.get("outliers_summary")
        if o is not None and not o.empty and "outlier_percent" in o.columns:
            row = o.sort_values("outlier_percent", ascending=False).iloc[0]
            if float(row["outlier_percent"]) > 5:
                out.append((f"{K['r_out']} {row['column']} {float(row['outlier_percent']):.0f} %", "r"))
    except Exception:
        pass
    if "pretraitement" not in info:
        out.append((K["r_nopp"], "r"))
    for run in info.get("modeles", []):
        for m in run.get("modeles", [])[:1]:
            a, b = m.get("val"), m.get("test")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a - b > 0.15:
                out.append((f"{K['r_over']} {m['nom']}", "r"))
    return out


def build_schema_pdf(dataset, narrative="", language="fr"):
    """Cahier des charges + résumé du pipeline en schémas uniquement (aucun paragraphe). `narrative` est ignoré."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as cv

    import agent as ag

    K = SCH.get(language, SCH["fr"])
    data = analysis_engine.load_dataset_results(dataset)
    if not data:
        raise ag.AIError("Dataset introuvable", 404)
    s, t = data["summary"], data["tables"]
    info = ag._pipeline_info(dataset)
    out_dir = Path(analysis_engine.RESULTS_DIR) / analysis_engine.clean_dataset_name(dataset) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cahier_des_charges_schemas_{analysis_engine.clean_dataset_name(dataset)}_{datetime.now():%Y%m%d_%H%M%S}.pdf"

    W, H = A4
    M = 40
    BLUE, GREEN, GREY = colors.HexColor("#1d4ed8"), colors.HexColor("#16a34a"), colors.HexColor("#94a3b8")
    ORANGE, RED, INK = colors.HexColor("#f59e0b"), colors.HexColor("#dc2626"), colors.HexColor("#0f172a")
    PALE, LINE = colors.HexColor("#f1f5f9"), colors.HexColor("#cbd5e1")
    c = cv.Canvas(str(path), pagesize=A4)
    c.setTitle(f"{K['title']} - {dataset}")
    c.setAuthor("EDA Studio")
    state = {"page": 1, "y": H - M}

    def S(v):
        s_ = "" if v is None else str(v)
        for a, b in (("→", "->"), ("≤", "<="), ("≥", ">="), ("×", "x"), ("−", "-"), ("\u202f", " "), ("\u00a0", " ")):
            s_ = s_.replace(a, b)
        return s_.encode("cp1252", "replace").decode("cp1252")

    def fit(txt, size, maxw, bold=False):
        txt = S(txt)
        font = "Helvetica-Bold" if bold else "Helvetica"
        while len(txt) > 1 and c.stringWidth(txt, font, size) > maxw:
            txt = txt[:-2] + "…" if not txt.endswith("…") else txt[:-2] + "…"
            txt = txt.replace("……", "…")
        return txt.replace("…", "...") if c.stringWidth(txt, font, size) > maxw else txt

    def text(x, y, txt, size=9, bold=False, col=INK, align="l", maxw=None):
        c.setFillColor(col)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        t_ = fit(txt, size, maxw, bold) if maxw else S(txt)
        {"l": c.drawString, "c": c.drawCentredString, "r": c.drawRightString}[align](x, y, t_)

    def box(x, y, w, h, fill=PALE, stroke=LINE, r=8, lw=0.8):
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(lw)
        c.roundRect(x, y, w, h, r, stroke=1, fill=1)

    def arrow(x1, y1, x2, y2, col=GREY):
        import math
        c.setStrokeColor(col)
        c.setFillColor(col)
        c.setLineWidth(1.6)
        c.line(x1, y1, x2, y2)
        a = math.atan2(y2 - y1, x2 - x1)
        p = c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(x2 - 8 * math.cos(a - .4), y2 - 8 * math.sin(a - .4))
        p.lineTo(x2 - 8 * math.cos(a + .4), y2 - 8 * math.sin(a + .4))
        p.close()
        c.drawPath(p, stroke=0, fill=1)

    def pill(x, y, w, label, col):
        box(x, y, w, 16, fill=col, stroke=col, r=8)
        text(x + w / 2, y + 4.5, label, 8, True, colors.white, "c", w - 6)

    def header(title):
        c.setFillColor(BLUE)
        c.rect(0, H - 46, W, 46, stroke=0, fill=1)
        text(M, H - 30, title, 15, True, colors.white)
        text(W - M, H - 30, f"{dataset}  |  {datetime.now():%d/%m/%Y}", 9, False, colors.white, "r")
        state["y"] = H - 46 - 22

    def footer():
        text(W - M, 22, f"{K['page']} {state['page']}", 8, False, GREY, "r")
        text(M, 22, f"{dataset} - {K['title']}", 8, False, GREY)

    def new_page(title):
        footer()
        c.showPage()
        state["page"] += 1
        header(title)

    def need(h_, title):
        if state["y"] - h_ < 45:
            new_page(title)

    def panel_title(label):
        state["y"] -= 14
        text(M, state["y"], label, 11, True, BLUE)
        c.setStrokeColor(BLUE)
        c.setLineWidth(1.2)
        c.line(M, state["y"] - 4, W - M, state["y"] - 4)
        state["y"] -= 20

    def hbars(items, maxv, col_fn, unit=""):
        lab_w, bar_w = 130, W - 2 * M - 130 - 55
        for lab, v in items:
            y = state["y"] - 13
            text(M, y + 2, lab, 8.5, False, INK, "l", lab_w - 8)
            c.setFillColor(PALE)
            c.rect(M + lab_w, y, bar_w, 12, stroke=0, fill=1)
            c.setFillColor(col_fn(v))
            c.rect(M + lab_w, y, max(1.5, bar_w * min(1.0, v / maxv if maxv else 0)), 12, stroke=0, fill=1)
            text(M + lab_w + bar_w + 6, y + 2, f"{v:.1f}{unit}", 8.5, True)
            state["y"] -= 18
        state["y"] -= 6

    cells = max(1, (s.get("n_rows") or 0) * (s.get("n_columns") or 1))
    miss_pct = 100 * (s.get("n_missing_cells") or 0) / cells
    pp, sp_, md = info.get("pretraitement"), info.get("split"), info.get("modeles")

    # ------------------------------------------------------------ page 1 : flux du pipeline
    header(K["title"])
    gap = 10
    tw = (W - 2 * M - 3 * gap) / 4
    for i, (big, lab) in enumerate([(ag._fmt(s.get("n_rows")), K["rows"]), (ag._fmt(s.get("n_columns")), K["cols"]),
                                    (f"{miss_pct:.1f} %", K["miss"]), (s.get("target_class") or K["none"], K["target"])]):
        x = M + i * (tw + gap)
        box(x, state["y"] - 50, tw, 50, fill=colors.white)
        text(x + tw / 2, state["y"] - 27, big, 16, True, BLUE, "c", tw - 10)
        text(x + tw / 2, state["y"] - 42, lab, 8, False, GREY, "c")
    state["y"] -= 50 + 28
    panel_title(K["pipe"])

    mlines = []
    for run in (md or [])[:2]:
        mlines.append(f"{run['famille']}: {K['best']} {run.get('meilleur') or '-'} ({run.get('metrique')})")
    stages = [
        (K["eda"], True, [f"{ag._fmt(s.get('n_rows'))} x {ag._fmt(s.get('n_columns'))}", f"{K['miss']} {miss_pct:.1f} %", f"{K['target']}: {s.get('target_class') or K['none']}"]),
        (K["pp"], bool(pp), ([f"{ag._fmt(pp.get('lignes_avant'))} -> {ag._fmt(pp.get('lignes_apres'))} {K['rows'].lower()}"] +
                             [e.split(":")[0] for e in pp.get("etapes", [])[:4]]) if pp else []),
        (K["split"], bool(sp_), ([f"{K['target']}: {sp_.get('cible') or '-'}", f"{K['test']} {float(sp_.get('part_test') or 0) * 100:.0f} %",
                                  f"{K['strategy']}: {sp_.get('strategie') or 'auto'}"]) if sp_ else []),
        (K["models"], bool(md), mlines),
        (K["pdf"], True, [K["pdf_gen"], K["pdf_cmp"], K["pdf_spec"]]),
    ]
    bh, bg = 84, 26
    for i, (name, ok, lines) in enumerate(stages):
        top = state["y"]
        col = GREEN if ok else GREY
        box(M, top - bh, W - 2 * M, bh, fill=colors.white, stroke=col, lw=1.6)
        c.setFillColor(col)
        c.circle(M + 28, top - bh / 2, 15, stroke=0, fill=1)
        text(M + 28, top - bh / 2 - 5, str(i + 1), 14, True, colors.white, "c")
        text(M + 56, top - bh / 2 - 5, name, 13, True, INK, "l", 150)
        pill(W - M - 100, top - 24, 86, K["done"] if ok else K["todo"], col)
        for j, ln in enumerate(lines[:4]):
            text(M + 220, top - 20 - j * 14, ln, 9, False, INK, "l", W - 2 * M - 340)
        if i < len(stages) - 1:
            arrow(W / 2, top - bh, W / 2, top - bh - bg + 3, BLUE)
        state["y"] = top - bh - bg
    footer()

    # ------------------------------------------------------------ page 2 : architecture
    c.showPage()
    state["page"] += 1
    header(K["arch"])
    top = state["y"]

    def node(cx, ytop, w, h, label, sub="", fill=colors.white, stroke=BLUE):
        box(cx - w / 2, ytop - h, w, h, fill=fill, stroke=stroke, lw=1.3)
        text(cx, ytop - h / 2 + (3 if sub else -3), label, 10, True, INK, "c", w - 10)
        if sub:
            text(cx, ytop - h / 2 - 9, sub, 8, False, GREY, "c", w - 10)

    cx = W / 2
    node(cx, top, 150, 38, K["user"], fill=colors.HexColor("#dbeafe"))
    arrow(cx, top - 38, cx, top - 62, BLUE)
    y2 = top - 62
    node(cx - 125, y2, 200, 44, K["web"], "app.py + templates")
    node(cx + 125, y2, 200, 44, K["chat"], "agent-chat.js")
    arrow(cx + 125, y2 - 44, cx + 125, y2 - 70, BLUE)
    y3 = y2 - 70
    node(cx + 125, y3, 200, 44, K["agent"], "agent.py + Groq")
    arrow(cx + 125, y3 - 44, cx + 125, y3 - 66, BLUE)
    y4 = y3 - 66
    tools = ["dataset_brief", "pdf_report", "model_comparison_pdf", "specification_pdf"]
    text(M, y4 - 6, K["tools"], 9, True, GREY)
    for i, tl in enumerate(tools):
        x = M + 60 + i * ((W - 2 * M - 60) / 4)
        box(x, y4 - 40, (W - 2 * M - 60) / 4 - 8, 30, fill=PALE)
        text(x + ((W - 2 * M - 60) / 4 - 8) / 2, y4 - 28, tl, 7.5, True, INK, "c", (W - 2 * M - 60) / 4 - 14)
    arrow(cx, y4 - 40, cx, y4 - 70, BLUE)
    y5 = y4 - 70
    text(M, y5 - 6, K["mods"], 9, True, GREY)
    mods = ["analysis_engine", "preprocessing", "split", "model_selection", "scenario_report", "spec_report"]
    mw = (W - 2 * M - 60) / 3 - 8
    for i, m in enumerate(mods):
        x = M + 60 + (i % 3) * (mw + 8)
        y = y5 - 10 - (i // 3) * 40
        box(x, y - 30, mw, 30, fill=colors.white, stroke=BLUE)
        text(x + mw / 2, y - 19, m + ".py", 9, True, INK, "c", mw - 8)
    arrow(cx, y5 - 80, cx, y5 - 108, BLUE)
    y6 = y5 - 108
    node(cx, y6, 260, 40, K["store"], "results/", fill=colors.HexColor("#fef3c7"), stroke=ORANGE)
    arrow(cx, y6 - 40, cx, y6 - 66, BLUE)
    y7 = y6 - 66
    for i, lab in enumerate([K["pdf_gen"], K["pdf_cmp"], K["pdf_spec"]]):
        x = M + i * ((W - 2 * M) / 3)
        w3 = (W - 2 * M) / 3 - 10
        box(x, y7 - 46, w3, 46, fill=colors.HexColor("#fee2e2"), stroke=RED, lw=1.2)
        text(x + w3 / 2, y7 - 24, "PDF", 12, True, RED, "c")
        text(x + w3 / 2, y7 - 38, lab, 8, False, INK, "c", w3 - 8)
    state["y"] = y7 - 60
    footer()

    # ------------------------------------------------------------ page 3 : données et résultats
    c.showPage()
    state["page"] += 1
    header(K["charts"])
    ttl = K["charts"]

    # types de variables
    nn, nc = len(s.get("numeric_columns") or []), len(s.get("categorical_columns") or [])
    panel_title(K["types"])
    bw = W - 2 * M
    tot = max(1, nn + nc)
    c.setFillColor(BLUE)
    c.rect(M, state["y"] - 22, bw * nn / tot, 22, stroke=0, fill=1)
    c.setFillColor(ORANGE)
    c.rect(M + bw * nn / tot, state["y"] - 22, bw * nc / tot, 22, stroke=0, fill=1)
    text(M + 6, state["y"] - 15, f"{K['num']} {nn}", 9, True, colors.white)
    text(W - M - 6, state["y"] - 15, f"{K['cat']} {nc}", 9, True, colors.white, "r")
    state["y"] -= 44

    # manquants
    need(120, ttl)
    panel_title(K["missing"])
    ms = [(m["column"], float(m["percent"])) for m in ag._missing_table(t)[:8]]
    if ms:
        hbars(ms, max(v for _, v in ms) or 1, lambda v: RED if v > 20 else ORANGE)
    else:
        box(M, state["y"] - 26, 220, 26, fill=colors.HexColor("#dcfce7"), stroke=GREEN)
        text(M + 110, state["y"] - 17, K["nomiss"], 9, True, GREEN, "c")
        state["y"] -= 42

    # outliers
    o = t.get("outliers_summary")
    try:
        if o is not None and not o.empty and "outlier_percent" in o.columns:
            od = o.sort_values("outlier_percent", ascending=False).head(6)
            items = [(str(r["column"]), float(r["outlier_percent"])) for _, r in od.iterrows()]
            need(30 + 18 * len(items), ttl)
            panel_title(K["outl"])
            hbars(items, max(v for _, v in items) or 1, lambda v: RED if v > 5 else ORANGE)
    except Exception:
        pass

    # équilibre de la cible
    tg = s.get("target_class")
    cb = ag._class_balance(s)
    if tg and cb:
        items = []
        for r in cb[:8]:
            try:
                items.append((str(r.get(tg, list(r.values())[0])), float(r.get("percent", 0))))
            except (TypeError, ValueError):
                pass
        if items:
            need(30 + 18 * len(items), ttl)
            panel_title(f"{K['balance']} - {tg}")
            hbars(items, 100, lambda v: BLUE, " %")

    # scores des modèles : barres verticales validation vs test
    for run in (md or []):
        ms_ = run["modeles"][:5]
        vals = [v for m in ms_ for v in (m.get("val"), m.get("test")) if isinstance(v, (int, float))]
        if not vals:
            continue
        need(190, ttl)
        panel_title(f"{K['scores']} - {run['famille']} ({run.get('metrique')})")
        lo, hi = min(0.0, min(vals)), max(1.0, max(vals))
        ch, cw = 110, W - 2 * M
        base_y = state["y"] - 14 - ch
        zero = base_y + ch * (0 - lo) / (hi - lo)
        c.setStrokeColor(LINE)
        c.line(M, zero, W - M, zero)
        gw = cw / len(ms_)
        for i, m in enumerate(ms_):
            for k_, (key, col) in enumerate((("val", BLUE), ("test", GREEN))):
                v = m.get(key)
                if not isinstance(v, (int, float)):
                    continue
                bx = M + i * gw + gw / 2 - 22 + k_ * 24
                hh = ch * abs(v) / (hi - lo)
                c.setFillColor(col)
                c.rect(bx, zero if v >= 0 else zero - hh, 20, max(hh, 1), stroke=0, fill=1)
                text(bx + 10, (zero + hh + 3) if v >= 0 else (zero - hh - 10), f"{v:.2f}", 7.5, True, INK, "c")
            text(M + i * gw + gw / 2, base_y - 12, m["nom"], 8, False, INK, "c", gw - 6)
        c.setFillColor(BLUE)
        c.rect(W - M - 110, state["y"] - 8, 8, 8, stroke=0, fill=1)
        text(W - M - 98, state["y"] - 8, K["cv"], 8)
        c.setFillColor(GREEN)
        c.rect(W - M - 60, state["y"] - 8, 8, 8, stroke=0, fill=1)
        text(W - M - 48, state["y"] - 8, K["test"], 8)
        state["y"] = base_y - 34

    # ------------------------------------------------------------ page 4 : exigences et risques
    new_page(K["req"])
    panel_title(K["req"])
    reqs = [("F1", K["F1"], True), ("F2", K["F2"], bool(pp)), ("F3", K["F3"], bool(sp_)), ("F4", K["F4"], bool(md)), ("F5", K["F5"], None), ("F6", K["F6"], True)]
    tw2 = (W - 2 * M - gap) / 2
    for i, (fid, lab, ok) in enumerate(reqs):
        x = M + (i % 2) * (tw2 + gap)
        y = state["y"] - (i // 2) * 62
        col = GREY if ok is False else (BLUE if ok is None else GREEN)
        box(x, y - 52, tw2, 52, fill=colors.white, stroke=col, lw=1.6)
        c.setFillColor(col)
        c.roundRect(x + 10, y - 40, 36, 28, 6, stroke=0, fill=1)
        text(x + 28, y - 31, fid, 11, True, colors.white, "c")
        text(x + 56, y - 24, lab, 10, True, INK, "l", tw2 - 70)
        text(x + 56, y - 40, K["demand"] if ok is None else (K["done"] if ok else K["todo"]), 8.5, True, col)
    state["y"] -= 3 * 62 + 24

    panel_title(K["risks"])
    rk = _short_risks(s, t, info, K)
    if not rk:
        box(M, state["y"] - 28, 200, 28, fill=colors.HexColor("#dcfce7"), stroke=GREEN)
        text(M + 100, state["y"] - 18, K["norisk"], 10, True, GREEN, "c")
    else:
        x, y = M, state["y"]
        for lab, lvl in rk:
            w_ = c.stringWidth(S(lab), "Helvetica-Bold", 9) + 22
            if x + w_ > W - M:
                x, y = M, y - 32
            col = RED if lvl == "r" else ORANGE
            box(x, y - 24, w_, 24, fill=colors.white, stroke=col, r=12, lw=1.5)
            c.setFillColor(col)
            c.circle(x + 11, y - 12, 4, stroke=0, fill=1)
            text(x + 20, y - 16, lab, 9, True, INK)
            x += w_ + 8
    footer()
    c.save()
    return path
