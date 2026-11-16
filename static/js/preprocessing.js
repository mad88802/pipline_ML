/* Étape 02 — Prétraitement : lit la config, lance /api/preprocessing/<dataset>/run, affiche le résultat. */
(function () {
  "use strict";
  var D = window.__PP__ || {}, P = D.profile;
  if (!P) return;

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var fmt = function (n) { return Number(n).toLocaleString("fr-FR"); };
  var esc = function (s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); };
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- sidebar : one panel at a time (same pattern as the EDA pages) ---------- */
  var titles = { overview: "Aperçu", dups: "Doublons", useless: "Colonnes inutiles", missing: "Valeurs manquantes",
                 outliers: "Outliers", noise: "Bruit", encoding: "Encodage", scaling: "Feature Scaling", balance: "Déséquilibre des classes", result: "Résultat" };
  var current = "overview";
  var STEP_OF = { dups: "duplicates", useless: "useless", missing: "missing", outliers: "outliers", noise: "noise", encoding: "encoding", scaling: "scaling", balance: "smote" };
  function showPanel(key) {
    if (!titles[key]) key = "overview";
    $$(".pp-panel-view").forEach(function (p) {
      var on = p.dataset.panel === key;
      p.hidden = !on;
    });
    $$("#ppNav .eda-nav-item").forEach(function (a) { a.classList.toggle("active", a.dataset.panel === key); });
    $("#ppTitle").textContent = titles[key];
    try { history.replaceState(null, "", "#pp-" + key); } catch (e) {}
    current = key;
    if (STEP_OF[key]) loadStep(key);
  }
  $$("#ppNav .eda-nav-item").forEach(function (a) {
    a.addEventListener("click", function (ev) { ev.preventDefault(); showPanel(a.dataset.panel); });
  });

  /* ---------- KPI ---------- */
  function countUp(el, to) {
    var from = Number((el.dataset.v || "0")), t0 = null;
    el.dataset.v = to;
    if (reduce || from === to) { el.textContent = fmt(to); return; }
    (function step(t) {
      if (t0 === null) t0 = t;
      var k = Math.min(1, (t - t0) / 700), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(Math.round(from + (to - from) * e));
      if (k < 1) requestAnimationFrame(step);
    })(performance.now());
  }
  function setKpi(key, value, before) {
    var box = $('.pp-kpi[data-k="' + key + '"]');
    countUp($(".pp-kv", box), value);
    var d = $(".pp-kd", box);
    if (before === undefined || before === value) { d.textContent = ""; d.className = "pp-kd"; return; }
    d.textContent = "avant " + fmt(before);
    d.className = "pp-kd " + (value < before ? "down" : "up");
  }
  setKpi("rows", P.n_rows); setKpi("cols", P.n_cols); setKpi("missing", P.n_missing); setKpi("dups", P.n_duplicates);

  /* ---------- controls ---------- */
  function segGet(name) { var b = $('.pp-seg[data-name="' + name + '"] button.on'); return b ? b.dataset.v : null; }
  function segSet(name, v) { $$('.pp-seg[data-name="' + name + '"] button').forEach(function (b) { b.classList.toggle("on", b.dataset.v === v); }); }
  $$(".pp-seg button").forEach(function (b) {
    b.addEventListener("click", function () { segSet(b.parentNode.dataset.name, b.dataset.v); refresh(); });
  });

  function readCfg() {
    var imp = $("#optImpute").checked;
    return {
      test_size: Math.min(50, Math.max(0, +$("#optTest").value || 0)) / 100,   /* split BEFORE the learned steps */
      drop_duplicates: $("#optDup").checked,
      drop_useless: $("#optUseless").checked,
      missing_col_threshold: +$("#optThr").value,
      num_missing: imp ? "knn" : "none",                        /* KNN imputation */
      cat_missing: imp ? "knn" : "none",
      outliers: $("#optCap").checked ? "ocsvm" : "none",        /* One-Class SVM : flagged train rows removed */
      ocsvm_nu: Math.min(50, Math.max(0.5, +$("#optNu").value || 5)) / 100,
      smooth_window: +$("#optWin").value || 0,                  /* moving average */
      encoding: $("#optTE").checked ? "target" : "none",        /* target encoding */
      te_smoothing: +$("#optTESm").value || 0,
      encode_target: $("#optEncT").checked,
      scaling: $("#optScale").checked ? $("#optScaleMethod").value : "none",   /* feature scaling */
      balance: $("#optBal").checked,                            /* SMOTE */
      smote_k: +$("#optSmK").value || 5,
      target: $("#optTarget").value || null,
      drop_columns: $$("#ppCols input.keep").filter(function (c) { return !c.checked; }).map(function (c) { return c.dataset.col; })
    };
  }
  function applyCfg(c) {
    $("#optTest").value = Math.round((c.test_size == null ? 0.2 : c.test_size) * 100);
    $("#optDup").checked = !!c.drop_duplicates;
    $("#optUseless").checked = !!c.drop_useless;
    $("#optThr").value = c.missing_col_threshold;
    $("#optImpute").checked = !!c.num_missing && c.num_missing !== "none";
    $("#optCap").checked = c.outliers === "ocsvm";
    $("#optNu").value = Math.round((c.ocsvm_nu == null ? 0.05 : c.ocsvm_nu) * 1000) / 10;
    $("#optWin").value = c.smooth_window;
    $("#optTE").checked = c.encoding === "target";
    $("#optTESm").value = c.te_smoothing;
    $("#optEncT").checked = !!c.encode_target;
    var hasScale = !!(c.scaling && c.scaling !== "none");
    $("#optScale").checked = hasScale;
    if (hasScale) $("#optScaleMethod").value = c.scaling;
    $("#optBal").checked = !!c.balance;
    $("#optSmK").value = c.smote_k;
    $("#optTarget").value = c.target || "";
    $$("#ppCols input.keep").forEach(function (c2) { c2.checked = true; });
    refresh();
  }

  /* ---------- live chips ---------- */
  function chip(id, txt, warn) {
    var e = $(id); e.textContent = txt; e.classList.toggle("warn", !!warn); e.hidden = !txt;
  }
  function barsHtml(counts, cls) {
    var keys = Object.keys(counts), max = Math.max.apply(null, keys.map(function (k) { return counts[k]; }).concat([1]));
    return keys.map(function (k) {
      return '<div class="pp-cbar ' + (cls || "") + '"><span class="pp-cl" title="' + esc(k) + '">' + esc(k) + '</span>' +
        '<span class="pp-ct"><i style="width:' + (100 * counts[k] / max) + '%"></i></span><b>' + fmt(counts[k]) + "</b></div>";
    }).join("");
  }
  function refresh() {
    var thr = +$("#optThr").value, tgt = $("#optTarget").value;
    $("#outThr").textContent = thr + " %";
    var hit = P.columns.filter(function (c) {
      return c.name !== tgt && c.kind !== "date" && (c.kind === "id" || c.kind === "constante" || c.pct_missing > thr);
    }).length;
    chip("#chipDup", P.n_duplicates ? fmt(P.n_duplicates) + " détectés" : "aucun", P.n_duplicates > 0);
    chip("#chipUse", hit ? hit + " colonne" + (hit > 1 ? "s" : "") : "aucune", hit > 0);
    chip("#chipMiss", P.n_missing ? fmt(P.n_missing) + " cellules" : "aucune", P.n_missing > 0);
    chip("#chipOut", P.n_outliers ? fmt(P.n_outliers) + " lignes (One-Class SVM)" : "aucun", P.n_outliers > 0);
    var cats = P.columns.filter(function (c) { return c.kind === "catégorielle" && c.name !== tgt; }).length;
    chip("#chipEnc", cats ? cats + " colonne" + (cats > 1 ? "s" : "") + " texte" : "aucune", cats > 0);
    $("#teWarn").hidden = !($("#optTE").checked && !tgt && cats > 0);
    var nums = P.columns.filter(function (c) { return c.kind === "numérique" && c.name !== tgt && c.n_unique > 2; }).length;
    chip("#chipScale", nums ? nums + " colonne" + (nums > 1 ? "s" : "") + " numérique" + (nums > 1 ? "s" : "") : "aucune", false);
    var hints = { standard: "Chaque variable est centrée (moyenne 0) puis divisée par son écart-type. Adapté à la régression linéaire/logistique, SVM, KNN et réseaux de neurones.",
                  minmax: "Chaque variable est ramenée entre 0 et 1. Sensible aux valeurs extrêmes : à utiliser après le traitement des outliers.",
                  robust: "Chaque variable est centrée sur la médiane et divisée par l'écart interquartile : peu influencée par les outliers." };
    $("#scaleHint").textContent = hints[$("#optScaleMethod").value] || "";

    /* SMOTE panel : class distribution of the selected target */
    var tc = P.columns.filter(function (c) { return c.name === tgt; })[0], cl = tc && tc.classes, box = $("#classBars"), bw = $("#balWarn");
    bw.hidden = true;
    if (!tgt) { box.innerHTML = ""; bw.textContent = "Choisissez une variable cible (colonne de gauche) pour voir la distribution des classes."; bw.hidden = false; chip("#chipBal", "", false); }
    else if (!cl) { box.innerHTML = ""; bw.textContent = "La cible « " + tgt + " » n'est pas une variable de classification (2 à 20 valeurs distinctes)."; bw.hidden = false; chip("#chipBal", "", false); }
    else {
      box.innerHTML = barsHtml(cl);
      var v = Object.keys(cl).map(function (k) { return cl[k]; }), r = Math.min.apply(null, v) / Math.max.apply(null, v);
      chip("#chipBal", r < 0.5 ? "déséquilibré (1 : " + (1 / r).toFixed(1) + ")" : "équilibré", r < 0.5);
      if (P.dataset_type === "time_series") { bw.textContent = "SMOTE est ignoré pour les séries temporelles."; bw.hidden = false; }
    }
    $$("#ppCols tr[data-col]").forEach(function (tr) { tr.classList.toggle("is-target", tr.dataset.col === tgt); });
    syncRecap();
  }
  ["#optThr", "#optTarget", "#optTE", "#optBal", "#optScale", "#optScaleMethod", "#optDup", "#optUseless", "#optImpute", "#optCap", "#optNu", "#optWin", "#optSmK"].forEach(function (s) { $(s).addEventListener("input", refresh); });

  /* ---------- column table ---------- */
  var kindCls = { "numérique": "num", "catégorielle": "cat", "date": "date", "id": "id", "constante": "const" };
  function colsTable() {
    var h = "<thead><tr><th>Colonne</th><th>Type</th><th>Manquants</th><th>Uniques</th><th>Outliers</th><th>Garder</th></tr></thead><tbody>";
    P.columns.forEach(function (c) {
      var w = Math.min(100, c.pct_missing);
      h += '<tr data-col="' + esc(c.name) + '"><td class="nm" title="' + esc(c.dtype) + '">' + esc(c.name) + '</td>' +
        '<td><span class="pp-kind ' + (kindCls[c.kind] || "") + '">' + esc(c.kind) + '</span></td>' +
        '<td><div class="pp-bar" title="' + fmt(c.n_missing) + '"><i style="width:' + w + '%"></i></div><span class="pp-pc">' + (c.pct_missing ? c.pct_missing + " %" : "—") + '</span></td>' +
        '<td>' + fmt(c.n_unique) + '</td><td>' + (c.n_outliers ? fmt(c.n_outliers) : "—") + '</td>' +
        '<td><label class="pp-check"><input type="checkbox" class="keep" data-col="' + esc(c.name) + '" checked><span></span></label></td></tr>';
    });
    $("#ppCols").innerHTML = h + "</tbody>";
    chip("#chipCols", P.columns.length + " colonnes");
  }
  colsTable();

  /* ---------- step results : computed automatically, like the EDA pages ---------- */
  var status = $("#ppStatus"), cache = {}, seq = {}, timer = null;
  function setStatus(t, bad) { if (!status) return; status.textContent = t || ""; status.classList.toggle("bad", !!bad); }
  function post(url, cfg) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok || j.status !== "ok") throw new Error(j.message || "Erreur serveur"); return j; }); });
  }
  function box(key) { return $('.pp-res[data-res="' + key + '"]'); }
  function themeLayout(extra) {
    var cs = getComputedStyle(document.documentElement), fg = (cs.getPropertyValue("--text-muted") || "#94a3b8").trim() || "#94a3b8";
    var l = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: fg, size: 12 }, margin: { l: 50, r: 20, t: 30, b: 45 },
              legend: { orientation: "h", y: 1.12 }, xaxis: { gridcolor: "rgba(148,163,184,.18)" }, yaxis: { gridcolor: "rgba(148,163,184,.18)" } };
    for (var k in extra) { l[k] = extra[k]; }
    return l;
  }
  var PCFG = { displayModeBar: false, responsive: true };
  function plot(el, data, layout) {
    if (typeof Plotly === "undefined") { el.innerHTML = '<p class="pp-warn">Plotly n’est pas chargé (vérifiez static/js/plotly.min.js).</p>'; return; }
    Plotly.newPlot(el, data, themeLayout(layout), PCFG);
  }
  function delta(a, b, unit) {
    var d = b - a; if (!d) return '<b>' + fmt(b) + "</b>";
    return "<s>" + fmt(a) + "</s> → <b>" + fmt(b) + '</b> <span class="pp-delta ' + (d < 0 ? "down" : "up") + '">' + (d > 0 ? "+" : "") + fmt(d) + "</span>";
  }
  function head(j) {
    var h = '<div class="pp-rhead"><div class="pp-rstat"><span class="pp-lab">Lignes</span>' + delta(j.before.rows, j.after.rows) + "</div>" +
      '<div class="pp-rstat"><span class="pp-lab">Colonnes</span>' + delta(j.before.cols, j.after.cols) + "</div>" +
      '<div class="pp-rstat"><span class="pp-lab">Valeurs manquantes</span>' + delta(j.before.missing, j.after.missing) + "</div></div>";
    if (!j.enabled) h += '<p class="pp-warn">Cette étape est désactivée : les données ne sont pas modifiées.</p>';
    else if (j.detail) h += '<p class="pp-detail">' + esc(j.detail) + "</p>";
    return '<div class="section-card pp-resbox">' + h;      /* closed by each renderer */
  }
  function table(pv) {
    return '<div class="pp-table-wrap" style="max-height:300px;"><table class="pp-table"><thead><tr>' + pv.columns.map(function (c) { return "<th>" + esc(c) + "</th>"; }).join("") +
      "</tr></thead><tbody>" + pv.rows.map(function (r) { return "<tr>" + r.map(cell).join("") + "</tr>"; }).join("") + "</tbody></table></div>";
  }
  function varSelect(id, names, label) {
    return '<label class="pp-field" style="max-width:280px;margin:1rem 0 .4rem;"><span class="pp-lab">' + label + '</span><select id="' + id + '">' +
      names.map(function (n) { return '<option value="' + esc(n) + '">' + esc(n) + "</option>"; }).join("") + "</select></label>";
  }
  function empty(msg) { return '<p class="pp-detail" style="color:var(--text-muted);">' + msg + "</p>"; }

  var R = {};
  R.duplicates = function (j, el) {
    el.innerHTML = head(j) + (j.sample ? '<h4 class="pp-sub">Lignes en double (aperçu)</h4>' + table(j.sample) : empty("Aucun doublon dans ce dataset.")) + "</div>";
  };
  R.useless = function (j, el) {
    var h = head(j);
    if (j.dropped.length) h += '<h4 class="pp-sub">Colonnes retirées</h4><ul class="pp-list">' + j.dropped.map(function (d) {
      return "<li><b>" + esc(d.name) + "</b><span>" + esc(d.reason) + "</span></li>"; }).join("") + "</ul>";
    else h += empty("Aucune colonne retirée.");
    el.innerHTML = h + "</div>";
  };
  R.missing = function (j, el) {
    var h = head(j);
    if (!j.columns.length) { el.innerHTML = h + empty("Aucune valeur manquante à ce stade.") + "</div>"; return; }
    h += '<div class="pp-chart" id="chMiss"></div><h4 class="pp-sub">Par colonne</h4><div class="pp-table-wrap" style="max-height:300px;"><table class="pp-table"><thead><tr><th>Colonne</th><th>Manquants avant</th><th>après</th><th>Moyenne avant → après</th><th>Écart-type avant → après</th></tr></thead><tbody>' +
      j.columns.map(function (c) {
        return "<tr><td class=\"nm\">" + esc(c.name) + "</td><td>" + fmt(c.before) + "</td><td>" + fmt(c.after) + "</td>" +
          (c.kind === "num" ? "<td>" + c.mean_before + " → " + c.mean_after + "</td><td>" + c.std_before + " → " + c.std_after + "</td>" : '<td class="nul">texte</td><td class="nul">—</td>') + "</tr>";
      }).join("") + "</tbody></table></div>";
    if (j.examples && j.examples.length) h += '<h4 class="pp-sub">Exemples de valeurs imputées</h4><div class="pp-chips">' + j.examples.map(function (e) {
      return '<span class="pp-chip">' + esc(e.col) + " · ligne " + e.row + " → <b>" + esc(e.value) + "</b></span>"; }).join("") + "</div>";
    el.innerHTML = h + "</div>";
    var cs = j.columns.slice(0, 25), names = cs.map(function (c) { return c.name; });
    plot($("#chMiss", el), [{ type: "bar", name: "Avant", x: names, y: cs.map(function (c) { return c.before; }), marker: { color: "#f59e0b" } },
                            { type: "bar", name: "Après", x: names, y: cs.map(function (c) { return c.after; }), marker: { color: "#14b8a6" } }],
         { barmode: "group", height: 300, yaxis: { title: "valeurs manquantes", gridcolor: "rgba(148,163,184,.18)" } });
  };
  R.outliers = function (j, el) {
    var h = head(j), V = j.svm;
    if (!j.columns.length) { el.innerHTML = h + empty("Aucune variable numérique analysable.") + "</div>"; return; }
    if (V) {
      h += '<p class="pp-detail">One-Class SVM apprend la <b>région normale</b> du train en <b>toutes les variables à la fois</b> : une ligne est retirée quand elle sort de cette région, même si chaque valeur prise seule paraît normale. Un boxplot par variable (règle IQR) n’est donc pas adapté ; voici les vues propres à cette méthode.</p>';
      h += '<h4 class="pp-sub">1 · Région apprise par le SVM (projection sur les 2 premières composantes)</h4><div class="pp-chart" id="chOutPca" style="min-height:420px;"></div>';
      h += '<div class="pp-two"><div><h4 class="pp-sub">2 · Score de décision</h4><div class="pp-chart" id="chOutScore"></div></div>' +
           '<div><h4 class="pp-sub">3 · Variables qui rendent les lignes suspectes</h4><div class="pp-chart" id="chOutProf"></div></div></div>';
      if (V.profile.length) h += '<h4 class="pp-sub">4 · Lignes conservées vs retirées, par variable</h4>' + varSelect("selOut", V.profile.map(function (s) { return s.name; }), "Variable à afficher") + '<div class="pp-chart" id="chOut"></div>';
      if (V.n_protected) h += '<p class="pp-warn">' + fmt(V.n_protected) + " lignes détectées ont été conservées pour ne pas supprimer plus de la moitié d’une classe.</p>";
    }
    h += '<h4 class="pp-sub">Détection · One-Class SVM (ν = ' + Math.round(j.nu * 1000) / 10 + ' %) : ' + fmt(j.n_rows_flagged) + " lignes aberrantes (" + j.pct_rows_flagged + ' % du train) supprimées · ' + fmt(j.rows_before) + " → " + fmt(j.rows_after) + ' lignes au total (le test n\'est jamais modifié)</h4><div class="pp-table-wrap" style="max-height:300px;"><table class="pp-table"><thead><tr><th>Variable</th><th>Min → Max avant</th><th>Min → Max après</th><th>Écart-type avant → après</th></tr></thead><tbody>' +
      j.columns.map(function (c) {
        return "<tr><td class=\"nm\">" + esc(c.name) + "</td><td>" + c.min_before + " → " + c.max_before + "</td><td>" + c.min_after + " → " + c.max_after + "</td><td>" + c.std_before + " → " + c.std_after + "</td></tr>";
      }).join("") + "</tbody></table></div>";
    el.innerHTML = h + "</div>";
    if (!V) return;
    var KEEP = "#14b8a6", OUT = "#ef4444", grid = "rgba(148,163,184,.18)";

    /* 1 · PCA slice + decision boundary */
    if (V.pca) {
      var g = V.pca.grid;
      plot($("#chOutPca", el), [
        { type: "heatmap", x: g.x, y: g.y, z: g.z, zmid: 0, colorscale: [[0, "rgba(239,68,68,.30)"], [0.5, "rgba(148,163,184,.06)"], [1, "rgba(20,184,166,.22)"]], showscale: false, hoverinfo: "skip" },
        { type: "contour", x: g.x, y: g.y, z: g.z, contours: { start: 0, end: 0, size: 1, coloring: "lines" }, line: { color: "#0f172a", width: 2.5 }, showscale: false, hoverinfo: "skip", name: "Frontière du SVM", showlegend: true },
        { type: "scattergl", mode: "markers", name: "Conservées (" + fmt(V.pca.n_shown[0]) + " affichées)", x: V.pca.kept.x, y: V.pca.kept.y, marker: { color: KEEP, size: 4, opacity: .55 } },
        { type: "scattergl", mode: "markers", name: "Retirées (" + fmt(V.pca.n_shown[1]) + " affichées)", x: V.pca.flagged.x, y: V.pca.flagged.y, marker: { color: OUT, size: 6, opacity: .85, line: { color: "#fff", width: .5 } } }
      ], { height: 420, xaxis: { title: "Composante 1 (" + V.pca.var[0] + " % de la variance)", gridcolor: grid, zeroline: false },
           yaxis: { title: "Composante 2 (" + V.pca.var[1] + " %)", gridcolor: grid, zeroline: false }, margin: { l: 55, r: 20, t: 30, b: 50 } });
    } else { $("#chOutPca", el).innerHTML = empty("Au moins 2 variables continues sont nécessaires pour la projection."); }

    /* 2 · decision score histogram */
    var all = V.score_kept.concat(V.score_flagged), mn = Math.min.apply(null, all), mx = Math.max.apply(null, all), sz = (mx - mn) / 40 || 1;
    plot($("#chOutScore", el), [
      { type: "histogram", name: "Conservées", x: V.score_kept, xbins: { start: mn, end: mx, size: sz }, marker: { color: KEEP }, opacity: .85 },
      { type: "histogram", name: "Retirées", x: V.score_flagged, xbins: { start: mn, end: mx, size: sz }, marker: { color: OUT }, opacity: .9 }
    ], { height: 340, barmode: "overlay", xaxis: { title: "decision_function (< 0 = hors de la région normale)", gridcolor: grid },
         yaxis: { title: "lignes", gridcolor: grid },
         shapes: [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: "#0f172a", width: 2, dash: "dash" } }] });

    /* 3 · which variables are abnormal in the flagged rows (mean |z|) */
    var pr = V.profile.slice().reverse();
    plot($("#chOutProf", el), [
      { type: "bar", orientation: "h", name: "Conservées", y: pr.map(function (r) { return r.name; }), x: pr.map(function (r) { return r.z_kept; }), marker: { color: KEEP } },
      { type: "bar", orientation: "h", name: "Retirées", y: pr.map(function (r) { return r.name; }), x: pr.map(function (r) { return r.z_flagged; }), marker: { color: OUT } }
    ], { height: Math.max(340, 70 + 30 * pr.length), barmode: "group", margin: { l: 130, r: 20, t: 30, b: 50 },
         xaxis: { title: "écart moyen à la normale (|z|)", gridcolor: grid } });

    /* 4 · kept vs flagged distribution of one variable (original units) */
    if (!V.profile.length) return;
    function draw() {
      var s = V.profile.filter(function (x) { return x.name === $("#selOut", el).value; })[0];
      plot($("#chOut", el), [
        { type: "violin", name: "Conservées", y: s.kept, side: "negative", spanmode: "hard", points: false, line: { color: KEEP }, fillcolor: "rgba(20,184,166,.45)", meanline: { visible: true }, scalegroup: "o" },
        { type: "violin", name: "Retirées", y: s.flagged, side: "positive", spanmode: "hard", points: "all", jitter: .3, pointpos: .6, marker: { color: OUT, size: 3, opacity: .6 }, line: { color: OUT }, fillcolor: "rgba(239,68,68,.35)", meanline: { visible: true }, scalegroup: "o" }
      ], { height: 340, violinmode: "overlay", title: { text: s.name + " · |z| moyen : " + s.z_kept + " (conservées) vs " + s.z_flagged + " (retirées)", font: { size: 13 } } });
    }
    $("#selOut", el).addEventListener("change", draw); draw();
  };
  R.noise = function (j, el) {
    if (!j.applicable) { el.innerHTML = head(j) + empty("Le lissage par moyenne mobile ne s'applique qu'aux séries temporelles.") + "</div>"; return; }
    if (!j.enabled) { el.innerHTML = head(j) + empty("Indiquez une fenêtre ≥ 2 pour lisser la série.") + "</div>"; return; }
    var h = head(j) + varSelect("selNoise", j.series.map(function (s) { return s.name; }), "Variable à afficher") + '<div class="pp-chart" id="chNoise"></div>' +
      '<h4 class="pp-sub">Rugosité (écart-type des variations d\'un point à l\'autre)</h4><div class="pp-table-wrap" style="max-height:260px;"><table class="pp-table"><thead><tr><th>Variable</th><th>Avant</th><th>Après</th><th>Réduction</th></tr></thead><tbody>' +
      j.columns.map(function (c) { return "<tr><td class=\"nm\">" + esc(c.name) + "</td><td>" + c.rough_before + "</td><td>" + c.rough_after + "</td><td>" + (c.reduction === null ? "—" : c.reduction + " %") + "</td></tr>"; }).join("") + "</tbody></table></div>";
    el.innerHTML = h + "</div>";
    function draw() {
      var s = j.series.filter(function (x) { return x.name === $("#selNoise", el).value; })[0];
      plot($("#chNoise", el), [{ type: "scatter", mode: "lines", name: "Original", x: j.x, y: s.raw, line: { color: "#94a3b8", width: 1 }, opacity: .6 },
                               { type: "scatter", mode: "lines", name: "Moyenne mobile (" + j.window + ")", x: j.x, y: s.smooth, line: { color: "#14b8a6", width: 2 } }], { height: 360 });
    }
    $("#selNoise", el).addEventListener("change", draw); draw();
  };
  R.encoding = function (j, el) {
    var h = head(j);
    if (j.needs_target) h += '<p class="pp-warn">Aucune variable cible : encodage par fréquence utilisé à la place.</p>';
    var cols = j.columns.filter(function (c) { return c.mapping; });
    if (!j.columns.length) { el.innerHTML = h + empty("Aucune colonne texte à encoder.") + "</div>"; return; }
    if (cols.length) h += varSelect("selEnc", cols.map(function (c) { return c.name; }), "Colonne à afficher") + '<div class="pp-chart" id="chEnc"></div>';
    if (j.sample) h += '<h4 class="pp-sub">Avant</h4>' + table(j.sample.before) + '<h4 class="pp-sub">Après</h4>' + table(j.sample.after);
    if (j.target_mapping) h += '<div class="pp-map"><span class="pp-lab">Cible</span>' + Object.keys(j.target_mapping).map(function (k) { return '<span class="pp-chip">' + esc(k) + " → " + j.target_mapping[k] + "</span>"; }).join("") + "</div>";
    el.innerHTML = h + "</div>";
    if (!cols.length) return;
    function draw() {
      var c = cols.filter(function (x) { return x.name === $("#selEnc", el).value; })[0], m = c.mapping.slice().reverse();
      plot($("#chEnc", el), [{ type: "bar", orientation: "h", y: m.map(function (r) { return r.cat; }), x: m.map(function (r) { return r.value; }),
                               text: m.map(function (r) { return fmt(r.count) + " lignes"; }), hovertemplate: "%{y}: %{x}<br>%{text}<extra></extra>", marker: { color: "#38bdf8" } }],
           { height: Math.max(240, 60 + 26 * m.length), margin: { l: 110, r: 20, t: 20, b: 45 }, xaxis: { title: "valeur encodée (moyenne)", gridcolor: "rgba(148,163,184,.18)" } });
    }
    $("#selEnc", el).addEventListener("change", draw); draw();
  };
  R.scaling = function (j, el) {
    var h = head(j);
    if (!j.enabled) { el.innerHTML = h + "</div>"; return; }
    if (!j.columns.length) { el.innerHTML = h + empty("Aucune colonne numérique à mettre à l'échelle.") + "</div>"; return; }
    h += varSelect("selScale", j.series.map(function (s) { return s.name; }), "Variable à afficher") + '<div class="pp-chart" id="chScale"></div>' +
      '<h4 class="pp-sub">Statistiques avant → après</h4><div class="pp-table-wrap" style="max-height:300px;"><table class="pp-table"><thead><tr><th>Variable</th><th>Moyenne</th><th>Écart-type</th><th>Min → Max avant</th><th>Min → Max après</th></tr></thead><tbody>' +
      j.columns.map(function (c) {
        return "<tr><td class=\"nm\">" + esc(c.name) + "</td><td>" + c.mean_before + " → " + c.mean_after + "</td><td>" + c.std_before + " → " + c.std_after + "</td><td>" + c.min_before + " → " + c.max_before + "</td><td>" + c.min_after + " → " + c.max_after + "</td></tr>";
      }).join("") + "</tbody></table></div>";
    el.innerHTML = h + "</div>";
    function draw() {
      var s = j.series.filter(function (x) { return x.name === $("#selScale", el).value; })[0];
      plot($("#chScale", el), [{ type: "histogram", name: "Avant", x: s.before, xaxis: "x", yaxis: "y", marker: { color: "#f59e0b" }, opacity: .85 },
                               { type: "histogram", name: "Après", x: s.after, xaxis: "x2", yaxis: "y2", marker: { color: "#14b8a6" }, opacity: .85 }],
           { height: 320, grid: { rows: 1, columns: 2, pattern: "independent" }, title: { text: s.name, font: { size: 13 } } });
    }
    $("#selScale", el).addEventListener("change", draw); draw();
  };
  R.smote = function (j, el) {
    var h = head(j);
    if (!j.counts_before) { el.innerHTML = h + empty("Choisissez une cible de classification pour utiliser SMOTE.") + "</div>"; return; }
    h += '<div class="pp-two"><div><h4 class="pp-sub">Classes · avant</h4><div class="pp-classes">' + barsHtml(j.counts_before) + '</div></div>' +
      '<div><h4 class="pp-sub">Classes · après</h4><div class="pp-classes">' + barsHtml(j.counts_after, "after") + "</div></div></div>";
    var feats = j.features || [];
    if (feats.length >= 2) h += '<div class="pp-two" style="margin-top:1rem;"><label class="pp-field"><span class="pp-lab">Axe X</span><select id="selSx">' + feats.map(function (f) { return "<option>" + esc(f) + "</option>"; }).join("") +
      '</select></label><label class="pp-field"><span class="pp-lab">Axe Y</span><select id="selSy">' + feats.map(function (f, i) { return "<option" + (i === 1 ? " selected" : "") + ">" + esc(f) + "</option>"; }).join("") + '</select></label></div><div class="pp-chart" id="chSm"></div>';
    el.innerHTML = h + "</div>";
    if (feats.length < 2) return;
    function draw() {
      var x = $("#selSx", el).value, y = $("#selSy", el).value;
      plot($("#chSm", el), [{ type: "scattergl", mode: "markers", name: "Originaux", x: j.orig[x], y: j.orig[y], text: j.orig_y, marker: { color: "#94a3b8", size: 5, opacity: .55 } },
                            { type: "scattergl", mode: "markers", name: "Synthétiques (SMOTE)", x: j.synth[x], y: j.synth[y], text: j.synth_y, marker: { color: "#14b8a6", size: 6, opacity: .75 } }],
           { height: 380, xaxis: { title: x, gridcolor: "rgba(148,163,184,.18)" }, yaxis: { title: y, gridcolor: "rgba(148,163,184,.18)" } });
    }
    $("#selSx", el).addEventListener("change", draw); $("#selSy", el).addEventListener("change", draw); draw();
  };

  function loadStep(panel) {
    var step = STEP_OF[panel], el = box(panel);
    if (!step || !el) return;
    var cfg = readCfg(), key = step + "|" + JSON.stringify(cfg);
    if (cache[key]) { R[step](cache[key], el); return; }
    var my = seq[step] = (seq[step] || 0) + 1;
    el.innerHTML = '<div class="section-card pp-resbox pp-loading"><span class="pp-spin"></span> Calcul en cours…</div>';
    post("/api/preprocessing/" + encodeURIComponent(D.dataset) + "/step/" + step, cfg)
      .then(function (j) { if (my !== seq[step]) return; cache[key] = j; R[step](j, el); })
      .catch(function (e) { if (my !== seq[step]) return; el.innerHTML = '<div class="section-card pp-resbox"><p class="pp-warn">' + esc(e.message || "Serveur injoignable") + "</p></div>"; });
  }
  /* any change of a control recomputes the step currently open (debounced) */
  function schedule() { clearTimeout(timer); timer = setTimeout(function () { if (STEP_OF[current]) loadStep(current); }, 450); }
  var mainEl = $(".eda-main-content"), sideEl = $(".eda-sidebar");
  [mainEl, sideEl].forEach(function (c) {
    c.addEventListener("input", function (e) { if (e.target.closest(".pp-res")) return; schedule(); });
    c.addEventListener("change", function (e) { if (e.target.closest(".pp-res")) return; schedule(); });
  });
  $$(".pp-seg button").forEach(function (b) { b.addEventListener("click", schedule); });

  /* ---------- final result : recap of the choices + explicit « Appliquer » ---------- */
  var lastWin = 5;
  var RECAP = [
    { label: "Doublons", panel: "dups", on: function () { return $("#optDup").checked; }, set: function (v) { $("#optDup").checked = v; },
      desc: function () { return "Supprimer les lignes identiques"; } },
    { label: "Colonnes inutiles", panel: "useless", on: function () { return $("#optUseless").checked; }, set: function (v) { $("#optUseless").checked = v; },
      desc: function () { return "Identifiants, constantes, colonnes vides à plus de " + $("#optThr").value + " %"; } },
    { label: "Valeurs manquantes", panel: "missing", on: function () { return $("#optImpute").checked; }, set: function (v) { $("#optImpute").checked = v; },
      desc: function () { return "Imputation KNN (numérique et texte)"; } },
    { label: "Outliers", panel: "outliers", on: function () { return $("#optCap").checked; }, set: function (v) { $("#optCap").checked = v; },
      desc: function () { return "One-Class SVM (ν = " + ($("#optNu").value || 5) + " %) : lignes aberrantes supprimées du train"; } },
    { label: "Bruit", panel: "noise", ts: true, on: function () { return (+$("#optWin").value || 0) >= 2; },
      set: function (v) { var w = +$("#optWin").value || 0; if (w >= 2) lastWin = w; $("#optWin").value = v ? lastWin : 0; },
      desc: function () { return P.dataset_type === "time_series" ? "Moyenne mobile (fenêtre " + (+$("#optWin").value >= 2 ? $("#optWin").value : lastWin) + ")" : "Disponible uniquement pour les séries temporelles"; } },
    { label: "Encodage", panel: "encoding", on: function () { return $("#optTE").checked; }, set: function (v) { $("#optTE").checked = v; },
      desc: function () { return "Target encoding (lissage " + ($("#optTESm").value || 0) + ")"; } },
    { label: "Feature Scaling", panel: "scaling", on: function () { return $("#optScale").checked; }, set: function (v) { $("#optScale").checked = v; },
      desc: function () { var o = $("#optScaleMethod"); return o.options[o.selectedIndex].textContent.split(" — ")[0] + " scaler"; } },
    { label: "Déséquilibre des classes", panel: "balance", on: function () { return $("#optBal").checked; }, set: function (v) { $("#optBal").checked = v; },
      desc: function () { return "SMOTE (k = " + ($("#optSmK").value || 5) + ") sur le train"; } }
  ];
  (function buildRecap() {
    var ul = $("#ppRecap");
    ul.innerHTML = RECAP.map(function (r, i) {
      return '<li class="pp-recap-row" data-i="' + i + '"><div class="pp-recap-txt"><b>' + esc(r.label) + '</b><small></small></div>' +
        '<button type="button" class="pp-recap-cfg" data-panel="' + r.panel + '">Configurer ›</button>' +
        '<span class="pp-recap-badge"></span>' +
        '<label class="pp-switch"><input type="checkbox"><span></span></label></li>';
    }).join("");
    $$(".pp-recap-row", ul).forEach(function (li) {
      var r = RECAP[+li.dataset.i];
      $("input", li).addEventListener("change", function () { r.set(this.checked); refresh(); });
      $(".pp-recap-cfg", li).addEventListener("click", function () { showPanel(r.panel); });
    });
  })();
  function syncRecap() {
    $$("#ppRecap .pp-recap-row").forEach(function (li) {
      var r = RECAP[+li.dataset.i], on = !!r.on(), off = r.ts && P.dataset_type !== "time_series";
      var cb = $("input", li);
      cb.checked = on && !off; cb.disabled = !!off;
      li.classList.toggle("on", on && !off); li.classList.toggle("disabled", !!off);
      $("small", li).textContent = r.desc();
      $(".pp-recap-badge", li).textContent = off ? "Non applicable" : (on ? "Activé" : "Désactivé");
    });
    updateApplyState();
  }

  /* what makes two configs « the same » for the Apply button (split / target come from the Split step) */
  function sig(c) {
    return JSON.stringify([c.drop_duplicates, c.drop_useless, +c.missing_col_threshold, c.num_missing, c.cat_missing, c.outliers, +c.ocsvm_nu,
      +c.smooth_window || 0, c.encoding, +c.te_smoothing, !!c.encode_target, !!c.balance, +c.smote_k || 5, c.scaling || "none", (c.drop_columns || []).slice().sort()]);
  }
  var appliedSig = null, appliedAt = null, applying = false;
  if (P.last_run && P.last_run.config) { appliedSig = null; appliedAt = P.last_run.created_at; }
  function updateApplyState() {
    var el = $("#ppApplyState"), btn = $("#btnApply"), nxt = $("#btnNext"), sub = $("#navResult .pp-nav-sub");
    if (!el || applying) return;
    var done = appliedSig !== null, dirty = done && appliedSig !== sig(readCfg());
    if (!done) {
      el.className = "pp-apply-state"; btn.textContent = "Appliquer ce prétraitement"; nxt.hidden = true;
      el.innerHTML = appliedAt ? "<b>Dernier prétraitement enregistré le " + esc(String(appliedAt).replace("T", " à ")) + "</b> — cliquez sur Appliquer pour enregistrer les choix ci-dessus."
                               : "<b>Pas encore appliqué</b> — aucun dataset prétraité n'est enregistré pour les étapes suivantes.";
      if (sub) sub.textContent = "À appliquer";
    } else if (dirty) {
      el.className = "pp-apply-state warn"; btn.textContent = "Appliquer les modifications"; nxt.hidden = true;
      el.innerHTML = "<b>Modifications non appliquées</b> — les étapes suivantes utilisent encore la version précédente.";
      if (sub) sub.textContent = "Modifications à appliquer";
    } else {
      el.className = "pp-apply-state ok"; btn.textContent = "Réappliquer"; nxt.hidden = false;
      el.innerHTML = "<b>✓ Appliqué le " + esc(String(appliedAt).replace("T", " à ")) + "</b> — ce dataset est utilisé par le Split et les étapes suivantes.";
      if (sub) sub.textContent = "Appliqué ✓";
    }
  }
  $("#btnApply").addEventListener("click", function () {
    if (applying) return;
    var btn = this, cfg = readCfg(), el = $("#ppApplyState");
    applying = true; btn.disabled = true; btn.classList.add("busy");
    el.className = "pp-apply-state"; el.innerHTML = '<span class="pp-spin"></span> Application du pipeline complet…';
    post("/api/preprocessing/" + encodeURIComponent(D.dataset) + "/run", cfg)
      .then(function (j) {
        applying = false; appliedSig = sig(cfg); appliedAt = j.created_at;
        render(j); updateApplyState();
      })
      .catch(function (e) {
        applying = false;
        el.className = "pp-apply-state bad"; el.innerHTML = "<b>Échec de l'application</b> — " + esc(e.message || "Serveur injoignable");
      })
      .then(function () { btn.disabled = false; btn.classList.remove("busy"); });
  });

  function cell(v) {
    if (v === null || v === undefined) return '<td class="nul">—</td>';
    if (typeof v === "number") return "<td>" + (Number.isInteger(v) ? v : +v.toFixed(3)) + "</td>";
    return "<td>" + esc(String(v).slice(0, 40)) + "</td>";
  }
  function render(j) {
    setKpi("rows", j.after.rows, j.before.rows); setKpi("cols", j.after.cols, j.before.cols);
    setKpi("missing", j.after.missing, j.before.missing); setKpi("dups", 0, P.n_duplicates);
  }

  /* ---------- start: the last applied choices if any, otherwise the recommended settings ---------- */
  var startCfg = Object.assign({}, P.recommended), lc = P.last_run && P.last_run.config;
  if (lc) Object.keys(lc).forEach(function (k) { if (k !== "target" && k !== "test_size" && k !== "drop_columns") startCfg[k] = lc[k]; });
  applyCfg(startCfg);
  if (lc && lc.drop_columns && lc.drop_columns.length) {
    $$("#ppCols input.keep").forEach(function (c2) { if (lc.drop_columns.indexOf(c2.dataset.col) >= 0) c2.checked = false; });
  }
  if (lc) { appliedSig = sig(readCfg()); }          /* what is saved = what the toggles show */
  syncRecap();
  var h = (location.hash || "").replace("#pp-", "");
  showPanel(h && h !== "result" ? h : "overview");
})();
