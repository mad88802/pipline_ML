/**
 * Interprétation IA (Groq) sous chaque graphique.
 * Ajoute un panneau « Interpréter » à chaque explorer et aux graphiques statiques,
 * sans toucher aux scripts des explorers : la variable active est lue dans les chips.
 */
(function () {
  "use strict";
  var dataset = document.body.getAttribute("data-dataset");
  if (!dataset) return;

  var EXPLORERS = [
    { root: "noiseExplorer", resolve: function (r) { return { kind: "noise", feature: activeKey(r, ".nf-chip.active[data-key]") }; } },
    { root: "outlierExplorer", resolve: function (r) { return { kind: "outliers", feature: activeKey(r, ".nf-chip.active[data-key]") }; } },
    {
      root: "distExplorer",
      resolve: function (r) {
        var f = activeKey(r, ".nf-chip.active[data-key]");
        if (f === "__corr__") return { kind: "correlation" };
        if (f === "__target__") return { kind: "target" };
        return { kind: "distribution", feature: f };
      }
    },
    {
      root: "fiExplorer",
      resolve: function (r) {
        var m = r.querySelector("#fiMode .nf-chip.active");
        var mode = m ? m.getAttribute("data-mode") : "anova";
        if (mode === "corr") return { kind: "fi_corr", target: activeKey(r, "#fiTargets .nf-chip.active") };
        return { kind: "fi_anova", target: activeKey(r, "#fiAnovaTargets .nf-chip.active") };
      }
    }
  ];

  var STATIC_IMAGES = [
    { suffix: "missing_values.png", kind: "missing" },
    { suffix: "reference_noise.png", kind: "reference_noise" }
  ];

  function activeKey(root, sel) {
    var el = root.querySelector(sel);
    return el ? el.getAttribute("data-key") || "" : "";
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }

  // Markdown minimal : **gras**, puces, paragraphes
  function render(md) {
    var html = "", inList = false;
    md.split(/\r?\n/).forEach(function (line) {
      var t = line.trim();
      var bullet = /^[-*•]\s+/.test(t);
      var body = esc(bullet ? t.replace(/^[-*•]\s+/, "") : t).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      if (bullet) { if (!inList) { html += "<ul>"; inList = true; } html += "<li>" + body + "</li>"; return; }
      if (inList) { html += "</ul>"; inList = false; }
      if (t) html += "<p>" + body + "</p>";
    });
    return html + (inList ? "</ul>" : "");
  }

  function makePanel(getParams) {
    var panel = document.createElement("div");
    panel.className = "ai-panel";
    panel.innerHTML =
      '<div class="ai-head"><button type="button" class="ai-btn">✨ Interpréter avec l’IA</button>' +
      '<button type="button" class="ai-refresh" hidden title="Régénérer">↻</button></div>' +
      '<div class="ai-out" hidden></div>';
    var btn = panel.querySelector(".ai-btn"), refreshBtn = panel.querySelector(".ai-refresh"), out = panel.querySelector(".ai-out");
    var seq = 0;

    function show(html, cls) { out.hidden = false; out.className = "ai-out" + (cls ? " " + cls : ""); out.innerHTML = html; }

    function call(extra) {
      var p = getParams();
      var id = ++seq;
      var body = Object.assign({}, p, extra || {});
      return fetch("/api/interpret/" + encodeURIComponent(dataset), {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
      }).then(function (r) { return r.json().catch(function () { return { status: "error", message: "Réponse invalide" }; }); })
        .then(function (d) { return id === seq ? d : null; });   // ignore les réponses périmées (chip changée entre-temps)
    }

    function apply(d) {
      if (!d) return;
      if (d.status !== "ok") { show(esc(d.message || "Erreur"), "ai-err"); btn.disabled = false; btn.textContent = "✨ Réessayer"; return; }
      if (!d.text) { out.hidden = true; refreshBtn.hidden = true; btn.disabled = false; btn.textContent = "✨ Interpréter avec l’IA"; return; }
      show(render(d.text));
      refreshBtn.hidden = false; btn.disabled = false; btn.textContent = "✨ Interprétation";
    }

    function run(extra) {
      btn.disabled = true; btn.textContent = "Analyse en cours…";
      show('<span class="ai-dots"><i></i><i></i><i></i></span>', "ai-loading");
      call(extra).then(apply).catch(function () { apply({ status: "error", message: "Serveur injoignable" }); });
    }

    btn.addEventListener("click", function () { run(); });
    refreshBtn.addEventListener("click", function () { run(); });

    // Quand la variable change : on remet le panneau à zéro (rien n'est stocké, l'IA est relancée à chaque clic)
    panel.reset = function () {
      seq++;                                         // ignore une réponse encore en route pour l'ancienne variable
      out.hidden = true; refreshBtn.hidden = true; btn.disabled = false; btn.textContent = "✨ Interpréter avec l’IA";
    };
    return panel;
  }

  function init() {
    EXPLORERS.forEach(function (cfg) {
      var root = document.getElementById(cfg.root);
      if (!root) return;
      var panel = makePanel(function () { return cfg.resolve(root); });
      root.appendChild(panel);
      root.addEventListener("click", function (e) {
        if (e.target.closest(".nf-chip")) setTimeout(panel.reset, 80);   // après la mise à jour de la chip active
      });
      panel.reset();
    });

    STATIC_IMAGES.forEach(function (s) {
      var img = document.querySelector('img[src$="' + s.suffix + '"]');
      if (!img) return;
      var host = img.closest(".lightbox-trigger") || img;
      var panel = makePanel(function () { return { kind: s.kind }; });
      host.parentNode.insertBefore(panel, host.nextSibling);
      panel.reset();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
