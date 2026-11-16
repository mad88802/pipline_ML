/**
 * Agent chat flottant : décrit un dataset et génère un rapport PDF (voir agent.py).
 * Le dataset actif est lu dans <body data-dataset="...">.
 */
(function () {
  "use strict";
  var dataset = document.body.getAttribute("data-dataset") || "";
  var KEY = "mls.agent." + (dataset || "_global");
  var history = [];
  try { history = JSON.parse(sessionStorage.getItem(KEY) || "[]"); } catch (e) { history = []; }
  var busy = false;

  var SUGGESTIONS = dataset
    ? ["Décris ce dataset", "Quelles colonnes ont des valeurs manquantes ?", "Quelles variables sont les plus importantes ?", "Génère le rapport général en PDF", "PDF : résultats des modèles avec/sans bruit et avec/sans outliers"]
    : ["Quels datasets sont disponibles ?", "Décris le dernier dataset"];

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function inline(s) { return esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>"); }

  function render(md) {
    var html = "", list = null;
    function close() { if (list) { html += "</" + list + ">"; list = null; } }
    String(md).split(/\r?\n/).forEach(function (line) {
      var t = line.trim(), ul = /^[-*•]\s+/.test(t), ol = /^\d+[.)]\s+/.test(t);
      if (ul || ol) {
        var tag = ul ? "ul" : "ol";
        if (list !== tag) { close(); html += "<" + tag + ">"; list = tag; }
        html += "<li>" + inline(t.replace(/^([-*•]|\d+[.)])\s+/, "")) + "</li>";
      } else {
        close();
        if (/^#+\s/.test(t)) html += "<p><strong>" + inline(t.replace(/^#+\s*/, "")) + "</strong></p>";
        else if (t) html += "<p>" + inline(t) + "</p>";
      }
    });
    close();
    return html;
  }

  // ---------- DOM
  var IC = {
    chat: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    spark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9z"/><path d="M19 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/></svg>',
    reset: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
    send: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M6 11l6-6 6 6"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    pdf: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 14h6M9 17h4"/></svg>',
    dl: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7 11l5 5 5-5M5 20h14"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>'
  };
  var root = document.createElement("div");
  root.id = "agentRoot";
  root.innerHTML =
    '<button type="button" id="agentFab" aria-label="Ouvrir l’assistant" aria-expanded="false" aria-controls="agentPanel" title="Assistant data">' +
      '<span class="ag-i-chat">' + IC.chat + '</span><span class="ag-i-close">' + IC.close + '</span></button>' +
    '<section id="agentPanel" role="dialog" aria-label="Assistant data" hidden>' +
      '<header class="ag-head"><span class="ag-avatar">' + IC.spark + '</span>' +
      '<div class="ag-title"><strong>Assistant data</strong><small id="agentCtx"></small></div>' +
      '<div class="agent-actions"><button type="button" id="agentClear" title="Nouvelle conversation" aria-label="Nouvelle conversation">' + IC.reset + '</button>' +
      '<button type="button" id="agentClose" title="Fermer" aria-label="Fermer">' + IC.close + '</button></div></header>' +
      '<div id="agentMsgs" aria-live="polite"></div>' +
      '<div id="agentSugg"></div>' +
      '<form id="agentForm"><div class="ag-box"><textarea id="agentInput" rows="1" placeholder="Posez une question sur le dataset…" aria-label="Votre message"></textarea>' +
      '<button type="submit" id="agentSend" aria-label="Envoyer">' + IC.send + '</button></div>' +
      '<span class="ag-hint">Entrée pour envoyer · Maj + Entrée pour un retour à la ligne</span></form>' +
    '</section>';
  document.body.appendChild(root);

  var fab = document.getElementById("agentFab"), panel = document.getElementById("agentPanel"),
      msgs = document.getElementById("agentMsgs"), input = document.getElementById("agentInput"),
      sugg = document.getElementById("agentSugg"), form = document.getElementById("agentForm"),
      send = document.getElementById("agentSend");
  document.getElementById("agentCtx").textContent = dataset ? "Dataset : " + dataset : "Tous les datasets";

  function save() { try { sessionStorage.setItem(KEY, JSON.stringify(history.slice(-30))); } catch (e) {} }

  function hhmm(t) { try { return new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); } catch (e) { return ""; } }

  function copyText(txt, btn) {
    function done() { btn.classList.add("ok"); btn.innerHTML = IC.check; btn.title = "Copié"; setTimeout(function () { btn.classList.remove("ok"); btn.innerHTML = IC.copy; btn.title = "Copier"; }, 1400); }
    if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(txt).then(done, function () {}); return; }
    var ta = document.createElement("textarea"); ta.value = txt; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); done(); } catch (e) {} ta.remove();
  }

  /* Returns { row, bubble }: the row is the whole message line, the bubble its text container. */
  function addMsg(role, content, files, t) {
    var row = document.createElement("div");
    row.className = "ag-row ag-" + role;
    var col = document.createElement("div"); col.className = "ag-col";
    var d = document.createElement("div"); d.className = "ag-msg ag-" + role;
    d.innerHTML = role === "assistant" ? render(content) : "<p>" + esc(content) + "</p>";
    (files || []).forEach(function (f) {
      var a = document.createElement("a");
      a.className = "ag-file"; a.href = f.url; a.target = "_blank"; a.rel = "noopener"; a.download = f.name;
      a.innerHTML = '<span class="ag-file-ico">' + IC.pdf + '</span><span class="ag-file-txt"><b>' + esc(f.name) + '</b><small>Rapport PDF · Télécharger</small></span><span class="ag-file-dl">' + IC.dl + '</span>';
      d.appendChild(a);
    });
    col.appendChild(d);
    var meta = document.createElement("div"); meta.className = "ag-meta";
    if (t) meta.appendChild(document.createTextNode(hhmm(t)));
    if (role === "assistant" && content) {
      var cb = document.createElement("button"); cb.type = "button"; cb.className = "ag-copy"; cb.title = "Copier"; cb.setAttribute("aria-label", "Copier la réponse"); cb.innerHTML = IC.copy;
      cb.addEventListener("click", function () { copyText(content, cb); }); meta.appendChild(cb);
    }
    col.appendChild(meta);
    if (role === "assistant") { var av = document.createElement("span"); av.className = "ag-avatar"; av.innerHTML = IC.spark; row.appendChild(av); }
    row.appendChild(col);
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
    return { row: row, bubble: d };
  }

  function showSugg() {
    sugg.innerHTML = "";
    if (history.length) return;
    var h = document.createElement("span"); h.className = "ag-sugg-h"; h.textContent = "Suggestions"; sugg.appendChild(h);
    var l = document.createElement("div"); l.className = "ag-sugg-l"; sugg.appendChild(l);
    SUGGESTIONS.forEach(function (s) {
      var b = document.createElement("button");
      b.type = "button"; b.innerHTML = "<span></span>" + IC.arrow; b.firstChild.textContent = s;
      b.addEventListener("click", function () { ask(s); });
      l.appendChild(b);
    });
  }

  function restore() {
    msgs.innerHTML = "";
    if (!history.length) addMsg("assistant", "Bonjour ! Je peux décrire " + (dataset ? "**" + dataset + "**" : "n’importe quel dataset analysé") + " et générer un **rapport PDF**. Que veux-tu savoir ?");
    history.forEach(function (m) { addMsg(m.role, m.content, m.files, m.t); });
    showSugg();
  }

  function ask(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    busy = true; send.disabled = true;
    var now = Date.now();
    history.push({ role: "user", content: text, t: now }); save();
    addMsg("user", text, null, now); showSugg();
    input.value = ""; input.style.height = "auto";
    var wait = addMsg("assistant", ""); wait.row.classList.add("ag-pending");
    wait.bubble.innerHTML = '<span class="ag-dots" aria-label="Rédaction en cours"><i></i><i></i><i></i></span>';

    fetch("/api/agent/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: dataset, messages: history.map(function (m) { return { role: m.role, content: m.content }; }) })
    }).then(function (r) { return r.json().catch(function () { return { status: "error", message: "Réponse invalide du serveur" }; }); })
      .then(function (d) {
        wait.row.remove();
        if (d.status !== "ok") { var e = addMsg("assistant", d.message || "Erreur"); e.bubble.classList.add("ag-err"); history.pop(); save(); return; }
        var at = Date.now();
        history.push({ role: "assistant", content: d.reply, files: d.files || [], t: at }); save();
        addMsg("assistant", d.reply, d.files, at);
      })
      .catch(function () { wait.row.remove(); var e = addMsg("assistant", "Serveur injoignable."); e.bubble.classList.add("ag-err"); history.pop(); save(); })
      .then(function () { busy = false; send.disabled = false; input.focus(); });
  }

  function toggle(open) {
    panel.hidden = !open; fab.classList.toggle("on", open); fab.setAttribute("aria-expanded", open ? "true" : "false");
    fab.setAttribute("aria-label", open ? "Fermer l’assistant" : "Ouvrir l’assistant");
    if (open) { input.focus(); msgs.scrollTop = msgs.scrollHeight; }
  }

  fab.addEventListener("click", function () { toggle(panel.hidden); });
  document.getElementById("agentClose").addEventListener("click", function () { toggle(false); });
  document.getElementById("agentClear").addEventListener("click", function () { history = []; save(); restore(); });
  form.addEventListener("submit", function (e) { e.preventDefault(); ask(input.value); });
  input.addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); } e.stopPropagation(); });
  input.addEventListener("input", function () { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px"; });

  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !panel.hidden) { toggle(false); fab.focus(); } });

  restore();
})();
