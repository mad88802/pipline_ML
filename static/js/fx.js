/* ==========================================================================
   ML Pipeline Studio — interactive UI layer (presentation only).
   Nothing here touches the analysis, the data, or the server: it only adds
   effects, navigation helpers and client-side filtering of what is already
   rendered on the page.
   ========================================================================== */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const root = document.documentElement;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const coarse = matchMedia("(pointer: coarse)").matches;
  const store = {
    get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} },
  };
  const DATA = (() => { try { return JSON.parse($("#fx-data").textContent); } catch (e) { return {}; } })();
  const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
  const ICON = {
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    palette: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 1 0 0 18c1.1 0 1.8-.8 1.8-1.7 0-.5-.2-.9-.5-1.2-.3-.3-.5-.7-.5-1.2 0-.9.8-1.7 1.8-1.7H17a4 4 0 0 0 4-4C21 7 17 3 12 3Z"/><circle cx="7.5" cy="11" r="1"/><circle cx="10" cy="7" r="1"/><circle cx="15" cy="7" r="1"/></svg>',
    chart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>',
    up: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 14 6-6 6 6"/></svg>',
  };

  /* ---------- Toast ---------- */
  let toastEl, toastT;
  function toast(msg) {
    if (!toastEl) { toastEl = el("div"); toastEl.id = "fx-toast"; document.body.appendChild(toastEl); }
    toastEl.textContent = msg; toastEl.classList.add("show");
    clearTimeout(toastT); toastT = setTimeout(() => toastEl.classList.remove("show"), 1700);
  }

  /* ---------- Accent + chart mode ---------- */
  const ACCENTS = ["ink"];
  const curAccent = () => root.dataset.accent || "ink";
  function setAccent(a) { root.dataset.accent = a; store.set("mls.accent", a); }
  function cycleAccent() { const n = ACCENTS[(ACCENTS.indexOf(curAccent()) + 1) % ACCENTS.length]; setAccent(n); toast("Couleur : " + n); }
  function toggleCharts() {
    const dark = root.classList.toggle("charts-dark");
    store.set("mls.chartsmode", dark ? "dark" : "light");
    toast(dark ? "Graphiques : mode sombre" : "Graphiques : couleurs d'origine");
  }

  /* ---------- Spotlight glow on cards ---------- */
  const CARD_SEL = ".kpi-card,.dataset-card,.section-card,.plot-card,.acc-item,.pstep,.road-card,.independent-upload-card,.outlier-col-card,.hub-stat";
  function initGlow() {
    $$(CARD_SEL).forEach(c => {
      if (c.querySelector(":scope > .fx-glow")) return;
      c.classList.add("fx-card"); c.appendChild(el("i", "fx-glow"));
    });
    if (coarse) return;
    document.addEventListener("pointermove", e => {
      for (let c = e.target.closest && e.target.closest(".fx-card"); c; c = c.parentElement && c.parentElement.closest(".fx-card")) {
        const r = c.getBoundingClientRect();
        c.style.setProperty("--mx", (e.clientX - r.left) + "px");
        c.style.setProperty("--my", (e.clientY - r.top) + "px");
      }
    }, { passive: true });
  }

  /* ---------- Scroll reveal ---------- */
  function initReveal() {
    if (reduce || !("IntersectionObserver" in window)) return;
    const els = $$(".kpi-card,.dataset-card,.section-card,.plot-card,.acc-item,.road-item,.hub-stat,.fact,.outlier-col-card")
      .filter(n => !n.closest(".acc-body") || n.classList.contains("plot-card"));
    let batch = 0, tick = null;
    const show = n => { n.style.setProperty("--d", Math.min(batch++, 8) * 55 + "ms"); n.classList.add("in"); io.unobserve(n); };
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => { if (en.isIntersecting) show(en.target); });
      clearTimeout(tick); tick = setTimeout(() => { batch = 0; }, 120);
    }, { threshold: 0, rootMargin: "0px 0px 200px 0px" });
    els.forEach(n => { n.classList.add("fx-reveal"); io.observe(n); });
    let sw = null;
    const sweep = () => $$(".fx-reveal:not(.in)").forEach(n => { if (n.getBoundingClientRect().top < innerHeight + 200 && n.offsetParent) { n.classList.add("in"); io.unobserve(n); } });
    addEventListener("scroll", () => { clearTimeout(sw); sw = setTimeout(sweep, 120); }, { passive: true });
    addEventListener("hashchange", sweep); addEventListener("load", sweep); setTimeout(sweep, 1200);
  }

  /* ---------- Animated counters (final text is always the original) ---------- */
  function initCounters() {
    if (reduce || !("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver(entries => entries.forEach(en => {
      if (!en.isIntersecting) return;
      io.unobserve(en.target); count(en.target);
    }), { threshold: 0.4 });
    $$(".kpi-value,.d-stat-val,.hub-stat b,.fact b").forEach(n => {
      if (n.children.length) return;
      if (/^\s*[\d,]+(\.\d+)?\s*%?\s*$/.test(n.textContent) && parseFloat(n.textContent.replace(/,/g, "")) > 0) io.observe(n);
    });
    function count(n) {
      const txt = n.textContent.trim();
      const m = txt.match(/^([\d,]+(?:\.\d+)?)(\s*%)?$/); if (!m) return;
      const end = parseFloat(m[1].replace(/,/g, "")), dec = (m[1].split(".")[1] || "").length, grp = m[1].includes(",");
      const t0 = performance.now(), dur = 950;
      (function step(now) {
        const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(2, -10 * p);
        n.textContent = (end * (p >= 1 ? 1 : e)).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec, useGrouping: grp }) + (m[2] || "");
        if (p < 1) requestAnimationFrame(step); else n.textContent = txt;
      })(t0);
    }
  }

  /* ---------- 3D tilt on dataset cards ---------- */
  function initTilt() {
    if (reduce || coarse) return;
    $$(".dataset-card:not(.upload-card)").forEach(c => {
      c.addEventListener("pointermove", e => {
        const r = c.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5;
        c.style.transition = "transform .08s"; c.style.transform = `perspective(900px) rotateX(${(-y * 7).toFixed(2)}deg) rotateY(${(x * 9).toFixed(2)}deg) translateY(-4px)`;
      });
      c.addEventListener("pointerleave", () => { c.style.transition = ""; c.style.transform = ""; });
    });
  }

  /* ---------- Scroll progress ---------- */
  function initProgress() {
    const bar = el("div"); bar.id = "fx-progress"; document.body.appendChild(bar);
    const upd = () => { const h = root.scrollHeight - innerHeight; bar.style.transform = `scaleX(${h > 0 ? Math.min(1, scrollY / h) : 0})`; };
    addEventListener("scroll", upd, { passive: true }); upd();
  }

  /* ---------- Constellation background ---------- */
  function initCanvas() {
    if (reduce || (DATA.theme && DATA.theme !== "data")) return; /* themed pages use initThemeFx instead */
    const cv = el("canvas"); cv.id = "fx-canvas"; document.body.prepend(cv);
    const ctx = cv.getContext("2d"); let W, H, pts = [], mouse = { x: -999, y: -999 }, raf, rgb = "129,140,248";
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const readColor = () => { const v = getComputedStyle(root).getPropertyValue("--ac-2").trim().split(/\s+/); if (v.length === 3) rgb = v.join(","); };
    function resize() {
      W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.min(72, Math.floor(W * H / (W < 700 ? 30000 : 21000)));
      pts = Array.from({ length: n }, () => ({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .28, vy: (Math.random() - .5) * .28, r: Math.random() * 1.4 + .5 }));
    }
    function frame() {
      ctx.clearRect(0, 0, W, H);
      for (const p of pts) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W) p.vx *= -1; if (p.y < 0 || p.y > H) p.vy *= -1;
        const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
        if (d < 130 && d > 0) { p.x += dx / d * .7; p.y += dy / d * .7; }
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.283); ctx.fillStyle = `rgba(${rgb},.7)`; ctx.fill();
      }
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
        const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
        if (d < 125) { ctx.strokeStyle = `rgba(${rgb},${(1 - d / 125) * .22})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke(); }
      }
      for (const p of pts) { const d = Math.hypot(p.x - mouse.x, p.y - mouse.y); if (d < 170) { ctx.strokeStyle = `rgba(${rgb},${(1 - d / 170) * .4})`; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke(); } }
      raf = requestAnimationFrame(frame);
    }
    addEventListener("resize", resize);
    addEventListener("pointermove", e => { mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });
    document.addEventListener("visibilitychange", () => { cancelAnimationFrame(raf); if (!document.hidden) frame(); });
    new MutationObserver(readColor).observe(root, { attributes: true, attributeFilter: ["data-accent"] });
    readColor(); resize(); frame();
  }

  /* ---------- Dock ---------- */
  function initDock() {}

  /* ---------- Tables: search + sort (client-side, display only) ---------- */
  function initTables() {
    $$(".data-table").forEach(tbl => {
      if (tbl.closest(".lightbox-content") || tbl.dataset.fx) return;
      const tb = tbl.tBodies[0], head = tbl.tHead; if (!tb || !head) return; tbl.dataset.fx = "1";
      const rows = () => Array.from(tb.rows);
      head.querySelectorAll("th").forEach((th, idx) => {
        th.classList.add("sortable"); th.tabIndex = 0;
        const sort = () => {
          const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
          head.querySelectorAll("th").forEach(h => h.removeAttribute("aria-sort")); th.setAttribute("aria-sort", dir);
          const val = r => (r.cells[idx] ? r.cells[idx].textContent.trim() : "");
          const num = s => parseFloat(s.replace(/[,%\s]/g, ""));
          const all = rows(), numeric = all.every(r => { const v = val(r); return v === "" || v === "—" || !isNaN(num(v)); });
          all.sort((a, b) => { const x = val(a), y = val(b); const c = numeric ? (isNaN(num(x)) ? -Infinity : num(x)) - (isNaN(num(y)) ? -Infinity : num(y)) : x.localeCompare(y, undefined, { numeric: true }); return dir === "ascending" ? c : -c; });
          all.forEach(r => tb.appendChild(r));
        };
        th.addEventListener("click", sort); th.addEventListener("keydown", e => { if (e.key === "Enter") sort(); });
      });
    });
  }

  /* ---------- Plot gallery: filter chips + search ---------- */
  function initGallery() {
    $$(".plots-grid").forEach(grid => {
      if (grid.dataset.noGallery) return;
      const cards = $$(".plot-card", grid); if (cards.length < 4) return;
      const cats = [...new Set(cards.map(c => c.dataset.category).filter(Boolean))];
      const bar = el("div", "gal-tools"); let active = "all", q = "";
      const chips = el("div", "chips"); bar.appendChild(chips);
      const mkChip = (key, label, n) => { const b = el("button", "chip" + (key === "all" ? " active" : ""), `${label}<small>${n}</small>`); b.type = "button"; b.dataset.k = key; chips.appendChild(b); return b; };
      if (cats.length > 1) { mkChip("all", "Tous", cards.length); cats.forEach(c => mkChip(c, c.replace(/_/g, " "), cards.filter(x => x.dataset.category === c).length)); }
      const sb = el("div", "search-box", `${ICON.search}<input class="form-control" placeholder="Chercher une variable…" aria-label="Chercher un graphique">`); bar.appendChild(sb);
      grid.parentNode.insertBefore(bar, grid);
      const run = () => cards.forEach(c => { const ok = (active === "all" || c.dataset.category === active) && (!q || (c.dataset.title || "").toLowerCase().includes(q)); c.hidden = !ok; });
      chips.addEventListener("click", e => { const b = e.target.closest(".chip"); if (!b) return; active = b.dataset.k; $$(".chip", chips).forEach(x => x.classList.toggle("active", x === b)); run(); });
      $("input", sb).addEventListener("input", e => { q = e.target.value.trim().toLowerCase(); run(); });
    });
  }

  /* ---------- Lightbox: prev / next, counter, download ---------- */
  function initLightboxNav() {
    const modal = $("#lightboxModal"); if (!modal) return;
    const box = $(".lightbox-content", modal), img = $("#lightboxImage"), head = $(".lightbox-header", modal);
    let list = [], idx = 0;
    const prev = el("button", "lb-nav lb-prev", "‹"), next = el("button", "lb-nav lb-next", "›"), cnt = el("span", "lb-count");
    const dl = el("a", "btn btn-outline", "Télécharger"); dl.style.cssText = "font-size:.75rem;padding:.3rem .7rem;margin-right:.6rem"; dl.setAttribute("download", "");
    prev.type = next.type = "button"; prev.setAttribute("aria-label", "Précédent"); next.setAttribute("aria-label", "Suivant");
    box.append(prev, next); head.insertBefore(cnt, $("#lightboxClose")); head.insertBefore(dl, $("#lightboxClose"));
    document.addEventListener("click", e => {
      const t = e.target.closest(".lightbox-trigger"); if (!t) return;
      const scope = t.closest(".plots-grid") || document;
      list = $$(".lightbox-trigger", scope).filter(n => !n.closest("[hidden]")); idx = Math.max(0, list.indexOf(t));
    }, true);
    const go = d => { if (list.length < 2) return; idx = (idx + d + list.length) % list.length; list[idx].click(); };
    prev.addEventListener("click", () => go(-1)); next.addEventListener("click", () => go(1));
    document.addEventListener("keydown", e => {
      if (!modal.classList.contains("active")) return;
      if (e.key === "ArrowLeft") go(-1); else if (e.key === "ArrowRight") go(1);
    });
    new MutationObserver(() => {
      const open = modal.classList.contains("active");
      prev.hidden = next.hidden = list.length < 2; cnt.textContent = list.length > 1 ? `${idx + 1} / ${list.length}` : "";
      if (open && img.src) { dl.href = img.getAttribute("src"); dl.download = (($("#lightboxTitle").textContent || "plot").replace(/\s+/g, "_")) + ".png"; }
    }).observe(modal, { attributes: true, attributeFilter: ["class"] });
    new MutationObserver(() => { cnt.textContent = list.length > 1 ? `${idx + 1} / ${list.length}` : ""; if (img.getAttribute("src")) dl.href = img.getAttribute("src"); }).observe(img, { attributes: true, attributeFilter: ["src"] });
  }

  /* ---------- Keyboard + page transitions ---------- */
  function initKeys() {
    document.addEventListener("keydown", e => {
      const tag = (e.target.tagName || "").toLowerCase(), typing = tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable;
      if (e.key === "Escape") return;
      if (typing) return;
      if (e.altKey && e.key === "ArrowRight" && DATA.next) location.href = DATA.next;
      else if (e.altKey && e.key === "ArrowLeft" && DATA.prev) location.href = DATA.prev;
    });
  }
  function initTransitions() {
    if (reduce) return;
    document.addEventListener("click", e => {
      const a = e.target.closest("a[href]");
      if (!a || e.defaultPrevented || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      if (a.target === "_blank" || a.hasAttribute("download") || a.origin !== location.origin) return;
      if (a.pathname === location.pathname && a.search === location.search) return;
      e.preventDefault(); document.body.classList.add("fx-leaving"); setTimeout(() => { location.href = a.href; }, 170);
    });
    addEventListener("pageshow", () => document.body.classList.remove("fx-leaving"));
  }

  /* ---------- Dataset theme background (weather, health, energy...) ---------- */
  function initTheme() {
    if (!DATA.theme) return;
    root.dataset.theme = DATA.theme; document.body.dataset.theme = DATA.theme;
    if (DATA.themeColors) {
      root.style.setProperty("--th-a", DATA.themeColors[0]);
      root.style.setProperty("--th-b", DATA.themeColors[1]);
    }
    if (DATA.themeBg) root.style.setProperty("--th-img", 'url("' + DATA.themeBg + '")');
    if (!reduce && !coarse) {
      let pend = false;
      addEventListener("pointermove", e => {
        if (pend) return; pend = true;
        requestAnimationFrame(() => { pend = false; root.style.setProperty("--px", (e.clientX / innerWidth * 2 - 1).toFixed(3)); root.style.setProperty("--py", (e.clientY / innerHeight * 2 - 1).toFixed(3)); });
      }, { passive: true });
    }
    if (!$("#fx-theme-bg")) { const bg = el("div"); bg.id = "fx-theme-bg"; document.body.prepend(bg); }
  }

  /* ---------- Interactive themed particles (rain, floating symbols, sparks...) ---------- */
  const THEME_FX = {
    weather: { m: "rain", c: "37,99,235" }, water: { m: "rain", c: "14,165,233" },
    health: { m: "rise", g: "✚", c: "13,148,136" }, virus: { m: "rise", g: "●", c: "124,58,237" },
    energy: { m: "spark", c: "245,158,11" }, solar: { m: "spark", c: "234,179,8" }, wind: { m: "spark", c: "14,165,233" },
    sensor: { m: "spark", c: "6,182,212" }, car: { m: "spark", c: "79,70,229" },
    fire: { m: "rise", g: "●", c: "249,115,22", glow: 1 }, finance: { m: "rise", g: "$", c: "37,99,235" },
    music: { m: "rise", g: "♪", c: "147,51,234" }, movie: { m: "rise", g: "★", c: "239,68,68" },
    education: { m: "rise", g: "✦", c: "219,39,119" }, plant: { m: "rise", g: "❋", c: "22,163,74" },
    sports: { m: "rise", g: "●", c: "22,163,74" },
  };
  function initThemeFx() {
    const t = DATA.theme; if (t !== "weather" || reduce) return; /* every other theme has its own scene: see initSceneFx */
    const cfg = Object.assign({ m: "rise", g: "●", c: "99,102,241" }, THEME_FX[t] || {});
    const bg = $("#fx-theme-bg"); if (!bg) return;
    const cv = el("canvas"); cv.id = "fx-theme-canvas"; bg.after(cv);
    const ctx = cv.getContext("2d"), dpr = Math.min(devicePixelRatio || 1, 2);
    const rnd = (a, b) => a + Math.random() * (b - a);
    let W, H, ps = [], bursts = [], splashes = [], raf, wind = 0;
    const mouse = { x: -999, y: -999, vx: 0 };
    const make = init => {
      if (cfg.m === "rain") return { x: rnd(-120, W + 120), y: init ? rnd(0, H) : -20, v: rnd(9, 16), l: rnd(10, 22), a: rnd(.25, .55), vx: 0 };
      if (cfg.m === "spark") return { x: rnd(0, W), y: rnd(0, H), vx: rnd(-.4, .4), vy: rnd(-.4, .4), r: rnd(1.2, 2.8), a: rnd(.5, .9) };
      return { x: rnd(0, W), y: init ? rnd(0, H) : H + 30, s: rnd(9, 16), v: rnd(.2, .5), ph: rnd(0, 6.28), a: rnd(.14, .4), vx: 0, vy: 0 };
    };
    function resize() {
      W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const area = W * H, n = cfg.m === "rain" ? Math.min(70, area / 22000) : cfg.m === "spark" ? Math.min(38, area / 38000) : Math.min(16, area / 80000);
      ps = Array.from({ length: Math.floor(n) }, () => make(true));
    }
    function burst(x, y) {
      for (let i = 0; i < 8; i++) { const a = rnd(0, 6.283), v = rnd(1, 3); bursts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 1, life: 1, r: rnd(1.5, 3.5) }); }
      
    }
    function frame() {
      ctx.clearRect(0, 0, W, H); ctx.globalAlpha = .6;
      mouse.vx *= .9; wind += (Math.max(-9, Math.min(9, mouse.vx * .5)) - wind) * .06;
      const rgb = cfg.c;
      for (let i = 0; i < ps.length; i++) {
        const p = ps[i];
        if (cfg.m === "rain") {
          const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
          if (d < 110 && d > 0) p.vx += dx / d * .9;
          p.vx *= .9; p.x += wind * (p.v / 12) + p.vx; p.y += p.v;
          ctx.strokeStyle = `rgba(${rgb},${p.a})`; ctx.lineWidth = 1.4; ctx.lineCap = "round";
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x - (wind * (p.v / 12) + p.vx) * 1.6, p.y - p.l); ctx.stroke();
          if (p.y > H - rnd(10, 90)) { splashes.push({ x: p.x, y: H - rnd(2, 40), r: 1, a: .5 }); ps[i] = make(false); }
        } else if (cfg.m === "spark") {
          const dx = mouse.x - p.x, dy = mouse.y - p.y, d = Math.hypot(dx, dy);
          if (d < 240 && d > 1) { p.vx += dx / d * .07; p.vy += dy / d * .07; }
          p.vx += rnd(-.03, .03); p.vy += rnd(-.03, .03); p.vx *= .985; p.vy *= .985;
          p.x += p.vx; p.y += p.vy;
          if (p.x < 0 || p.x > W) p.vx *= -1; if (p.y < 0 || p.y > H) p.vy *= -1;
          ctx.fillStyle = `rgba(${rgb},${p.a})`; ctx.shadowColor = `rgba(${rgb},.9)`; ctx.shadowBlur = 10;
          ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.283); ctx.fill(); ctx.shadowBlur = 0;
          if (d < 150) { /* electric arc to the cursor */
            ctx.strokeStyle = `rgba(${rgb},${(1 - d / 150) * .7})`; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(p.x, p.y);
            for (let k = 1; k < 4; k++) ctx.lineTo(p.x + dx * k / 4 + rnd(-7, 7), p.y + dy * k / 4 + rnd(-7, 7));
            ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
          }
        } else {
          const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
          if (d < 130 && d > 0) { p.vx += dx / d * .6; p.vy += dy / d * .6; }
          p.vx *= .93; p.vy *= .93; p.ph += .012;
          p.x += Math.sin(p.ph) * .4 + p.vx; p.y -= p.v - p.vy;
          ctx.font = `${p.s}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillStyle = `rgba(${rgb},${p.a})`; if (cfg.glow) { ctx.shadowColor = `rgba(${rgb},.8)`; ctx.shadowBlur = 12; }
          ctx.fillText(cfg.g, p.x, p.y); ctx.shadowBlur = 0;
          if (p.y < -30) ps[i] = make(false);
        }
      }
      for (let i = splashes.length - 1; i >= 0; i--) {
        const q = splashes[i]; q.r += .6; q.a -= .03;
        if (q.a <= 0) { splashes.splice(i, 1); continue; }
        ctx.strokeStyle = `rgba(${rgb},${q.a})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.ellipse(q.x, q.y, q.r * 2.2, q.r * .7, 0, 0, 6.283); ctx.stroke();
      }
      for (let i = bursts.length - 1; i >= 0; i--) {
        const b = bursts[i]; b.x += b.vx; b.y += b.vy; b.vy += .09; b.vx *= .98; b.life -= .02;
        if (b.life <= 0) { bursts.splice(i, 1); continue; }
        ctx.fillStyle = `rgba(${rgb},${b.life * .8})`; ctx.beginPath(); ctx.arc(b.x, b.y, b.r * b.life + .5, 0, 6.283); ctx.fill();
      }
      raf = requestAnimationFrame(frame);
    }
    let lastX = 0;
    addEventListener("pointermove", e => { mouse.vx = e.clientX - lastX; lastX = e.clientX; mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });
    addEventListener("pointerleave", () => { mouse.x = mouse.y = -999; });
    addEventListener("pointerdown", e => { if (!e.target.closest("a,button,input,select,textarea,label,summary")) burst(e.clientX, e.clientY); });
    addEventListener("resize", resize);
    document.addEventListener("visibilitychange", () => { cancelAnimationFrame(raf); if (!document.hidden) frame(); });
    resize(); frame();
  }


  /* ---------- Per-theme animated scenes (see theme-scenes.js) ---------- */
  function initSceneFx() {
    const t = DATA.theme, SC = window.MLS_SCENES;
    if (!t || t === "data" || t === "weather" || reduce || !SC || !SC[t]) return;
    const bg = $("#fx-theme-bg"); if (!bg) return;
    const cv = el("canvas"); cv.id = "fx-theme-canvas"; bg.after(cv);
    const ctx = cv.getContext("2d"), dpr = Math.min(devicePixelRatio || 1, 2), rgb = (window.MLS_SCENE_RGB || {})[t] || "99,102,241";
    const bursts = [];
    const S = {
      ctx, W: 0, H: 0, t: 0, mouse: { x: -999, y: -999, vx: 0 }, col: a => `rgba(${rgb},${a})`,
      burst(x, y, n) { for (let i = 0; i < (n || 8); i++) { const a = Math.random() * 6.283, v = 1 + Math.random() * 2.5; bursts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 1, life: 1, r: 1.5 + Math.random() * 2 }); } },
    };
    let scene, raf, lastX = 0;
    function resize() {
      S.W = innerWidth; S.H = innerHeight; cv.width = S.W * dpr; cv.height = S.H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      scene = SC[t](S);
    }
    function frame(now) {
      S.t = now / 1000; S.mouse.vx *= .9;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, S.W, S.H); ctx.globalAlpha = .6;
      try { scene.draw(S.t); } catch (e) { /* a broken frame must never break the page */ }
      for (let i = bursts.length - 1; i >= 0; i--) {
        const b = bursts[i]; b.x += b.vx; b.y += b.vy; b.vy += .09; b.vx *= .98; b.life -= .02;
        if (b.life <= 0) { bursts.splice(i, 1); continue; }
        ctx.fillStyle = S.col(b.life * .8); ctx.beginPath(); ctx.arc(b.x, b.y, b.r * b.life + .5, 0, 6.283); ctx.fill();
      }
      ctx.globalAlpha = 1; raf = requestAnimationFrame(frame);
    }
    addEventListener("pointermove", e => { S.mouse.vx = e.clientX - lastX; lastX = e.clientX; S.mouse.x = e.clientX; S.mouse.y = e.clientY; }, { passive: true });
    root.addEventListener("pointerleave", () => { S.mouse.x = S.mouse.y = -999; });
    addEventListener("pointerdown", e => {
      if (e.target.closest("a,button,input,select,textarea,label,summary")) return;
      S.burst(e.clientX, e.clientY, 8); if (scene && scene.click) scene.click(e.clientX, e.clientY);
    });
    addEventListener("resize", resize);
    document.addEventListener("visibilitychange", () => { cancelAnimationFrame(raf); if (!document.hidden) raf = requestAnimationFrame(frame); });
    resize(); raf = requestAnimationFrame(frame);
  }

  /* ---------- Motion helpers: ripple, magnetic buttons, KPI tilt, stagger, back-to-top ---------- */
  function initMotion() {
    $$(".eda-nav-list").forEach(list => $$(".eda-nav-item", list).forEach((n, i) => n.style.setProperty("--i", i)));
    if (!reduce) {
      const RIPPLE = ".btn,.btn-primary,.btn-outline,.chip,.nf-chip,.ai-btn,.eda-nav-item,.acc-head";
      document.addEventListener("pointerdown", e => {
        const t = e.target.closest && e.target.closest(RIPPLE);
        if (!t || t.disabled) return;
        const r = t.getBoundingClientRect(), d = Math.max(r.width, r.height) * 2, ink = el("span", "fx-ripple");
        ink.style.cssText = `width:${d}px;height:${d}px;left:${e.clientX - r.left - d / 2}px;top:${e.clientY - r.top - d / 2}px`;
        if (getComputedStyle(t).position === "static") t.style.position = "relative";
        t.appendChild(ink); ink.addEventListener("animationend", () => ink.remove());
      }, { passive: true });
    }
    if (!reduce && !coarse) {
      $$(".btn-primary").forEach(b => {
        b.addEventListener("pointermove", e => { const r = b.getBoundingClientRect(); b.style.setProperty("--tx", ((e.clientX - r.left - r.width / 2) * .1).toFixed(1) + "px"); b.style.setProperty("--ty", ((e.clientY - r.top - r.height / 2) * .2).toFixed(1) + "px"); });
        b.addEventListener("pointerleave", () => { b.style.removeProperty("--tx"); b.style.removeProperty("--ty"); });
      });
      $$(".kpi-card").forEach(c => {
        c.addEventListener("pointermove", e => { const r = c.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5; c.style.transition = "transform .08s"; c.style.transform = `perspective(800px) rotateX(${(-y * 5).toFixed(2)}deg) rotateY(${(x * 6).toFixed(2)}deg) translateY(-3px)`; });
        c.addEventListener("pointerleave", () => { c.style.transition = ""; c.style.transform = ""; });
      });
    }
    if (!$("#fx-top")) {
      const up = el("button", null, ICON.up); up.id = "fx-top"; up.type = "button"; up.setAttribute("aria-label", "Haut de page");
      up.addEventListener("click", () => scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" }));
      document.body.appendChild(up);
      const upd = () => up.classList.toggle("show", scrollY > 600); addEventListener("scroll", upd, { passive: true }); upd();
    }
  }

  /* ---------- Boot ---------- */
  function boot() {
    initTheme(); initCanvas(); initThemeFx(); initSceneFx();
    initProgress(); initKeys();
    initTables(); initGallery(); initLightboxNav();
    initGlow(); initTilt(); initReveal(); initCounters(); initTransitions(); initMotion();
    // accordion items reveal their content on open -> hook glow/tables for late nodes
    document.addEventListener("click", e => { if (e.target.closest(".acc-head")) setTimeout(() => { initGlow(); initTables(); }, 30); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
