/**
 * ML Pipeline Studio - Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  initPipelineCases();
  initSubNavPills();
  initPlotGallery();
  initLightbox();
  initOutlierInspector();
  initCorrelationColors();
  initTableSearch();
  initAnalysisModal();
  initDataParadigms();
  initIndependentUpload();
  initStreamingSimulator();
  initDescToggles();
  initOutlierColToggles();
  initFeatureImportanceTabs();

});

/* ==========================================================================
   DESC SECTION TOGGLES (Case 1: Analyse Descriptive)
   ========================================================================== */
function initDescToggles() {
  const toggleBtns = document.querySelectorAll(".desc-toggle");
  if (!toggleBtns.length) return;

  toggleBtns.forEach(btn => {
    updateBtnStyle(btn, btn.classList.contains("active"));
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const section = document.getElementById(targetId);
      if (!section) return;
      const isActive = btn.classList.toggle("active");
      section.style.display = isActive ? "" : "none";
      updateBtnStyle(btn, isActive);
    });
  });

  function updateBtnStyle(btn, active) {
    btn.style.opacity = active ? "1" : "0.45";
    btn.style.borderColor = active ? "var(--accent-sky)" : "";
  }
}

/* ==========================================================================
   OUTLIER COLUMN TOGGLES (Case 3: Outliers — per-column cards)
   ========================================================================== */
function initOutlierColToggles() {
  const toggleBtns = document.querySelectorAll(".outlier-col-toggle");
  const showAllBtn = document.getElementById("outlier-show-all");
  const hideAllBtn = document.getElementById("outlier-hide-all");
  if (!toggleBtns.length) return;

  function getCard(key) {
    return document.querySelector(`.outlier-col-card[data-col="${key}"]`);
  }

  function setCardVisible(key, visible) {
    const card = getCard(key);
    if (card) {
      card.style.display = visible ? "" : "none";
    }
  }

  function updateToggleBtnStyle(btn, active) {
    btn.classList.toggle("active", active);
    btn.style.opacity = active ? "1" : "0.4";
    btn.style.borderColor = active ? "var(--accent-sky)" : "";
  }

  // Initialize all visible
  toggleBtns.forEach(btn => {
    updateToggleBtnStyle(btn, true);

    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-col");
      const isActive = !btn.classList.contains("active");
      updateToggleBtnStyle(btn, isActive);
      setCardVisible(key, isActive);
    });
  });

  if (showAllBtn) {
    showAllBtn.addEventListener("click", () => {
      toggleBtns.forEach(btn => {
        updateToggleBtnStyle(btn, true);
        setCardVisible(btn.getAttribute("data-col"), true);
      });
    });
  }

  if (hideAllBtn) {
    hideAllBtn.addEventListener("click", () => {
      toggleBtns.forEach(btn => {
        updateToggleBtnStyle(btn, false);
        setCardVisible(btn.getAttribute("data-col"), false);
      });
    });
  }
}

/* ==========================================================================
   FEATURE IMPORTANCE TABS (Case 5: Variance / Corrélation avec Cible)
   ========================================================================== */
function initFeatureImportanceTabs() {
  const tabVariance = document.getElementById("fi-tab-variance");
  const tabCorrelation = document.getElementById("fi-tab-correlation");
  const panelVariance = document.getElementById("fi-panel-variance");
  const panelCorrelation = document.getElementById("fi-panel-correlation");
  if (!tabVariance || !tabCorrelation) return;

  function activateTab(activeTab, activePanel, inactiveTab, inactivePanel) {
    activeTab.style.borderBottomColor = activeTab === tabVariance ? "#38bdf8" : "#2563eb";
    activeTab.style.color = activeTab === tabVariance ? "#38bdf8" : "#2563eb";
    activeTab.style.fontWeight = "700";
    activeTab.classList.add("active");

    inactiveTab.style.borderBottomColor = "transparent";
    inactiveTab.style.color = "var(--text-muted)";
    inactiveTab.style.fontWeight = "600";
    inactiveTab.classList.remove("active");

    activePanel.style.display = "";
    inactivePanel.style.display = "none";
  }

  tabVariance.addEventListener("click", () => {
    activateTab(tabVariance, panelVariance, tabCorrelation, panelCorrelation);
  });

  tabCorrelation.addEventListener("click", () => {
    activateTab(tabCorrelation, panelCorrelation, tabVariance, panelVariance);
  });

  // --- Correlation Target Selector ---
  const select = document.getElementById("corr-target-select");
  const btn = document.getElementById("corr-target-btn");
  const btnIcon = document.getElementById("corr-btn-icon");
  const btnText = document.getElementById("corr-btn-text");
  const btnSpinner = document.getElementById("corr-btn-spinner");
  const container = document.getElementById("corr-results-container");
  if (!select || !btn || !container) return;

  select.addEventListener("change", () => {
    btn.disabled = !select.value;
  });

  btn.addEventListener("click", async () => {
    const target = select.value;
    if (!target) return;
    const dataset = select.getAttribute("data-dataset");

    // Loading state
    btn.disabled = true;
    if (btnIcon) btnIcon.style.display = "none";
    if (btnSpinner) btnSpinner.style.display = "inline-block";
    if (btnText) btnText.textContent = "Calcul en cours...";

    try {
      const response = await fetch(`/api/correlation-target/${encodeURIComponent(dataset)}/${encodeURIComponent(target)}`);
      const data = await response.json();

      if (!response.ok || data.status !== "ok") {
        throw new Error(data.message || "Erreur lors du calcul.");
      }

      renderCorrelationResults(container, data.target, data.features);
    } catch (err) {
      container.innerHTML = `
        <div style="background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.3); border-radius:var(--radius-lg); padding:1.25rem; color:#f87171; text-align:center;">
          <strong>Erreur :</strong> ${err.message}
        </div>`;
    } finally {
      btn.disabled = false;
      if (btnIcon) btnIcon.style.display = "inline";
      if (btnSpinner) btnSpinner.style.display = "none";
      if (btnText) btnText.textContent = "Calculer les Corrélations";
    }
  });
}

function renderCorrelationResults(container, target, features) {
  if (!features.length) {
    container.innerHTML = `
      <div style="text-align:center; padding:2rem; color:var(--text-muted);">
        <div style="font-size:2rem;">📭</div>
        <p>Aucune feature trouvée pour calculer la corrélation avec « ${target} ».</p>
      </div>`;
    return;
  }

  // Medal helper
  function medal(rank) {
    if (rank === 1) return '<span style="font-size:1rem;">🥇</span>';
    if (rank === 2) return '<span style="font-size:1rem;">🥈</span>';
    if (rank === 3) return '<span style="font-size:1rem;">🥉</span>';
    return `<span style="color:var(--text-muted); font-weight:600;">#${rank}</span>`;
  }

  // Strength badge
  function badge(f) {
    let bg, border;
    if (f.corr_abs >= 0.7) { bg = "rgba(52,211,153,.15)"; border = "rgba(52,211,153,.3)"; }
    else if (f.corr_abs >= 0.4) { bg = "rgba(251,191,36,.15)"; border = "rgba(251,191,36,.3)"; }
    else if (f.corr_abs >= 0.2) { bg = "rgba(251,146,60,.15)"; border = "rgba(251,146,60,.3)"; }
    else { bg = "rgba(248,113,113,.15)"; border = "rgba(248,113,113,.3)"; }
    return `<span class="badge" style="background:${bg}; color:${f.color}; border:1px solid ${border}; font-size:0.73rem;">${f.strength}</span>`;
  }

  // Table rows
  const tableRows = features.map((f, idx) => `
    <tr class="corr-item-stagger" style="animation-delay:${idx * 30}ms;">
      <td style="text-align:center;">${medal(f.rank)}</td>
      <td><strong style="color:#fff;">${f.name}</strong></td>
      <td style="color:#2563eb; font-weight:600;">${f.corr_abs}</td>
      <td>
        <div style="display:flex; align-items:center; gap:0.5rem;">
          <div style="flex:1; background:rgba(167,139,250,.12); border-radius:9999px; height:8px; overflow:hidden;">
            <div class="fi-bar-fill" style="height:100%; width:${f.corr_pct}%; border-radius:9999px; background:linear-gradient(90deg,${f.color},${f.color}aa); transition:width 0.6s cubic-bezier(0.16,1,0.3,1);"></div>
          </div>
          <span style="color:${f.color}; font-weight:700; min-width:42px;">${f.corr_pct}%</span>
        </div>
      </td>
      <td>${badge(f)}</td>
    </tr>`).join("");

  // Cards
  const cards = features.map((f, idx) => `
    <div class="corr-card-interactive corr-item-stagger" style="animation-delay:${idx * 45}ms; background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:var(--radius-lg); padding:1.1rem; cursor:default;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem;">
        <div>
          <span style="font-size:0.68rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Rang #${f.rank}</span>
          <h4 style="font-family:var(--font-heading); font-size:0.92rem; color:#fff; margin:0.1rem 0 0;">${f.name}</h4>
        </div>
        <span style="font-size:1.25rem; font-weight:800; color:${f.color};">${f.corr_pct}%</span>
      </div>
      <div style="margin-bottom:0.5rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:var(--text-muted); margin-bottom:0.25rem;">
          <span>🔗 |r| avec « ${target} »</span>
          <strong style="color:${f.color};">${f.corr_abs}</strong>
        </div>
        <div style="background:rgba(167,139,250,.1); border-radius:9999px; height:10px; overflow:hidden;">
          <div class="fi-bar-fill" style="height:100%; width:${f.corr_pct}%; border-radius:9999px; background:linear-gradient(90deg,${f.color},${f.color}aa); transition:width 0.7s cubic-bezier(0.16,1,0.3,1);"></div>
        </div>
      </div>
      <div style="font-size:0.72rem; color:var(--text-muted); background:rgba(15,23,42,0.04); border-radius:6px; padding:0.4rem 0.6rem; display:flex; justify-content:space-between; align-items:center;">
        <span>Force de la liaison linéaire</span>
        ${badge(f)}
      </div>
    </div>`).join("");

  container.innerHTML = `
    <!-- Résumé -->
    <div class="corr-item-stagger" style="background:rgba(167,139,250,.08); border:1px solid rgba(167,139,250,.25); border-radius:var(--radius-lg); padding:1rem 1.25rem; display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
      <span style="font-size:1.5rem;">🎯</span>
      <div>
        <strong style="color:#2563eb; font-size:0.95rem;">Cible sélectionnée : ${target}</strong>
        <div style="color:var(--text-secondary); font-size:0.82rem; margin-top:0.2rem;">
          ${features.length} features analysées • Feature la plus corrélée : <strong style="color:#fff;">${features[0].name}</strong> (|r| = ${features[0].corr_abs})
        </div>
      </div>
    </div>

    <!-- Tableau -->
    <div class="section-card" style="margin-top:1.25rem;">
      <div class="section-title">
        <h3>📊 Corrélation de Pearson |r| — Features vs « ${target} »</h3>
        <p class="section-desc">Classement des features par intensité de corrélation linéaire avec la variable cible.</p>
      </div>
      <div style="overflow-x:auto; margin-top:1rem;">
        <table class="data-table" style="width:100%; font-size:0.8rem;">
          <thead>
            <tr>
              <th style="width:40px;">#</th>
              <th>Feature</th>
              <th>|r| Pearson</th>
              <th style="min-width:160px;">Intensité</th>
              <th>Force</th>
            </tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    </div>

    <!-- Cartes visuelles -->
    <div class="section-card" style="margin-top:1.25rem;">
      <div class="section-title">
        <h3>📈 Cartes de Corrélation — Chaque Feature vs « ${target} »</h3>
        <p class="section-desc">Intensité visuelle de la liaison linéaire Pearson entre chaque feature et la cible.</p>
      </div>
      <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap:1rem; margin-top:1rem;">
        ${cards}
      </div>
    </div>`;
}

/* ==========================================================================
   1. Pipeline Cases Navigation (Stepper)
   ========================================================================== */
function initPipelineCases() {
  const caseCards = document.querySelectorAll(".pipeline-case-card");
  const casePanels = document.querySelectorAll(".pipeline-case-panel");

  function activateCase(caseId) {
    if (!caseId) return;

    // Update case card buttons
    caseCards.forEach(card => {
      const target = card.getAttribute("data-pipeline-case");
      if (target === caseId) {
        card.classList.add("active");
      } else {
        card.classList.remove("active");
      }
    });

    // Update panels
    casePanels.forEach(panel => {
      if (panel.id === caseId) {
        panel.classList.add("active");
      } else {
        panel.classList.remove("active");
      }
    });

    // Update URL hash without jumping
    if (history.replaceState) {
      history.replaceState(null, null, "#" + caseId);
    }
  }

  caseCards.forEach(card => {
    card.addEventListener("click", () => {
      const targetId = card.getAttribute("data-pipeline-case");
      activateCase(targetId);

      // Smooth scroll into panel view if needed
      const panel = document.getElementById(targetId);
      if (panel) {
        panel.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  // Restore state from URL hash if valid
  const currentHash = window.location.hash.replace("#", "");
  if (currentHash && document.getElementById(currentHash)) {
    activateCase(currentHash);
  }
}

/* ==========================================================================
   2. Sub-Navigation Pills (Inside Case 1: Analyse Descriptive)
   ========================================================================== */
function initSubNavPills() {
  const pillBtns = document.querySelectorAll(".sub-pill-btn");
  const subPanels = document.querySelectorAll(".sub-case-panel");

  pillBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-sub");

      pillBtns.forEach(b => b.classList.remove("active"));
      subPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });
}

/* ==========================================================================
   3. Plot Gallery & Filtering (Inside Case 2: Bruit & Outliers)
   ========================================================================== */
function initPlotGallery() {
  const filterBtns = document.querySelectorAll(".filter-btn");
  const plotCards = document.querySelectorAll(".plot-card");
  const searchInput = document.getElementById("plotSearchInput");

  function applyFilters() {
    const activeFilter = document.querySelector(".filter-btn.active")?.getAttribute("data-filter") || "all";
    const query = (searchInput?.value || "").toLowerCase().trim();

    plotCards.forEach(card => {
      const category = card.getAttribute("data-category");
      const title = (card.getAttribute("data-title") || "").toLowerCase();
      const feature = (card.getAttribute("data-feature") || "").toLowerCase();

      const matchesCategory = (activeFilter === "all") || (category === activeFilter);
      const matchesSearch = !query || title.includes(query) || feature.includes(query);

      if (matchesCategory && matchesSearch) {
        card.style.display = "flex";
      } else {
        card.style.display = "none";
      }
    });
  }

  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      applyFilters();
    });
  });

  if (searchInput) {
    searchInput.addEventListener("input", applyFilters);
  }
}

/* ==========================================================================
   4. Lightbox Modal
   ========================================================================== */
function initLightbox() {
  const modal = document.getElementById("lightboxModal");
  const modalImg = document.getElementById("lightboxImage");
  const modalTitle = document.getElementById("lightboxTitle");
  const closeBtn = document.getElementById("lightboxClose");
  const downloadBtn = document.getElementById("lightboxDownload");

  if (!modal) return;

  function openLightbox(url, title) {
    modalImg.src = url;
    modalTitle.textContent = title;
    if (downloadBtn) {
      downloadBtn.href = url;
      downloadBtn.download = title.replace(/\s+/g, "_") + ".png";
    }
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    modal.classList.remove("active");
    modalImg.src = "";
    document.body.style.overflow = "";
  }

  // Delegate click for plot preview triggers
  document.addEventListener("click", e => {
    const trigger = e.target.closest(".lightbox-trigger");
    if (trigger) {
      const url = trigger.getAttribute("data-url");
      const title = trigger.getAttribute("data-title") || "Chart Preview";
      openLightbox(url, title);
    }
  });

  closeBtn?.addEventListener("click", closeLightbox);

  modal.addEventListener("click", e => {
    if (e.target === modal) {
      closeLightbox();
    }
  });

  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      closeLightbox();
    }
  });
}

/* ==========================================================================
   5. Outlier Deep-Dive Inspector
   ========================================================================== */
function initOutlierInspector() {
  const select = document.getElementById("outlierColumnSelect");
  const container = document.getElementById("outlierRowsContainer");
  const metaText = document.getElementById("outlierMetaText");
  const datasetName = document.body.getAttribute("data-dataset");

  if (!select || !container) return;

  async function loadOutliers(columnKey) {
    container.innerHTML = `
      <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
        <span class="spinner"></span> Chargement des observations aberrantes...
      </div>
    `;

    try {
      const res = await fetch(`/api/outliers/${datasetName}/${columnKey}?limit=100`);
      const data = await res.json();

      if (data.error || !data.rows || data.rows.length === 0) {
        container.innerHTML = `
          <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
            Aucun enregistrement d'outlier trouvé pour cette variable.
          </div>
        `;
        if (metaText) metaText.textContent = "0 outlier détecté";
        return;
      }

      if (metaText) {
        metaText.textContent = `Affichage des ${data.rows.length} premiers sur ${data.total.toLocaleString()} outliers potentiels identifiés`;
      }

      let html = `
        <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>#</th>
                ${data.columns.map(col => `<th>${col}</th>`).join("")}
              </tr>
            </thead>
            <tbody>
              ${data.rows.map((row, idx) => `
                <tr>
                  <td style="color: var(--text-muted); font-weight: 600;">${idx + 1}</td>
                  ${data.columns.map(col => `
                    <td class="${col !== 'Formatted Date' ? 'td-highlight' : ''}">${row[col]}</td>
                  `).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;

      container.innerHTML = html;
    } catch (err) {
      console.error(err);
      container.innerHTML = `
        <div style="text-align: center; padding: 2rem; color: var(--accent-rose);">
          Erreur lors du chargement des enregistrements d'outliers.
        </div>
      `;
    }
  }

  select.addEventListener("change", () => {
    loadOutliers(select.value);
  });

  // Load first column by default
  if (select.value) {
    loadOutliers(select.value);
  }
}

/* ==========================================================================
   6. Dynamic Correlation Matrix Heatmap Colors
   ========================================================================== */
function initCorrelationColors() {
  const cells = document.querySelectorAll(".corr-cell");

  cells.forEach(cell => {
    const rawVal = cell.getAttribute("data-val");
    if (!rawVal || rawVal === "None" || rawVal === "—") {
      cell.style.background = "transparent";
      cell.textContent = "—";
      return;
    }

    const val = parseFloat(rawVal);
    if (isNaN(val)) return;

    // Colormap from cool blue (-1.0) to neutral (0.0) to warm crimson (+1.0)
    let bg = "";
    let textColor = "#f8fafc";

    if (val > 0) {
      // Positive correlation -> Orange/Rose gradient
      const intensity = Math.min(Math.abs(val), 1.0);
      const alpha = (intensity * 0.5 + 0.1).toFixed(2);
      bg = `rgba(244, 63, 94, ${alpha})`;
      if (intensity > 0.6) textColor = "#ffffff";
    } else if (val < 0) {
      // Negative correlation -> Cyan/Blue gradient
      const intensity = Math.min(Math.abs(val), 1.0);
      const alpha = (intensity * 0.5 + 0.1).toFixed(2);
      bg = `rgba(14, 165, 233, ${alpha})`;
      if (intensity > 0.6) textColor = "#ffffff";
    } else {
      bg = "rgba(15, 23, 42, 0.04)";
    }

    cell.style.backgroundColor = bg;
    cell.style.color = textColor;
  });
}

/* ==========================================================================
   7. Table Live Search Filtering
   ========================================================================== */
function initTableSearch() {
  const searchInputs = document.querySelectorAll("[data-table-search]");

  searchInputs.forEach(input => {
    const tableId = input.getAttribute("data-table-search");
    const table = document.getElementById(tableId);
    if (!table) return;

    input.addEventListener("input", () => {
      const query = input.value.toLowerCase().trim();
      const rows = table.querySelectorAll("tbody tr");

      rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? "" : "none";
      });
    });
  });
}

/* ==========================================================================
   8. Run / Upload Analysis Modal
   ========================================================================== */
function initAnalysisModal() {
  const openBtn = document.getElementById("openAnalysisModalBtn");
  const modal = document.getElementById("analysisModal");
  const closeBtn = document.getElementById("analysisModalClose");
  const cancelBtn = document.getElementById("analysisModalCancel");
  const form = document.getElementById("analysisForm");
  const submitBtn = document.getElementById("analysisSubmitBtn");
  const submitText = document.getElementById("analysisSubmitText");
  const spinner = document.getElementById("analysisSpinner");
  const statusMsg = document.getElementById("analysisStatusMsg");

  if (!modal) return;

  function openModal() {
    modal.classList.add("active");
  }

  function closeModal() {
    modal.classList.remove("active");
    if (statusMsg) {
      statusMsg.textContent = "";
      statusMsg.style.display = "none";
    }
  }

  openBtn?.addEventListener("click", openModal);
  closeBtn?.addEventListener("click", closeModal);
  cancelBtn?.addEventListener("click", closeModal);

  modal.addEventListener("click", e => {
    if (e.target === modal) closeModal();
  });

  if (form) {
    form.addEventListener("submit", async e => {
      e.preventDefault();

      const formData = new FormData(form);

      if (submitBtn) submitBtn.disabled = true;
      if (spinner) spinner.style.display = "inline-block";
      if (submitText) submitText.textContent = "Traitement en cours...";
      if (statusMsg) {
        statusMsg.style.display = "block";
        statusMsg.style.color = "var(--text-secondary)";
        statusMsg.textContent = "Calcul des statistiques descriptives, seuils IQR, corrélations et rendu des graphiques...";
      }

      try {
        const res = await fetch("/api/run-analysis", {
          method: "POST",
          body: formData
        });

        const data = await res.json();

        if (res.ok && data.status === "success") {
          statusMsg.style.color = "var(--accent-emerald)";
          statusMsg.textContent = data.message + " Redirection...";
          setTimeout(() => {
            window.location.href = data.redirect_url;
          }, 1200);
        } else {
          throw new Error(data.message || "L'analyse a échoué");
        }
      } catch (err) {
        console.error(err);
        if (statusMsg) {
          statusMsg.style.color = "var(--accent-rose)";
          statusMsg.textContent = "Erreur : " + err.message;
        }
        if (submitBtn) submitBtn.disabled = false;
        if (spinner) spinner.style.display = "none";
        if (submitText) submitText.textContent = "Lancer le Pipeline";
      }
    });
  }
}

/* ==========================================================================
   9. Data Paradigms Flow (Streaming, Structured, Unstructured)
   ========================================================================== */
function initDataParadigms() {
  const nodes = document.querySelectorAll(".paradigm-node");
  if (!nodes.length) return;

  nodes.forEach(node => {
    node.addEventListener("click", () => {
      const paradigm = node.getAttribute("data-paradigm");
      if (!paradigm) return;

      // Update active state on nodes
      nodes.forEach(n => {
        if (n === node) {
          n.classList.add("active");
          const circle = n.querySelector(".paradigm-circle");
          if (circle) circle.classList.add("active");
        } else {
          n.classList.remove("active");
          const circle = n.querySelector(".paradigm-circle");
          if (circle) circle.classList.remove("active");
        }
      });

      // Update active view
      const views = document.querySelectorAll(".paradigm-view");
      views.forEach(v => {
        v.classList.remove("active");
      });

      const targetView = document.getElementById(
        "view" + paradigm.charAt(0).toUpperCase() + paradigm.slice(1)
      );
      if (targetView) {
        targetView.classList.add("active");
      }
    });
  });

  // Handle URL hash if coming from mini-banner (#streaming, #unstructured)
  const hash = window.location.hash.toLowerCase();
  if (hash === "#streaming") {
    document.getElementById("nodeStreaming")?.click();
  } else if (hash === "#unstructured") {
    document.getElementById("nodeUnstructured")?.click();
  }
}

/* ==========================================================================
   10. Independent Upload Case ("Case Indépendante de Téléversement")
   ========================================================================== */
function initIndependentUpload() {
  const form = document.getElementById("independentAnalysisForm");
  const fileInput = document.getElementById("indepCsvFileInput");
  const dropzone = document.getElementById("independentDropzone");
  const nameInput = document.getElementById("indepDatasetName");
  const statusMsg = document.getElementById("indepStatusMsg");
  const submitBtn = document.getElementById("indepSubmitBtn");
  const submitText = document.getElementById("indepSubmitText");
  const submitIcon = document.getElementById("indepSubmitIcon");
  const spinner = document.getElementById("indepSpinner");
  const fileInfoPill = document.getElementById("indepFileInfoPill");
  const selectedName = document.getElementById("indepSelectedName");
  const selectedSize = document.getElementById("indepSelectedSize");
  const fileHint = document.getElementById("indepFileHint");

  if (!form || !fileInput || !dropzone) return;

  function handleFileSelected(file) {
    if (!file) return;

    // Display file name & size
    const sizeStr = file.size > 1024 * 1024
      ? (file.size / (1024 * 1024)).toFixed(2) + " MB"
      : (file.size / 1024).toFixed(1) + " KB";

    if (selectedName) selectedName.textContent = file.name;
    if (selectedSize) selectedSize.textContent = sizeStr;
    if (fileInfoPill) fileInfoPill.style.display = "inline-flex";
    if (fileHint) fileHint.style.display = "none";

    // Auto populate dataset name if empty or default
    if (nameInput && (!nameInput.value || nameInput.value.trim() === "")) {
      const cleanName = file.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_");
      nameInput.value = cleanName;
    }
  }

  // File input change
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  // Drag and drop events
  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", e => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files[0]) {
      fileInput.files = dt.files;
      handleFileSelected(dt.files[0]);
    }
  });

  // Reset form handler
  form.addEventListener("reset", () => {
    if (fileInfoPill) fileInfoPill.style.display = "none";
    if (fileHint) fileHint.style.display = "block";
    if (statusMsg) {
      statusMsg.style.display = "none";
      statusMsg.textContent = "";
    }
  });

  // Submit handler
  form.addEventListener("submit", async e => {
    e.preventDefault();

    const formData = new FormData(form);

    if (submitBtn) submitBtn.disabled = true;
    if (spinner) spinner.style.display = "inline-block";
    if (submitIcon) submitIcon.style.display = "none";
    if (submitText) submitText.textContent = "Calcul du pipeline en cours...";
    if (statusMsg) {
      statusMsg.style.display = "block";
      statusMsg.style.color = "var(--text-secondary)";
      statusMsg.style.borderColor = "var(--accent-sky)";
      statusMsg.innerHTML = "⚡ <strong>Traitement actif :</strong> Ingestion des données, calcul des seuils IQR, corrélations de Pearson & génération des figures...";
    }

    try {
      const res = await fetch("/api/run-analysis", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (res.ok && data.status === "success") {
        if (statusMsg) {
          statusMsg.style.color = "var(--accent-emerald)";
          statusMsg.style.borderColor = "rgba(16, 185, 129, 0.4)";
          statusMsg.innerHTML = "✓ <strong>Succès :</strong> " + data.message + " Redirection vers le pipeline complet...";
        }
        setTimeout(() => {
          window.location.href = data.redirect_url;
        }, 900);
      } else {
        throw new Error(data.message || "L'exécution de l'analyse a échoué.");
      }
    } catch (err) {
      console.error(err);
      if (statusMsg) {
        statusMsg.style.color = "var(--accent-rose)";
        statusMsg.style.borderColor = "rgba(244, 63, 94, 0.4)";
        statusMsg.innerHTML = "⚠️ <strong>Erreur :</strong> " + err.message;
      }
      if (submitBtn) submitBtn.disabled = false;
      if (spinner) spinner.style.display = "none";
      if (submitIcon) submitIcon.style.display = "inline";
      if (submitText) submitText.textContent = "Lancer le Pipeline d'Analyse";
    }
  });
}

/* ==========================================================================
   11. Streaming Telemetry Simulator
   ========================================================================== */
function initStreamingSimulator() {
  const terminalLog = document.getElementById("streamTerminalLog");
  const toggleBtn = document.getElementById("toggleStreamBtn");
  const statusDot = document.getElementById("streamStatusDot");
  const toggleText = document.getElementById("streamToggleText");
  const throughputEl = document.getElementById("streamThroughput");
  const latencyEl = document.getElementById("streamLatency");

  if (!terminalLog || !toggleBtn) return;

  let isStreaming = true;
  let counter = 4;

  const timer = setInterval(() => {
    if (!isStreaming) return;

    counter++;
    const now = new Date();
    const timeStr = now.toTimeString().split(" ")[0];
    const sensorId = "0" + Math.floor(Math.random() * 9 + 1);
    const temp = (18 + Math.random() * 8).toFixed(1);
    const hum = Math.floor(55 + Math.random() * 20);
    const wind = Math.floor(10 + Math.random() * 25);

    const line = document.createElement("div");
    line.className = "term-line";
    line.innerHTML = `<span class="t-time">[${timeStr}]</span> <span class="t-topic">[sensor-${sensorId}]</span> Temp: ${temp}°C | Humidity: ${hum}% | Wind: ${wind} km/h | Status: <span class="t-ok">OK</span>`;

    terminalLog.appendChild(line);
    terminalLog.scrollTop = terminalLog.scrollHeight;

    // Keep log to max 30 items
    while (terminalLog.children.length > 30) {
      terminalLog.removeChild(terminalLog.firstChild);
    }

    // Dynamic metrics
    if (throughputEl) {
      const tp = 1400 + Math.floor(Math.random() * 160);
      throughputEl.innerHTML = tp.toLocaleString() + ' <small>ev/s</small>';
    }
    if (latencyEl) {
      const lat = (7.5 + Math.random() * 2.2).toFixed(1);
      latencyEl.innerHTML = lat + ' <small>ms</small>';
    }
  }, 1800);

  toggleBtn.addEventListener("click", () => {
    isStreaming = !isStreaming;
    if (isStreaming) {
      statusDot.classList.add("live");
      toggleText.textContent = "Pause Flux";
    } else {
      statusDot.classList.remove("live");
      toggleText.textContent = "Reprendre Flux";
    }
  });
}

/* ==========================================================================
   12. Independent Upload & Analysis Form (Page 1)
   ========================================================================== */
function initIndependentUpload() {
  const form = document.getElementById("independentAnalysisForm");
  const fileInput = document.getElementById("indepCsvFileInput");
  const dropzone = document.getElementById("independentDropzone");
  const datasetNameInput = document.getElementById("indepDatasetName");
  const statusMsg = document.getElementById("indepStatusMsg");
  const submitBtn = document.getElementById("indepSubmitBtn");
  const spinner = document.getElementById("indepSpinner");
  const submitIcon = document.getElementById("indepSubmitIcon");
  const submitText = document.getElementById("indepSubmitText");

  const fileInfoPill = document.getElementById("indepFileInfoPill");
  const selectedName = document.getElementById("indepSelectedName");
  const selectedSize = document.getElementById("indepSelectedSize");
  const fileNameDisplay = document.getElementById("indepFileNameDisplay");
  const fileHint = document.getElementById("indepFileHint");

  if (!form || !fileInput) return;

  function handleFileSelected(file) {
    if (!file) return;

    if (fileInfoPill && selectedName && selectedSize) {
      selectedName.textContent = file.name;
      const sizeKb = (file.size / 1024).toFixed(1);
      selectedSize.textContent = sizeKb > 1024 ? (sizeKb / 1024).toFixed(2) + " MB" : sizeKb + " KB";
      fileInfoPill.style.display = "inline-flex";
    }

    if (fileNameDisplay) {
      fileNameDisplay.textContent = file.name;
    }
    if (fileHint) {
      fileHint.textContent = "Fichier prêt pour l'analyse.";
    }

    // Auto-fill dataset name if empty
    if (datasetNameInput && !datasetNameInput.value.trim()) {
      const stem = file.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
      datasetNameInput.value = stem;
    }
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  if (dropzone) {
    ["dragenter", "dragover"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files[0]) {
        fileInput.files = dt.files;
        handleFileSelected(dt.files[0]);
      }
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    const hasFile = fileInput.files && fileInput.files.length > 0;
    const hasDatasetName = datasetNameInput && datasetNameInput.value.trim().length > 0;

    if (!hasFile && !hasDatasetName) {
      if (statusMsg) {
        statusMsg.style.display = "block";
        statusMsg.className = "upload-status-box error";
        statusMsg.textContent = "Veuillez sélectionner un fichier CSV ou spécifier un dataset existant.";
      }
      return;
    }

    // UI Loading state
    if (submitBtn) submitBtn.disabled = true;
    if (spinner) spinner.style.display = "inline-block";
    if (submitIcon) submitIcon.style.display = "none";
    if (submitText) submitText.textContent = "Exécution du Pipeline ML en cours...";
    if (statusMsg) {
      statusMsg.style.display = "block";
      statusMsg.className = "upload-status-box loading";
      statusMsg.textContent = "Traitement en cours : Analyse Descriptive, Bruit, Outliers, Distributions, Feature Importance...";
    }

    try {
      const response = await fetch("/api/run-analysis", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.status === "success") {
        if (statusMsg) {
          statusMsg.className = "upload-status-box success";
          statusMsg.textContent = "Pipeline terminé avec succès ! Redirection vers la Page 2 (Pipeline Studio)...";
        }
        setTimeout(() => {
          window.location.href = data.redirect_url;
        }, 800);
      } else {
        throw new Error(data.message || "Erreur lors du traitement du pipeline.");
      }
    } catch (err) {
      if (statusMsg) {
        statusMsg.className = "upload-status-box error";
        statusMsg.textContent = "Erreur : " + err.message;
      }
      if (submitBtn) submitBtn.disabled = false;
      if (spinner) spinner.style.display = "none";
      if (submitIcon) submitIcon.style.display = "inline-block";
      if (submitText) submitText.textContent = "Lancer l'Analyse & Ouvrir le Pipeline";
    }
  });
}




/* ==========================================================================
   OVERVIEW ACCORDION — items are hidden; clicking one shows its result.
   ========================================================================== */
document.addEventListener("DOMContentLoaded", () => {
  const items = document.querySelectorAll(".acc-item");
  if (!items.length) return;

  function setOpen(item, open) {
    item.classList.toggle("open", open);
    item.querySelector(".acc-head").setAttribute("aria-expanded", open ? "true" : "false");
    item.querySelector(".acc-body").hidden = !open;
  }

  items.forEach(item => {
    item.querySelector(".acc-head").addEventListener("click", () => setOpen(item, !item.classList.contains("open")));
  });
  document.getElementById("acc-open-all")?.addEventListener("click", () => items.forEach(i => setOpen(i, true)));
  document.getElementById("acc-close-all")?.addEventListener("click", () => items.forEach(i => setOpen(i, false)));
});
