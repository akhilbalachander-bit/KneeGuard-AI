/* KneeGuard AI dashboard.
   Talks to the FastAPI backend: /api/reference, /api/scan, /api/assess. */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const state = {
    scanId: null,
    scan: null,
    bands: [],
    surfaces: [],
  };

  // Status roles map to the reserved status palette. The icon is what carries
  // the state for anyone who cannot rely on the colour.
  const STATUS = {
    good: { colour: "var(--status-good)", icon: "✓", label: "Low" },
    warning: { colour: "var(--status-warning)", icon: "!", label: "Moderate" },
    serious: { colour: "var(--status-serious)", icon: "▲", label: "High" },
    critical: { colour: "var(--status-critical)", icon: "■", label: "Critical" },
  };

  const statusOf = (role) => STATUS[role] || STATUS.good;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const body = await response.json();
        if (body && body.detail) {
          detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
      } catch (_) { /* non-JSON error body — keep the status message */ }
      throw new Error(detail);
    }
    return response.json();
  }

  /* ── Theme ──────────────────────────────────────────────────────── */

  function initTheme() {
    const stored = localStorage.getItem("kneeguard-theme");
    const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
    const theme = stored || (prefersLight ? "light" : "dark");
    document.documentElement.dataset.theme = theme;
    $("theme-toggle").textContent = theme === "dark" ? "☾" : "☀";

    $("theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("kneeguard-theme", next);
      $("theme-toggle").textContent = next === "dark" ? "☾" : "☀";
    });
  }

  /* ── Form ───────────────────────────────────────────────────────── */

  function readForm() {
    const form = $("workload-form");
    const data = new FormData(form);
    return {
      age: Number(data.get("age")),
      sex: data.get("sex"),
      minutes_last_7d: Number(data.get("minutes_last_7d")),
      minutes_prior_28d: Number(data.get("minutes_prior_28d")),
      consecutive_days: Number(data.get("consecutive_days")),
      surface: data.get("surface"),
      soreness: Number(data.get("soreness")),
      sleep_hours: Number(data.get("sleep_hours")),
      prior_injury: data.get("prior_injury") === "on",
      contact_events: Number(data.get("contact_events")),
      slide_tackles: Number(data.get("slide_tackles")),
      scan_id: state.scanId,
      explain: true,
    };
  }

  function updateAcwr() {
    const acute = Number($("workload-form").minutes_last_7d.value) || 0;
    const chronic = Number($("workload-form").minutes_prior_28d.value) || 0;
    const ratio = acute / Math.max(chronic / 4, 30);

    $("acwr-value").textContent = ratio.toFixed(2);
    const note = $("acwr-note");
    if (ratio > 1.3) {
      note.textContent = "Above the 1.3 safe ceiling — this is a load spike.";
      note.dataset.state = "over";
    } else {
      note.textContent = "Within the 1.3 safe ceiling.";
      note.dataset.state = "ok";
    }
  }

  function bindForm() {
    $("soreness").addEventListener("input", (e) => {
      $("soreness-out").textContent = e.target.value;
    });
    $("sleep_hours").addEventListener("input", (e) => {
      $("sleep-out").textContent = Number(e.target.value).toFixed(1);
    });
    ["minutes_last_7d", "minutes_prior_28d"].forEach((name) => {
      $("workload-form")[name].addEventListener("input", updateAcwr);
    });
    updateAcwr();
  }

  /* ── Scan ───────────────────────────────────────────────────────── */

  function setScanStatus(message, isError) {
    const node = $("scan-status");
    node.textContent = message;
    node.dataset.state = isError ? "error" : "ok";
  }

  function renderScan(payload) {
    state.scanId = payload.scan_id;
    state.scan = payload.scan;

    const scan = payload.scan;
    const box = $("scan-result");
    box.hidden = false;
    box.replaceChildren();

    if (payload.overlay_image) {
      const img = el("img", "scan-overlay");
      img.src = payload.overlay_image;
      img.alt =
        "Uploaded frame with the detected hip, knee and ankle skeleton drawn on it. " +
        "The yellow line is where the knee should track; the coloured lines are where it did.";
      box.appendChild(img);
    }

    const metrics = el("div", "scan-metrics");
    const cards = [
      ["Peak knee valgus", scan.frontal_view ? `${scan.peak_valgus_deg}°` : "n/a"],
      ["Landing flexion", scan.landing_assessed ? `${scan.landing_flexion_deg}°` : "n/a"],
      ["Knee : ankle ratio", scan.kasr !== null ? scan.kasr.toFixed(2) : "n/a"],
      ["Frames analysed", String(scan.frames_analysed)],
    ];
    cards.forEach(([label, value]) => {
      const card = el("div", "scan-metric");
      card.appendChild(el("span", "scan-metric-label", label));
      card.appendChild(el("span", "scan-metric-value", value));
      metrics.appendChild(card);
    });
    box.appendChild(metrics);

    if (scan.notes && scan.notes.length) {
      const list = el("ul", "scan-notes");
      scan.notes.forEach((note) => list.appendChild(el("li", null, note)));
      box.appendChild(list);
    }

    setScanStatus(
      `Scan ready — ${scan.frames_analysed} frame${scan.frames_analysed === 1 ? "" : "s"} analysed. ` +
      "Press Assess risk to fuse it with the workload model.",
      false
    );
  }

  async function uploadFile(file) {
    if (!file) return;
    document.querySelectorAll(".btn-demo").forEach((b) => b.classList.remove("is-active"));
    setScanStatus(`Analysing ${file.name}…`, false);
    const body = new FormData();
    body.append("file", file);
    try {
      renderScan(await api("/api/scan", { method: "POST", body }));
    } catch (error) {
      setScanStatus(error.message, true);
      $("scan-result").hidden = true;
      state.scanId = null;
      state.scan = null;
    }
  }

  function bindScanner() {
    const zone = $("dropzone");
    const input = $("file-input");

    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => uploadFile(input.files[0]));

    ["dragenter", "dragover"].forEach((name) =>
      zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.add("is-over"); })
    );
    ["dragleave", "drop"].forEach((name) =>
      zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.remove("is-over"); })
    );
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
  }

  async function loadDemo(key, button) {
    document.querySelectorAll(".btn-demo").forEach((b) => b.classList.remove("is-active"));
    button.classList.add("is-active");
    setScanStatus("Loading synthetic scan…", false);
    try {
      renderScan(await api(`/api/scan/demo/${key}`, { method: "POST" }));
    } catch (error) {
      setScanStatus(error.message, true);
    }
  }

  /* ── Results ────────────────────────────────────────────────────── */

  function renderHero(overall) {
    const status = statusOf(overall.status);
    $("hero-value").textContent = Math.round(overall.risk_index);

    const fill = $("hero-meter-fill");
    fill.style.width = `${Math.max(2, overall.risk_index)}%`;
    fill.style.backgroundColor = status.colour;

    $("hero-meter-wrap").setAttribute(
      "aria-label",
      `Overall risk index ${Math.round(overall.risk_index)} out of 100 — ${overall.band} band.`
    );

    const ticks = $("meter-ticks");
    ticks.replaceChildren();
    state.bands.forEach((band) => {
      if (band.min === 0) return;
      const tick = el("span", "meter-tick");
      tick.appendChild(el("b", "tick-num", String(band.min)));
      tick.appendChild(el("i", "tick-label", band.label));
      tick.style.left = `${band.min}%`;
      ticks.appendChild(tick);
    });

    const caption = $("hero-caption");
    caption.replaceChildren();
    caption.append(
      document.createTextNode("Highest single-ligament risk is the "),
      el("strong", null, overall.primary_ligament),
      document.createTextNode(
        `, in the ${overall.band} band. The index is a percentile against the reference cohort, not a chance of injury.`
      )
    );
  }

  function badge(status) {
    const info = statusOf(status);
    const node = el("span", "badge");
    node.style.color = info.colour;
    node.appendChild(el("span", "badge-icon", info.icon));
    node.appendChild(document.createTextNode(info.label));
    return node;
  }

  function renderTiles(ligaments) {
    const container = $("ligament-tiles");
    container.replaceChildren();

    ligaments.forEach((lig) => {
      const status = statusOf(lig.status);
      const tile = el("div", "tile");

      const head = el("div", "tile-head");
      const nameBox = el("div");
      nameBox.appendChild(el("span", "tile-name", lig.ligament));
      nameBox.appendChild(el("span", "tile-full", lig.name));
      head.appendChild(nameBox);
      head.appendChild(badge(lig.status));
      tile.appendChild(head);

      const value = el("div", "tile-value", String(Math.round(lig.risk_index)));
      value.style.color = status.colour;
      tile.appendChild(value);

      const meter = el("div", "tile-meter");
      const fill = el("div", "tile-meter-fill");
      fill.style.width = `${Math.max(2, lig.risk_index)}%`;
      fill.style.backgroundColor = status.colour;
      meter.appendChild(fill);
      tile.appendChild(meter);

      const shift = lig.biomechanical_shift;
      tile.appendChild(
        el(
          "span",
          "tile-shift",
          shift === 0
            ? `Workload only: ${lig.workload_only_index}`
            : `Workload ${lig.workload_only_index} → ${lig.risk_index} (${shift > 0 ? "+" : ""}${shift} from the scan)`
        )
      );

      tile.appendChild(el("p", "tile-mech", lig.mechanism));

      if (lig.drivers && lig.drivers.length) {
        tile.appendChild(el("p", "tile-list-title", "Top workload drivers"));
        const list = el("ul", "tile-list");
        lig.drivers.slice(0, 3).forEach((d) => list.appendChild(el("li", null, d.description)));
        tile.appendChild(list);
      }

      if (lig.adjustments && lig.adjustments.length) {
        tile.appendChild(el("p", "tile-list-title", "Movement findings"));
        const list = el("ul", "tile-list");
        lig.adjustments.forEach((a) => list.appendChild(el("li", null, a.detail)));
        tile.appendChild(list);
      }

      container.appendChild(tile);
    });
  }

  function renderExplanation(explanation) {
    const card = $("explain-card");
    if (!explanation) { card.hidden = true; return; }
    card.hidden = false;

    $("explain-headline").textContent = explanation.headline;
    $("explain-why").textContent = explanation.why;
    $("explain-movement").textContent = explanation.movement_note;

    const source = $("explain-source");
    if (explanation.generated_by === "claude") {
      source.textContent = `Written by ${explanation.model}`;
      source.className = "pill pill-good";
    } else {
      source.textContent = "Rule-based (no Claude API key set)";
      source.className = "pill pill-muted";
      source.title = explanation.note || "";
    }
  }

  function renderPlan(plan) {
    const container = $("action-plan");
    container.replaceChildren();

    plan.forEach((exercise) => {
      const card = el("div", "plan-card");

      const head = el("div", "plan-head");
      head.appendChild(el("span", "plan-name", exercise.name));
      const tags = el("div", "plan-tags");
      exercise.ligaments.forEach((lig) => tags.appendChild(el("span", "plan-tag", lig)));
      head.appendChild(tags);
      card.appendChild(head);

      card.appendChild(el("p", "plan-target", exercise.target));
      card.appendChild(el("p", "plan-dose", exercise.dose));
      card.appendChild(el("p", "plan-why", exercise.rationale));

      container.appendChild(card);
    });
  }

  function renderDetail(report) {
    const body = $("detail-table").querySelector("tbody");
    body.replaceChildren();

    report.ligaments.forEach((lig) => {
      const row = el("tr");
      const header = el("th", null, lig.ligament);
      header.setAttribute("scope", "row");
      row.appendChild(header);
      [
        String(lig.risk_index),
        String(lig.workload_only_index),
        `${lig.biomechanical_shift > 0 ? "+" : ""}${lig.biomechanical_shift}`,
        `${(lig.modelled_probability * 100).toFixed(2)}%`,
      ].forEach((text) => row.appendChild(el("td", null, text)));

      const bandCell = el("td");
      bandCell.appendChild(badge(lig.status));
      row.appendChild(bandCell);
      body.appendChild(row);
    });

    const metrics = report.model_metrics;
    const lines = ["acl", "mcl", "pcl"].map((lig) => {
      const m = metrics[lig];
      return `${lig.toUpperCase()}: ROC AUC ${m.roc_auc.toFixed(3)}, Brier ${m.brier.toFixed(4)}, cohort event rate ${(m.event_rate * 100).toFixed(2)}%`;
    });
    lines.push(
      `Logistic regression per ligament, trained on ${metrics.n_train.toLocaleString()} synthetic athlete-weeks and held out on ${metrics.n_test.toLocaleString()}.`
    );
    $("model-metrics").replaceChildren(...lines.map((line) => el("div", null, line)));
  }

  async function assess() {
    const button = $("assess-btn");
    button.disabled = true;
    button.textContent = "Assessing…";

    try {
      const report = await api("/api/assess", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(readForm()),
      });

      $("results-empty").hidden = true;
      $("results").hidden = false;

      renderHero(report.overall);
      renderTiles(report.ligaments);
      renderExplanation(report.explanation);
      renderPlan(report.action_plan);
      renderDetail(report);
      $("disclaimer").textContent = report.disclaimer;

      $("results").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (error) {
      $("results-empty").hidden = false;
      $("results-empty").replaceChildren(el("p", null, `Could not assess: ${error.message}`));
      $("results").hidden = true;
    } finally {
      button.disabled = false;
      button.textContent = "Assess risk";
    }
  }

  /* ── Boot ───────────────────────────────────────────────────────── */

  async function boot() {
    initTheme();
    bindForm();
    bindScanner();
    $("assess-btn").addEventListener("click", assess);

    try {
      const health = await api("/api/health");
      const pill = $("health-pill");
      if (!health.pose_model_ready) {
        pill.textContent = "Scanner offline — pose model missing";
        pill.className = "pill pill-warn";
      } else if (health.claude_configured) {
        pill.textContent = `Ready · ${health.claude_model}`;
        pill.className = "pill pill-good";
      } else {
        pill.textContent = "Ready · rule-based explanations";
        pill.className = "pill pill-muted";
        pill.title = "Set ANTHROPIC_API_KEY to have Claude write the explanation.";
      }
    } catch (_) {
      $("health-pill").textContent = "Backend unreachable";
      $("health-pill").className = "pill pill-warn";
    }

    try {
      const reference = await api("/api/reference");
      state.bands = reference.bands;
      state.surfaces = reference.surfaces;

      const select = $("surface-select");
      select.replaceChildren();
      reference.surfaces.forEach((surface) => {
        const option = el("option", null, surface.label);
        option.value = surface.value;
        select.appendChild(option);
      });
      select.value = "turf_dry";

      const buttons = $("demo-buttons");
      buttons.replaceChildren();
      reference.demo_scenarios.forEach((scenario) => {
        const button = el("button", "btn btn-demo", scenario.label);
        button.type = "button";
        button.title = scenario.description;
        button.addEventListener("click", () => loadDemo(scenario.key, button));
        buttons.appendChild(button);
      });
    } catch (error) {
      setScanStatus(`Could not load reference data: ${error.message}`, true);
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
