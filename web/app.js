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

  /** Absolute API origin when the frontend is hosted apart from the backend. */
  const API_BASE = (window.KNEEGUARD_API_BASE || "").replace(/\/$/, "");

  const apiUrl = (path) => API_BASE + path;

  async function api(path, options) {
    const response = await fetch(apiUrl(path), options);
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

  /* ── Camera ─────────────────────────────────────────────────────── */

  const RECORD_SECONDS = 3;
  const COUNTDOWN_FROM = 3;

  const camera = {
    stream: null,
    recorder: null,
    facing: "environment",
    captured: null, // { blob, filename }
    timers: [],
  };

  function clearCameraTimers() {
    camera.timers.forEach((id) => clearTimeout(id));
    camera.timers.forEach((id) => clearInterval(id));
    camera.timers = [];
  }

  function showCameraError(message) {
    const node = $("camera-error");
    node.hidden = false;
    node.textContent = message;
  }

  function hideCameraError() {
    $("camera-error").hidden = true;
  }

  /** getUserMedia only exists in a secure context (HTTPS or localhost). */
  function cameraSupported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  function insecureContextMessage() {
    return (
      `The browser hides the camera on insecure origins, so it is unavailable at ` +
      `${location.origin}. Open the app at http://localhost:${location.port || 80}, ` +
      `or serve it over HTTPS (./run.sh --https) to use a phone on the same network. ` +
      `You can still use the Upload tab.`
    );
  }

  function setCameraButtons(stateName) {
    const live = stateName === "live";
    const review = stateName === "review";
    $("camera-start").hidden = stateName !== "idle";
    $("camera-record").hidden = !live;
    $("camera-photo").hidden = !live;
    $("camera-flip").hidden = !live;
    $("camera-stop").hidden = !live;
    $("camera-retake").hidden = !review;
    $("camera-use").hidden = !review;
    $("camera-stage").dataset.state = stateName;
  }

  function stopStream() {
    if (camera.stream) {
      camera.stream.getTracks().forEach((track) => track.stop());
      camera.stream = null;
    }
  }

  async function startCamera() {
    hideCameraError();

    if (!cameraSupported()) {
      showCameraError(
        window.isSecureContext === false
          ? insecureContextMessage()
          : "This browser does not expose a camera API. Use the Upload tab instead."
      );
      return;
    }

    try {
      stopStream();
      camera.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: camera.facing,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
    } catch (error) {
      const name = error && error.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        showCameraError(
          "Camera permission was denied. Allow camera access for this site in your " +
          "browser settings, then press Start camera again — or use the Upload tab."
        );
      } else if (name === "NotFoundError" || name === "OverconstrainedError") {
        showCameraError("No camera found on this device. Use the Upload tab instead.");
      } else if (name === "NotReadableError") {
        showCameraError(
          "The camera is already in use by another app. Close it and try again."
        );
      } else {
        showCameraError(`Could not start the camera: ${error && error.message}`);
      }
      return;
    }

    const video = $("camera-video");
    video.srcObject = camera.stream;
    video.hidden = false;
    $("camera-playback").hidden = true;
    $("camera-still").hidden = true;
    $("camera-placeholder").hidden = true;
    await video.play().catch(() => { /* autoplay policies — preview still binds */ });
    setCameraButtons("live");
  }

  function turnOffCamera() {
    clearCameraTimers();
    stopStream();
    const video = $("camera-video");
    video.srcObject = null;
    video.hidden = true;
    $("camera-playback").hidden = true;
    $("camera-still").hidden = true;
    $("camera-placeholder").hidden = false;
    $("camera-countdown").hidden = true;
    $("camera-recording").hidden = true;
    setCameraButtons("idle");
  }

  async function flipCamera() {
    camera.facing = camera.facing === "environment" ? "user" : "environment";
    await startCamera();
  }

  function pickRecorderMime() {
    const candidates = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
      "video/mp4",
    ];
    if (typeof MediaRecorder === "undefined") return null;
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || null;
  }

  function countdown(from) {
    return new Promise((resolve) => {
      const node = $("camera-countdown");
      node.hidden = false;
      let value = from;
      node.textContent = String(value);

      const tick = setInterval(() => {
        value -= 1;
        if (value <= 0) {
          clearInterval(tick);
          node.textContent = "GO";
          const done = setTimeout(() => {
            node.hidden = true;
            resolve();
          }, 400);
          camera.timers.push(done);
        } else {
          node.textContent = String(value);
        }
      }, 700);
      camera.timers.push(tick);
    });
  }

  function showCapturedVideo(blob, filename) {
    camera.captured = { blob, filename };
    const playback = $("camera-playback");
    playback.src = URL.createObjectURL(blob);
    playback.hidden = false;
    $("camera-video").hidden = true;
    $("camera-still").hidden = true;
    setCameraButtons("review");
  }

  async function recordClip() {
    const mime = pickRecorderMime();
    if (!mime) {
      showCameraError(
        "This browser cannot record video. Use Take photo, or the Upload tab."
      );
      return;
    }

    $("camera-record").disabled = true;
    $("camera-photo").disabled = true;
    await countdown(COUNTDOWN_FROM);

    const chunks = [];
    let recorder;
    try {
      recorder = new MediaRecorder(camera.stream, { mimeType: mime });
    } catch (error) {
      showCameraError(`Could not start recording: ${error.message}`);
      $("camera-record").disabled = false;
      $("camera-photo").disabled = false;
      return;
    }
    camera.recorder = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size) chunks.push(event.data);
    };

    recorder.onstop = () => {
      $("camera-recording").hidden = true;
      $("camera-record").disabled = false;
      $("camera-photo").disabled = false;
      camera.recorder = null;
      clearCameraTimers();

      const blob = new Blob(chunks, { type: mime });
      if (!blob.size) {
        showCameraError("Recording came back empty. Try again.");
        setCameraButtons("live");
        return;
      }
      // Extension must match the mime so the server routes it as video.
      showCapturedVideo(blob, mime.startsWith("video/mp4") ? "capture.mp4" : "capture.webm");
    };

    recorder.start();

    const badge = $("camera-recording");
    badge.hidden = false;
    const startedAt = Date.now();
    const ticker = setInterval(() => {
      const elapsed = (Date.now() - startedAt) / 1000;
      $("rec-timer").textContent = `REC ${Math.min(elapsed, RECORD_SECONDS).toFixed(1)}s`;
    }, 100);
    camera.timers.push(ticker);

    const autostop = setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, RECORD_SECONDS * 1000);
    camera.timers.push(autostop);
  }

  function takePhoto() {
    const video = $("camera-video");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          showCameraError("Could not capture a frame. Try again.");
          return;
        }
        camera.captured = { blob, filename: "capture.jpg" };
        const still = $("camera-still");
        still.src = URL.createObjectURL(blob);
        still.hidden = false;
        $("camera-video").hidden = true;
        $("camera-playback").hidden = true;
        setCameraButtons("review");
      },
      "image/jpeg",
      0.92
    );
  }

  function retake() {
    camera.captured = null;
    const playback = $("camera-playback");
    if (playback.src) URL.revokeObjectURL(playback.src);
    playback.removeAttribute("src");
    playback.hidden = true;

    const still = $("camera-still");
    if (still.src) URL.revokeObjectURL(still.src);
    still.removeAttribute("src");
    still.hidden = true;

    $("camera-video").hidden = false;
    setCameraButtons("live");
  }

  async function submitCapture() {
    if (!camera.captured) return;
    const { blob, filename } = camera.captured;
    const file = new File([blob], filename, { type: blob.type });
    $("camera-use").disabled = true;
    try {
      await uploadFile(file);
    } finally {
      $("camera-use").disabled = false;
    }
  }

  function bindCamera() {
    $("camera-start").addEventListener("click", startCamera);
    $("camera-stop").addEventListener("click", turnOffCamera);
    $("camera-flip").addEventListener("click", flipCamera);
    $("camera-record").addEventListener("click", recordClip);
    $("camera-photo").addEventListener("click", takePhoto);
    $("camera-retake").addEventListener("click", retake);
    $("camera-use").addEventListener("click", submitCapture);

    // Surface the insecure-origin case up front rather than on first click.
    if (!cameraSupported()) {
      showCameraError(
        window.isSecureContext === false
          ? insecureContextMessage()
          : "This browser does not expose a camera API. Use the Upload tab instead."
      );
      $("camera-start").disabled = true;
    }

    // Never leave the camera running in the background.
    window.addEventListener("pagehide", turnOffCamera);
  }

  function bindTabs() {
    const tabs = Array.from(document.querySelectorAll(".tab"));
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((other) => {
          const selected = other === tab;
          other.setAttribute("aria-selected", String(selected));
          $(`pane-${other.dataset.pane}`).hidden = !selected;
        });
        if (tab.dataset.pane !== "camera") turnOffCamera();
      });
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

      // Signed in -> save it. Signed out -> hold it, so signing in from the
      // results screen still captures the scorecard just produced.
      accounts.lastReport = report;
      if (accountsReady() && accounts.user) {
        saveAssessment(report);
      } else {
        updateSaveHint();
      }

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

  /* ── Accounts (Supabase) ────────────────────────────────────────── */

  const accounts = {
    client: null,
    user: null,
    mode: "signin", // "signin" | "signup"
    lastReport: null,
  };

  let toastTimer = null;

  /** Brief confirmation banner — auth state changes are otherwise invisible. */
  function showToast(message, kind) {
    const node = $("toast");
    node.replaceChildren();
    node.appendChild(el("span", "toast-icon", kind === "in" ? "\u25CF" : "\u25CB"));
    node.appendChild(document.createTextNode(message));
    node.dataset.kind = kind || "in";
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, 4000);
  }

  function accountsReady() {
    return Boolean(accounts.client);
  }

  function initAccounts(cfg) {
    if (!cfg || !cfg.enabled) return; // Not configured — feature stays hidden.

    if (!window.supabase || typeof window.supabase.createClient !== "function") {
      // CDN blocked or offline. Everything else still works, so say nothing
      // louder than a console note.
      console.warn("Supabase library unavailable — accounts disabled.");
      return;
    }

    accounts.client = window.supabase.createClient(cfg.url, cfg.anon_key);
    $("account-area").hidden = false;

    accounts.client.auth.getSession().then(({ data }) => {
      setUser(data && data.session ? data.session.user : null);
    });
    accounts.client.auth.onAuthStateChange((event, session) => {
      setUser(session ? session.user : null);
      // INITIAL_SESSION fires on every page load; only announce real changes.
      if (event === "SIGNED_IN") {
        showToast(`Signed in as ${session.user.email}`, "in");
      } else if (event === "SIGNED_OUT") {
        showToast("Signed out", "out");
      }
    });
  }

  function setUser(user) {
    accounts.user = user;
    const signedIn = Boolean(user);

    $("account-email").textContent = signedIn ? `Signed in · ${user.email}` : "";
    $("account-email").hidden = !signedIn;
    $("account-signin").hidden = signedIn;
    $("account-signout").hidden = !signedIn;
    $("account-history").hidden = !signedIn;

    // Someone who signs in after running an assessment should not lose it.
    if (signedIn && accounts.lastReport) saveAssessment(accounts.lastReport);
    updateSaveHint();
  }

  function updateSaveHint(message) {
    const hint = $("save-hint");
    if (!hint) return;
    if (message) {
      hint.textContent = message;
      hint.hidden = false;
      return;
    }
    if (!accountsReady()) { hint.hidden = true; return; }
    hint.hidden = false;
    hint.textContent = accounts.user
      ? "Saving to your history…"
      : "Sign in to save this scorecard to your history.";
  }

  /** Flatten a report into one assessments row. Media is never included. */
  function assessmentRow(report) {
    const inputs = report.workload.inputs;
    const byLigament = {};
    report.ligaments.forEach((l) => { byLigament[l.ligament] = l.risk_index; });

    return {
      user_id: accounts.user.id,
      age: inputs.age,
      sex: inputs.sex,
      minutes_last_7d: inputs.minutes_last_7d,
      minutes_prior_28d: inputs.minutes_prior_28d,
      consecutive_days: inputs.consecutive_days,
      surface: inputs.surface,
      soreness: inputs.soreness,
      sleep_hours: inputs.sleep_hours,
      prior_injury: inputs.prior_injury,
      contact_events: inputs.contact_events,
      slide_tackles: inputs.slide_tackles,
      acwr: report.workload.acwr,
      overall_index: report.overall.risk_index,
      overall_band: report.overall.band,
      primary_ligament: report.overall.primary_ligament,
      acl_index: byLigament.ACL,
      mcl_index: byLigament.MCL,
      pcl_index: byLigament.PCL,
      ligaments: report.ligaments,
      action_plan: report.action_plan,
      scan: report.scan, // numeric summary only — no image data
      explanation: report.explanation,
    };
  }

  async function saveAssessment(report) {
    if (!accountsReady() || !accounts.user) return;
    try {
      const { error } = await accounts.client
        .from("assessments")
        .insert(assessmentRow(report));
      if (error) throw new Error(error.message);
      accounts.lastReport = null;
      updateSaveHint("Saved to your history.");
    } catch (error) {
      updateSaveHint(`Could not save: ${error.message}`);
    }
  }

  function openModal(id) {
    $(id).hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeModal(id) {
    $(id).hidden = true;
    document.body.style.overflow = "";
  }

  function setAuthMode(mode) {
    accounts.mode = mode;
    const signup = mode === "signup";
    $("auth-title").textContent = signup ? "Create an account" : "Sign in";
    $("auth-submit").textContent = signup ? "Create account" : "Sign in";
    $("auth-toggle").textContent = signup
      ? "I already have an account"
      : "Create an account instead";
    $("auth-password").setAttribute(
      "autocomplete", signup ? "new-password" : "current-password"
    );
    // Deliberately does not touch #auth-error: this is called *after* setting
    // messages like "account created", and clearing here hid them instantly.
  }

  function setAuthNotice(message) {
    const node = $("auth-error");
    node.hidden = !message;
    node.textContent = message || "";
  }

  async function submitAuth(event) {
    event.preventDefault();
    const email = $("auth-email").value.trim();
    const password = $("auth-password").value;
    const button = $("auth-submit");

    setAuthNotice("");
    button.disabled = true;
    button.textContent = accounts.mode === "signup" ? "Creating…" : "Signing in…";

    try {
      const auth = accounts.client.auth;
      const { data, error: authError } =
        accounts.mode === "signup"
          ? await auth.signUp({ email, password })
          : await auth.signInWithPassword({ email, password });

      if (authError) throw new Error(authError.message);

      // With email confirmation switched on, signUp returns a user but no
      // session — say so rather than looking like nothing happened.
      if (accounts.mode === "signup" && data && data.user && !data.session) {
        setAuthMode("signin");
        setAuthNotice(
          "Account created. Confirm it from the email we sent, then sign in."
        );
        return;
      }
      closeModal("auth-dialog");
      $("auth-form").reset();
    } catch (exc) {
      setAuthNotice(exc.message);
    } finally {
      button.disabled = false;
      button.textContent = accounts.mode === "signup" ? "Create account" : "Sign in";
    }
  }

  function historyCard(row) {
    const card = el("div", "history-item");

    const head = el("div", "history-head");
    const when = new Date(row.created_at);
    head.appendChild(
      el("span", "history-date", when.toLocaleString(undefined, {
        dateStyle: "medium", timeStyle: "short",
      }))
    );
    head.appendChild(badge(bandStatus(row.overall_band)));
    card.appendChild(head);

    const scores = el("div", "history-scores");
    [["Overall", row.overall_index], ["ACL", row.acl_index],
     ["MCL", row.mcl_index], ["PCL", row.pcl_index]].forEach(([label, value]) => {
      const cell = el("div", "history-score");
      cell.appendChild(el("span", "history-score-label", label));
      cell.appendChild(el("span", "history-score-value",
        value === null || value === undefined ? "—" : String(Math.round(value))));
      scores.appendChild(cell);
    });
    card.appendChild(scores);

    const meta = [];
    if (row.acwr !== null && row.acwr !== undefined) meta.push(`ACWR ${Number(row.acwr).toFixed(2)}`);
    if (row.surface) meta.push(row.surface.replace("_", " "));
    if (row.soreness !== null && row.soreness !== undefined) meta.push(`soreness ${row.soreness}/10`);
    if (row.scan && row.scan.frames_analysed) {
      meta.push(row.scan.frontal_view
        ? `valgus ${row.scan.peak_valgus_deg}°`
        : "scan: not front-on");
    }
    if (meta.length) card.appendChild(el("p", "history-meta", meta.join(" · ")));

    const remove = el("button", "btn btn-ghost history-delete", "Delete");
    remove.type = "button";
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      const { error } = await accounts.client
        .from("assessments").delete().eq("id", row.id);
      if (error) {
        remove.disabled = false;
        remove.textContent = "Delete failed";
        return;
      }
      card.remove();
    });
    card.appendChild(remove);

    return card;
  }

  /** Map a stored band label back to a status role for the badge. */
  function bandStatus(band) {
    return { low: "good", moderate: "warning", high: "serious", critical: "critical" }[band]
      || "good";
  }

  async function openHistory() {
    openModal("history-dialog");
    const list = $("history-list");
    list.replaceChildren(el("p", "history-empty", "Loading…"));

    const { data, error } = await accounts.client
      .from("assessments")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(50);

    if (error) {
      list.replaceChildren(el("p", "history-empty", `Could not load history: ${error.message}`));
      return;
    }
    if (!data || !data.length) {
      list.replaceChildren(
        el("p", "history-empty", "Nothing saved yet. Run an assessment while signed in.")
      );
      return;
    }
    $("history-sub").textContent = `${data.length} saved · newest first`;
    list.replaceChildren(...data.map(historyCard));
  }

  function bindAccounts() {
    $("account-signin").addEventListener("click", () => {
      setAuthMode("signin");
      setAuthNotice("");
      openModal("auth-dialog");
      $("auth-email").focus();
    });
    $("account-signout").addEventListener("click", async () => {
      await accounts.client.auth.signOut();
    });
    $("account-history").addEventListener("click", openHistory);

    $("auth-close").addEventListener("click", () => closeModal("auth-dialog"));
    $("history-close").addEventListener("click", () => closeModal("history-dialog"));
    $("auth-toggle").addEventListener("click", () => {
      setAuthMode(accounts.mode === "signup" ? "signin" : "signup");
      setAuthNotice("");
    });
    $("auth-form").addEventListener("submit", submitAuth);

    // Click the backdrop or press Escape to dismiss.
    ["auth-dialog", "history-dialog"].forEach((id) => {
      $(id).addEventListener("click", (event) => {
        if (event.target === $(id)) closeModal(id);
      });
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      ["auth-dialog", "history-dialog"].forEach((id) => {
        if (!$(id).hidden) closeModal(id);
      });
    });
  }

  /* ── Boot ───────────────────────────────────────────────────────── */

  async function boot() {
    initTheme();
    bindForm();
    bindScanner();
    bindTabs();
    bindCamera();
    bindAccounts();
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
      initAccounts(reference.supabase);

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
