// SmartGate front-end: polls the server for the live state and renders it.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ tabs
let currentTab = "gate";
document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab-page").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
    currentTab = b.dataset.tab;
    if (currentTab === "residents") loadResidents();
    if (currentTab === "log") loadLog();
  })
);

// ----------------------------------------------------------------- sound
let soundOn = true;
let audioCtx = null;
$("#btn-mute").addEventListener("click", () => {
  soundOn = !soundOn;
  $("#btn-mute").textContent = soundOn ? "🔊 Sound on" : "🔇 Sound off";
});
function tone(freqs, type, dur) {
  if (!soundOn) return;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    freqs.forEach((f, i) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = type; o.frequency.value = f;
      const t = audioCtx.currentTime + i * dur * 0.9;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g).connect(audioCtx.destination);
      o.start(t); o.stop(t + dur + 0.02);
    });
  } catch (e) { /* audio not available */ }
}
const chime = () => tone([660, 880, 1320], "sine", 0.18);
const buzz = () => tone([150, 150], "square", 0.28);
document.addEventListener("click", () => { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); }, { once: true });

// --------------------------------------------------------------- verdict
let lastVerdictKey = "";
function plateClass(type) {
  type = (type || "").toLowerCase();
  if (type.startsWith("yellow")) return "yellow";
  if (type.startsWith("green")) return "green";
  if (type.startsWith("black")) return "black";
  return "";
}
function renderVerdict(st) {
  const v = st.verdict, r = st.readings || {};
  const box = $("#verdict"), plate = $("#v-plate"), visitorBtn = $("#btn-visitor");
  box.className = "card verdict";
  plate.classList.add("hidden");
  visitorBtn.classList.add("hidden");
  $("#v-details").textContent = "";

  if (v) {
    box.classList.add(v.status);
    plate.className = "plate-big " + plateClass(v.plate_type);
    plate.textContent = v.formatted;
    const details = [v.plate_type, `${v.time}`, v.source === "upload" ? "from uploaded photo" : "from camera"].filter(Boolean);
    $("#v-details").textContent = details.join(" · ");
    if (v.status === "granted") {
      $("#v-icon").textContent = "✔";
      $("#v-title").textContent = "ACCESS GRANTED";
      $("#v-sub").innerHTML = `<b>${esc(v.owner)}</b> · Flat <b>${esc(v.flat)}</b>`;
    } else if (v.status === "denied") {
      $("#v-icon").textContent = "✖";
      $("#v-title").textContent = "ACCESS DENIED";
      $("#v-sub").textContent = "Vehicle not registered with the society.";
      visitorBtn.classList.remove("hidden");
    } else if (v.status === "visitor") {
      $("#v-icon").textContent = "➜";
      $("#v-title").textContent = "VISITOR ENTRY";
      $("#v-sub").textContent = "Allowed once by the guard. Logged as a visitor.";
    }
    const key = v.plate + v.status + v.time;
    if (key !== lastVerdictKey) {
      lastVerdictKey = key;
      if (v.status === "granted" || v.status === "visitor") chime(); else buzz();
    }
  } else if (r.chars && (r.corrected || r.skipped)) {
    box.classList.add("reading");
    $("#v-icon").textContent = "◠";
    $("#v-title").textContent = "Reading plate…";
    $("#v-sub").textContent = r.skipped ? r.skipped : (r.formatted || r.corrected);
  } else if (st.paused) {
    box.classList.add("notfound");
    $("#v-icon").textContent = "?";
    $("#v-title").textContent = "No readable plate";
    $("#v-sub").textContent = "No number plate could be read in that photo.";
  } else {
    box.classList.add("idle");
    $("#v-icon").textContent = "◌";
    $("#v-title").textContent = "Waiting for vehicle";
    $("#v-sub").textContent = "Hold a number plate up to the camera.";
  }
  plate.classList.toggle("hidden", !v);
}

function renderReadings(st) {
  const r = st.readings || {};
  $("#r-easy").textContent = r.easy ? `${r.easy}  (${Math.round((r.easy_conf || 0) * 100)}% sure)` :
    (st.reader === "loading" ? "loading model…" : st.reader === "unavailable" ? "not installed" : "–");
  $("#r-tpl").textContent = r.template || "–";
  $("#r-corr").textContent = r.corrected ?
    `${r.formatted}${r.valid ? "" : " (not a valid plate)"}${r.fixes ? `  · ${r.fixes} fix${r.fixes > 1 ? "es" : ""}` : ""}` : "–";
  $("#r-votes").textContent = st.paused ? "single photo (no voting)" :
    (r.votes ? (r.votes >= r.votes_needed ? `confirmed (${r.votes} matching reads)` :
      `${r.votes} of ${r.votes_needed} matching reads needed`) : "–");
  let hint = "";
  if (r.used) hint = `Decision based on ${r.used}.`;
  if (r.skipped) hint = r.skipped + ".";
  else if (r.chars && !r.corrected) hint = `Plate found (${r.chars} characters) but it couldn't be read yet.`;
  $("#r-hint").textContent = hint;
}

// ----------------------------------------------------------------- steps
let stepsVersion = -1;
function renderSteps(steps) {
  if ($("#bts-freeze").checked || !steps) return;
  const filter = $("#bts-filter").value;
  const wrap = $("#steps");
  if (!steps.length) { wrap.innerHTML = '<p class="empty">Waiting for the first frame…</p>'; return; }
  const shown = steps.filter((s) => !filter || s.unit.includes(filter));
  const keys = shown.map((s) => s.key).join(",");
  if (wrap.dataset.keys !== keys) {
    wrap.dataset.keys = keys;
    wrap.innerHTML = shown.map((s) => `
      <div class="step ${s.key === "template" ? "wide" : ""}" data-key="${s.key}">
        <img alt="${esc(s.title)}">
        <div class="step-body">
          <div class="step-title"><span>${esc(s.title)}</span><span class="unit">${esc(s.unit)}</span></div>
          <div class="step-text"></div>
        </div>
      </div>`).join("") || '<p class="empty">No steps for that unit in this frame.</p>';
  }
  shown.forEach((s) => {
    const el = wrap.querySelector(`[data-key="${s.key}"]`);
    if (!el) return;
    if (s.img) el.querySelector("img").src = "data:image/jpeg;base64," + s.img;
    el.querySelector(".step-text").textContent = s.text;
  });
}
$("#bts-filter").addEventListener("change", () => { $("#steps").dataset.keys = ""; stepsVersion = -1; });
$("#steps").addEventListener("click", (e) => {
  if (e.target.tagName !== "IMG") return;
  const z = document.createElement("div");
  z.className = "zoom";
  z.innerHTML = `<img src="${e.target.src}">`;
  z.addEventListener("click", () => z.remove());
  document.body.appendChild(z);
});

// ------------------------------------------------------------------ poll
async function poll() {
  try {
    const st = await (await fetch(`/api/state?since=${stepsVersion}`)).json();
    const cam = $("#pill-camera");
    cam.textContent = st.camera.ok ? `Camera ${st.camera.index} live` : "Camera off";
    cam.className = "pill " + (st.camera.ok ? "ok" : "bad");
    const rd = $("#pill-reader");
    rd.textContent = { ready: "OCR ready", loading: "OCR loading…", unavailable: "OCR unavailable" }[st.reader] || st.reader;
    rd.className = "pill " + (st.reader === "ready" ? "ok" : st.reader === "unavailable" ? "bad" : "");
    $("#camera-msg").classList.toggle("hidden", st.camera.ok);
    $("#camera-msg").textContent = st.camera.message;
    if (document.activeElement !== $("#camera-select")) $("#camera-select").value = st.camera.index;
    $("#paused-banner").classList.toggle("hidden", !st.paused);
    $("#meta").textContent = st.paused ? "" : `${st.fps} analyses/s · motion ${Math.round(st.motion * 100)}%`;
    renderVerdict(st);
    renderReadings(st);
    if (st.steps) { renderSteps(st.steps); stepsVersion = st.steps_version; }
  } catch (e) { /* server restarting */ }
  setTimeout(poll, 300);
}
poll();

// -------------------------------------------------------------- controls
$("#camera-select").addEventListener("change", async (e) => {
  await fetch("/api/camera", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index: +e.target.value }) });
  setTimeout(() => { $("#video").src = "/video_feed?" + Date.now(); }, 1500);
});
$("#upload").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("image", f);
  $("#v-title").textContent = "Analysing photo…";
  const res = await (await fetch("/api/analyze", { method: "POST", body: fd })).json();
  if (!res.ok) alert(res.error);
  stepsVersion = -1;
  e.target.value = "";
});
$("#btn-resume").addEventListener("click", () => fetch("/api/resume", { method: "POST" }));
$("#btn-visitor").addEventListener("click", () => fetch("/api/visitor", { method: "POST" }));

// ------------------------------------------------------------- residents
let residents = [];
async function loadResidents() {
  residents = await (await fetch("/api/vehicles")).json();
  renderResidents();
}
function renderResidents() {
  const q = $("#res-search").value.trim().toLowerCase().replace(/\s+/g, "");
  const rows = residents.filter((r) => !q ||
    (r.plate + r.owner + r.flat_no).toLowerCase().replace(/\s+/g, "").includes(q));
  const flats = new Set(residents.map((r) => r.flat_no)).size;
  $("#res-count").textContent = residents.length ? `· ${residents.length} vehicle${residents.length > 1 ? "s" : ""}, ${flats} flat${flats > 1 ? "s" : ""}` : "";
  $("#res-body").innerHTML = rows.map((r) => `
    <tr><td><b>${esc(r.flat_no)}</b></td><td>${esc(r.owner)}</td><td class="mono">${esc(r.formatted)}</td>
    <td class="muted">${esc(r.added_at.replace("T", " "))}</td>
    <td><button class="del" data-plate="${esc(r.plate)}">Remove</button></td></tr>`).join("");
  $("#res-empty").classList.toggle("hidden", residents.length > 0);
}
$("#res-search").addEventListener("input", renderResidents);
$("#res-body").addEventListener("click", async (e) => {
  const p = e.target.dataset.plate;
  if (!p || !confirm(`Remove ${p} from the society?`)) return;
  await fetch("/api/vehicles/" + encodeURIComponent(p), { method: "DELETE" });
  loadResidents();
});
$("#add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const res = await (await fetch("/api/vehicles", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.fromEntries(fd)) })).json();
  const msg = $("#add-msg");
  if (res.ok) {
    msg.className = "form-msg " + (res.warning ? "warn" : "ok");
    msg.textContent = res.warning || `Added ${res.formatted}.`;
    e.target.reset();
    e.target.plate.focus();
    loadResidents();
  } else {
    msg.className = "form-msg err";
    msg.textContent = res.error;
  }
});

// ------------------------------------------------------------------- log
async function loadLog() {
  const rows = await (await fetch("/api/log")).json();
  $("#log-count").textContent = rows.length ? `· ${rows.length}` : "";
  $("#log-body").innerHTML = rows.map((r) => {
    const visitor = (r.source || "").startsWith("visitor");
    const cls = visitor ? "visitor" : r.allowed ? "granted" : "denied";
    const label = visitor ? "Visitor" : r.allowed ? "Granted" : "Denied";
    return `<tr><td class="muted">${esc(r.time.replace("T", " "))}</td><td class="mono">${esc(r.formatted)}</td>
      <td><span class="tag ${cls}">${label}</span></td><td>${esc(r.owner || "—")}</td><td>${esc(r.flat_no || "—")}</td>
      <td class="muted">${esc(r.plate_type || "")}</td><td class="muted">${esc(r.source || "")}</td></tr>`;
  }).join("");
  $("#log-empty").classList.toggle("hidden", rows.length > 0);
}
$("#btn-clear-log").addEventListener("click", async () => {
  if (!confirm("Clear the whole entry log?")) return;
  await fetch("/api/log", { method: "DELETE" });
  loadLog();
});
setInterval(() => { if (currentTab === "log") loadLog(); }, 3000);
