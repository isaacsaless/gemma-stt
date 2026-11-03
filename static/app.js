"use strict";

const $ = (s) => document.querySelector(s);
const state = { jobId: null, words: [], source: "", audioUrl: null };

/* ---------- file picking / dropping ---------- */
const dropCard = $("#picker"), fileInput = $("#file");
$("#pick").onclick = () => fileInput.click();
dropCard.querySelector(".drop-hint").onclick = () => fileInput.click();
dropCard.addEventListener("dragover", (e) => { e.preventDefault(); dropCard.classList.add("drag"); });
dropCard.addEventListener("dragleave", () => dropCard.classList.remove("drag"));
dropCard.addEventListener("drop", (e) => {
  e.preventDefault(); dropCard.classList.remove("drag");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.onchange = () => fileInput.files.length && uploadFile(fileInput.files[0]);

$("#go").onclick = () => {
  const p = $("#pathInput").value.trim();
  if (!p) return;
  setLoading(p);
  fetch("/api/transcribe", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: p }),
  }).then(r => r.json().then(j => [r.ok, j])).then(([ok, j]) => {
    if (!ok) throw new Error(j.detail || "erro");
    watch(j.id, p);
  }).catch(err => showError(String(err)));
};

async function uploadFile(f) {
  setLoading(f.name);
  const fd = new FormData();
  fd.append("file", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  const j = await r.json();
  watch(j.id, f.name);
}

/* ---------- progress (SSE) ---------- */
const progressCard = $("#progressCard");
const msgs = { "áudio carregado": "áudio" };

function setLoading(name) {
  $("#picker").classList.add("hidden");
  progressCard.classList.remove("hidden");
  $("#pfile").textContent = name;
  $("#pbar").style.width = "4%";
  $("#pstate").textContent = "…";
  $("#pmsg").textContent = "carregando";
}

function watch(id, name) {
  state.jobId = id;
  const es = new EventSource(`/api/job/${id}/events`);
  let ticks = 0;
  es.onerror = () => { es.close(); poll(id, name); };
  es.addEventListener("progress", (e) => {
    const msg = e.data || "";
    $("#pmsg").textContent = msg;
    ticks++;
    $("#pbar").style.width = Math.min(4 + ticks * 6, 97) + "%";
  });
  es.addEventListener("error", (e) => { es.close(); setError(e.data); });
  es.addEventListener("end", async () => {
    es.close();
    const r = await fetch(`/api/job/${id}`).then(r => r.json());
    if (r.status === "done") finish(r.result, name);
    else setError(r.error);
  });
  poll(id, name);
}

async function poll(id, name) {
  for (;;) {
    const j = await fetch(`/api/job/${id}`).then(r => r.json());
    if (j.error) return setError(j.error);
    if (j.message) $("#pmsg").textContent = j.message;
    if (j.status === "done") { if (j.result) finish(j.result, name); return; }
    await new Promise(res => setTimeout(res, 700));
  }
}

function setError(msg) {
  $("#pstate").textContent = "erro"; $("#pbar").style.width = "0%";
  $("#pmsg").textContent = msg || "falhou";
  $("#pmsg").style.color = "var(--err)";
  $("#picker").classList.remove("hidden");
}

/* ---------- transcript ---------- */
function finish(result, name) {
  state.words = result.words;
  $("#progressCard").classList.add("hidden");
  $("#pmsg").style.color = "";
  const t = $("#transcriptCard");
  t.classList.remove("hidden");
  $("#searchCard").classList.remove("hidden");
  $("#ttitle").textContent = name;
  $("#tstats").textContent =
    `${result.duration.toFixed(0)} s de áudio · ${result.n_words} palavras · ${result.realtime_x}× tempo real · ` +
    (result.language ? `${result.language} · ${result.engine || ""}` : result.engine || "");
  $("#player").src = `/api/job/${state.jobId}/audio`;
  $("#transcript").innerHTML = "";
  renderTranscript(result.words, "");
}

function renderTranscript(words, highlight) {
  const out = $("#transcript");
  const frag = document.createDocumentFragment();
  const re = highlight ? new RegExp(`(${highlight.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "i") : null;
  for (let i = 0; i < words.length; i += 24) {
    const ln = document.createElement("span"); ln.className = "ln";
    const slice = words.slice(i, i + 24);
    for (let k = 0; k < slice.length; k++) {
      const w = slice[k];
      const span = document.createElement("span");
      span.className = "w"; span.dataset.i = i + k; span.textContent = w.w;
      if (re && re.test(w.w)) span.classList.add("hl");
      ln.appendChild(span);
      ln.appendChild(document.createTextNode(" "));
    }
    frag.appendChild(ln);
    if (i + 24 < words.length) frag.append("\n");
  }
  out.replaceChildren(frag);
  out.onclick = (e) => {
    const t = e.target.closest(".w");
    if (t) seek(parseFloat(words[+t.dataset.i].start));
  };
}

function seek(t) { const p = $("#player"); p.currentTime = t; p.play(); }

/* ---------- search ---------- */
$("#searchBtn").onclick = search;
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });

async function search() {
  const q = $("#q").value.trim();
  const box = $("#results");
  if (!state.jobId || !q) { box.innerHTML = ""; return; }
  box.innerHTML = `<div class="noresults">buscando…</div>`;
  const r = await fetch("/api/search", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: state.jobId, q }),
  }).then(r => r.json());
  if (!r.hits || !r.hits.length) { box.innerHTML = `<div class="noresults">nada encontrado</div>`; return; }
  const fmt = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const queryTerms = q.toLowerCase().split(/\s+/).filter(w => w.length > 3);
  box.innerHTML = r.hits.map(h => {
    let text = h.text;
    for (const term of queryTerms) text = text.replace(new RegExp(`(${term})`, "g"), "<mark>$1</mark>");
    return `<div class="hit" data-start="${h.start}">
      <div class="meta"><span class="sc">${h.score.toFixed(3)}</span> · ${fmt(h.start)} – ${fmt(h.end)}</div>
      <div class="text">${text}</div>
    </div>`;
  }).join("");
  box.querySelectorAll(".hit").forEach(el => el.onclick = () => seek(parseFloat(el.dataset.start)));
}

/* ---------- misc ---------- */
$("#btnCopy").onclick = () => navigator.clipboard.writeText(state.words.map(w => w.w).join(" "))
  .then(() => { $("#btnCopy").textContent = "copiado!"; setTimeout(() => { $("#btnCopy").textContent = "copiar"; }, 1200); });

window.__state = state;
