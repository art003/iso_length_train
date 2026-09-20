const logEl = document.getElementById("log");
const rowsEl = document.getElementById("rows");
const thumbsEl = document.getElementById("thumbs");
const busy = document.getElementById("busy");
const runBtn = document.getElementById("run");
const stopBtn = document.getElementById("stop");
const jobsEl = document.getElementById("jobs");
const editorEl = document.getElementById("editor");
const edImg = document.getElementById("ed-img");
const edCanvas = document.getElementById("ed-canvas");
const edList = document.getElementById("ed-list");
const trainLogEl = document.getElementById("train-log");
const trainStatsEl = document.getElementById("train-stats");
const trainBtn = document.getElementById("btn-train");
const auditBtn = document.getElementById("btn-audit");
let poll = null;
let trainPoll = null;
let currentJob = "";
let lastSheets = [];
let lastJobs = [];
let editorView = null;
let editorBusy = false;
let reviewFilter = null;

document.querySelectorAll(".header__links a, .header__logo").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    showPage(a.dataset.page);
  });
});

function showPage(name) {
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("is-on", p.id === "page-" + name));
  document.querySelectorAll(".header__links a").forEach((a) => a.classList.toggle("is-on", a.dataset.page === name));
}

document.getElementById("pdf").addEventListener("change", () => {
  const f = document.getElementById("pdf").files[0];
  document.getElementById("pdf-name").textContent = f ? f.name : "файл не выбран — можно перетащить сюда";
  if (f) document.getElementById("use-default").checked = false;
});

const drop = document.getElementById("drop");
["dragenter", "dragover"].forEach((ev) => {
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("is-on");
  });
});
["dragleave", "drop"].forEach((ev) => {
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("is-on");
  });
});
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (!f || !f.name.toLowerCase().endsWith(".pdf")) return;
  const dt = new DataTransfer();
  dt.items.add(f);
  document.getElementById("pdf").files = dt.files;
  document.getElementById("pdf").dispatchEvent(new Event("change"));
});

function setRunning(on) {
  runBtn.disabled = on;
  stopBtn.disabled = !on;
  if (trainBtn) trainBtn.disabled = on;
  if (auditBtn) auditBtn.disabled = on;
  busy.classList.toggle("is-on", on);
}

function fillTrain(t) {
  if (!trainStatsEl || !t) return;
  const lines = [];
  lines.push(t.model_exists ? "<b>Модель есть:</b> models\\iso_clf.pkl" : "<b>Модели нет</b> — нажми «Обучить модель»");
  if (t.model_kind) lines.push("Модель: <b>" + t.model_kind + "</b> · " + (t.device || ""));
  if (t.validation) {
    lines.push(
      "Validation: macro-F1 <b>" +
        Number(t.validation.macro_f1 || 0).toFixed(3) +
        "</b>, accuracy <b>" +
        Number(t.validation.accuracy || 0).toFixed(3) +
        "</b>"
    );
  }
  if (t.train && t.train.macro_f1 != null) {
    lines.push("Train macro-F1: <b>" + Number(t.train.macro_f1 || 0).toFixed(3) + "</b>");
  }
  if (t.overfit_gap_macro_f1 != null) {
    lines.push(
      "Переобучение (зазор): <b>" +
        Number(t.overfit_gap_macro_f1).toFixed(3) +
        "</b> " +
        (t.overfit_ok ? "ok" : "высокий")
    );
  }
  if (t.holdout_check) {
    lines.push(
      t.holdout_check.ok
        ? "Holdout: <b>ok</b> — тест-листы 278–307 не в train"
        : "Holdout: <b>ошибка утечки</b>"
    );
  }
  if (t.training_sheets) {
    lines.push("Листов в train (без holdout 278–307): <b>" + t.training_sheets + "</b>");
  }
  if (t.audit) {
    lines.push("Аудит: " + (t.audit.sheets || 0) + " листов, проблем " + (t.audit.problems_count || 0));
  }
  trainStatsEl.innerHTML = lines.map((x) => "<div>" + x + "</div>").join("");
  const sm = document.getElementById("stat-model");
  if (sm) sm.textContent = t.model_exists ? "есть" : "нет";
}

async function startTrain(kind) {
  if (trainLogEl) trainLogEl.textContent = kind === "audit" ? "проверка...\n" : "обучение...\n";
  showPage("train");
  setRunning(true);
  const r = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind }),
  });
  const data = await r.json();
  if (!r.ok) {
    if (trainLogEl) trainLogEl.textContent += "ошибка: " + (data.error || r.status) + "\n";
    setRunning(false);
    return;
  }
  if (trainPoll) clearInterval(trainPoll);
  trainPoll = setInterval(tickTrain, 800);
}

async function tickTrain() {
  const r = await fetch("/api/status");
  const s = await r.json();
  if (trainLogEl) {
    trainLogEl.textContent = (s.log || []).join("\n") || "работаю...";
    trainLogEl.scrollTop = trainLogEl.scrollHeight;
  }
  fillTrain(s.train);
  if (!s.running) {
    clearInterval(trainPoll);
    trainPoll = null;
    setRunning(false);
    if (s.error && trainLogEl) trainLogEl.textContent += "\nошибка: " + s.error;
  }
}

if (trainBtn) trainBtn.addEventListener("click", () => startTrain("train"));
if (auditBtn) auditBtn.addEventListener("click", () => startTrain("audit"));

document.querySelectorAll(".review-set[data-range]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const [a, b] = btn.dataset.range.split("-").map(Number);
    reviewFilter = { start: a, end: b, title: btn.querySelector("b").textContent };
    const r = await fetch("/api/job/" + encodeURIComponent("Изометрии_без_API"));
    const data = await r.json();
    if (!r.ok) {
      alert(data.error || "не открылась разметка");
      return;
    }
    fillResults(data);
    showPage("results");
  });
});

document.querySelectorAll("[data-open]").forEach((btn) => {
  btn.addEventListener("click", () => {
    fetch("/api/open-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: btn.dataset.open }),
    });
  });
});

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  reviewFilter = null;
  const fd = new FormData();
  const file = document.getElementById("pdf").files[0];
  if (file) fd.append("pdf", file);
  fd.append("use_default", document.getElementById("use-default").checked ? "1" : "0");
  fd.append("api", "Локальная модель");
  fd.append("model", "local");
  fd.append("key", "");
  fd.append("pages", document.getElementById("pages").value);
  fd.append("replay", "0");
  fd.append("fresh", "1");
  setRunning(true);
  logEl.textContent = "запуск...\n";
  const r = await fetch("/api/start", { method: "POST", body: fd });
  const data = await r.json();
  if (!r.ok) {
    logEl.textContent += "ошибка: " + (data.error || r.status) + "\n";
    setRunning(false);
    return;
  }
  poll = setInterval(tick, 800);
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  await fetch("/api/stop", { method: "POST" });
});

document.getElementById("open-out").addEventListener("click", () => {
  const fd = new FormData();
  if (currentJob) fd.append("job", currentJob);
  fetch("/api/open-output", { method: "POST", body: fd });
});

jobsEl.addEventListener("change", async () => {
  reviewFilter = null;
  const id = jobsEl.value;
  if (!id) return;
  const r = await fetch("/api/job/" + encodeURIComponent(id));
  const data = await r.json();
  if (r.ok) fillResults(data);
});

async function tick() {
  const r = await fetch("/api/status");
  const s = await r.json();
  logEl.textContent = (s.log || []).join("\n") || "считаю...";
  logEl.scrollTop = logEl.scrollHeight;
  fillTrain(s.train);
  if (s.metrics) fillResults(s);
  if (!s.running) {
    clearInterval(poll);
    poll = null;
    setRunning(false);
    if (s.error) logEl.textContent += "\nошибка: " + s.error;
  }
}

function fillJobs(jobs) {
  lastJobs = jobs || [];
  const cur = currentJob;
  jobsEl.innerHTML = lastJobs
    .map((j) => {
      const label = j.pdf || j.id;
      const sum = j.total_m != null ? " · " + j.total_m + " м" : "";
      return `<option value="${j.id}">${label}${sum}</option>`;
    })
    .join("");
  if (cur && lastJobs.some((j) => j.id === cur)) jobsEl.value = cur;
}

function fillResults(s, keepEditor) {
  const m = s.metrics;
  if (!m) return;
  if (!keepEditor) closeEditor();
  currentJob = m.job || m.folder || "";
  fillJobs(s.jobs);
  document.getElementById("stat-pages").textContent = m.pages ?? "—";
  document.getElementById("stat-total").textContent = m.total_m ?? "—";
  document.getElementById("stat-api").textContent = m.provider || "—";
  document.getElementById("results-title").textContent = reviewFilter
    ? "Ручная разметка: " + reviewFilter.title
    : "Таблица длин";
  document.getElementById("results-pdf").textContent = (m.pdf || "") + (currentJob ? " → output\\" + currentJob : "");
  const live = m.live_api ? ` · живой API: ${m.live_api.provider}` : "";
  document.getElementById("totals").innerHTML =
    `<b>${m.total_mm}</b> мм = <b>${m.total_m}</b> м · ok ${m.ok} · review ${m.review}` + live;
  lastSheets = (m.sheets || []).filter(
    (row) => !reviewFilter || (row.sheet_no >= reviewFilter.start && row.sheet_no <= reviewFilter.end)
  );
  const humanChecked = new Set(m.human_checked_sheets || []);
  rowsEl.innerHTML = lastSheets
    .map(
      (row, i) =>
        `<tr data-i="${i}" data-sheet="${row.sheet_no}">
          <td>${row.sheet_no}</td>
          <td>${row.line_id || ""}</td>
          <td>${row.length_mm}</td>
          <td>${row.length_m}</td>
          <td><span class="badge badge-${
            reviewFilter && !humanChecked.has(row.sheet_no) ? "review" : row.status
          }">${
            reviewFilter && !humanChecked.has(row.sheet_no) ? "не проверен" : row.status
          }</span></td>
          <td>${row.formula || ""}</td>
        </tr>`
    )
    .join("");
  rowsEl.querySelectorAll("tr[data-sheet]").forEach((tr) => {
    tr.addEventListener("click", () => openSheet(Number(tr.dataset.sheet)));
  });
  const thumbs = reviewFilter
    ? lastSheets.map((row) => "/media/" + row.markup)
    : s.thumbs || [];
  thumbsEl.innerHTML = thumbs.map((t, i) => `<img src="${t}" data-i="${i}" alt="">`).join("");
  thumbsEl.querySelectorAll("img").forEach((img) => {
    img.addEventListener("click", () => {
      const sheet = lastSheets[Number(img.dataset.i)];
      if (sheet) openSheet(sheet.sheet_no);
    });
  });
  if (currentJob) {
    document.getElementById("dl-xlsx").href = "/media/" + currentJob + "/lengths.xlsx";
    document.getElementById("dl-html").href = "/media/" + currentJob + "/report.html";
  }
}

async function openSheet(sheetNo) {
  if (!currentJob) return;
  const r = await fetch("/api/sheet/" + encodeURIComponent(currentJob) + "/" + sheetNo);
  const view = await r.json();
  if (!r.ok) {
    alert(view.error || "не открылся лист");
    return;
  }
  drawEditor(view);
}

function drawEditor(view) {
  editorView = view;
  document.getElementById("ed-confirm").textContent = view.locked
    ? "Снять подтверждение"
    : "Подтвердить лист";
  document.getElementById("ed-title").textContent = "Лист " + view.sheet_no + "  " + (view.line_id || "");
  document.getElementById("ed-sum").textContent =
    (view.length_mm || 0) +
    " мм · " +
    (view.length_m || 0) +
    " м · " +
    (view.status || "") +
    (view.locked ? " · закрыт" : "") +
    (view.formula ? " · " + view.formula : "");
  edList.innerHTML = (view.items || [])
    .map(
      (it) =>
        `<li><button type="button" class="is-${it.decision}" data-id="${it.id}">${it.id} · ${it.value_mm} · ${it.label}</button></li>`
    )
    .join("");
  edList.querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => decide(b.dataset.id));
  });
  editorEl.hidden = false;
  const src = "/media/" + view.overlay + "?t=" + Date.now();
  const same = edImg.dataset.overlay === view.overlay && edImg.complete && edImg.naturalWidth;
  if (!same) {
    edImg.dataset.overlay = view.overlay;
    edImg.onload = () => placeHits(view);
    edImg.src = src;
  } else {
    requestAnimationFrame(() => placeHits(view));
  }
}

function placeHits(view) {
  edCanvas.querySelectorAll(".hit").forEach((x) => x.remove());
  const sx = edImg.clientWidth / (view.width || 1);
  const sy = edImg.clientHeight / (view.height || 1);
  (view.items || []).forEach((it) => {
    if (!it.x && !it.y) return;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "hit hit-" + it.decision;
    b.style.left = it.x * sx + "px";
    b.style.top = it.y * sy + "px";
    b.textContent = it.id + " " + it.value_mm;
    b.title = it.label;
    b.addEventListener("click", () => decide(it.id));
    edCanvas.appendChild(b);
  });
}

async function decide(cid) {
  if (editorBusy || !editorView) return;
  editorBusy = true;
  try {
    const r = await fetch("/api/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job: currentJob, sheet_no: editorView.sheet_no, id: cid }),
    });
    const data = await r.json();
    if (!r.ok) {
      alert(data.error || "не сохранилась правка");
      return;
    }
    fillResults(data, true);
    const again = await fetch("/api/sheet/" + encodeURIComponent(currentJob) + "/" + editorView.sheet_no);
    const view = await again.json();
    if (again.ok) drawEditor(view);
  } finally {
    editorBusy = false;
  }
}

function closeEditor() {
  editorEl.hidden = true;
  editorView = null;
}

document.getElementById("ed-close").addEventListener("click", closeEditor);
document.getElementById("ed-confirm").addEventListener("click", async () => {
  if (!editorView || !currentJob) return;
  const btn = document.getElementById("ed-confirm");
  btn.disabled = true;
  const r = await fetch("/api/confirm-sheet", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job: currentJob,
      sheet_no: editorView.sheet_no,
      confirmed: !editorView.locked,
    }),
  });
  const data = await r.json();
  btn.disabled = false;
  if (!r.ok) {
    alert(data.error || "не удалось подтвердить лист");
    return;
  }
  fillTrain(data.train);
  drawEditor(data.view);
});
editorEl.addEventListener("click", (e) => {
  if (e.target === editorEl) closeEditor();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !editorEl.hidden) closeEditor();
});
window.addEventListener("resize", () => {
  if (!editorEl.hidden && editorView) placeHits(editorView);
});

fetch("/api/status")
  .then((r) => r.json())
  .then((s) => {
    fillTrain(s.train);
    if (s.metrics) fillResults(s);
    setRunning(!!s.running);
    if (s.running) {
      if (s.mode === "train" || s.mode === "audit") {
        showPage("train");
        trainPoll = setInterval(tickTrain, 800);
      } else {
        poll = setInterval(tick, 800);
      }
    }
  });
