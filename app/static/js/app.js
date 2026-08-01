/* ── Theme ─────────────────────────────────────────────────────── */
const html = document.documentElement;
const themeBtn = document.getElementById('themeBtn');

function applyTheme(dark) {
  html.classList.toggle('dark', dark);
  themeBtn.textContent = dark ? 'Light' : 'Dark';
  localStorage.setItem('catdog-theme', dark ? 'dark' : 'light');
}
applyTheme(localStorage.getItem('catdog-theme') === 'dark');
themeBtn.onclick = () => applyTheme(!html.classList.contains('dark'));

/* ── DOM refs ──────────────────────────────────────────────────── */
const dropzone      = document.getElementById('dropzone');
const fileInput     = document.getElementById('fileInput');
const startCamBtn   = document.getElementById('startCam');
const snapBtn       = document.getElementById('snap');
const cancelCamBtn  = document.getElementById('cancelCam');
const video         = document.getElementById('video');
const canvas        = document.getElementById('canvas');
const previewRow    = document.getElementById('previewRow');
const previewImg    = document.getElementById('previewImg');
const fnameEl       = document.getElementById('fname');
const statusEl      = document.getElementById('status');
const resultSection = document.getElementById('resultSection');
const verdictLabel  = document.getElementById('verdictLabel');
const verdictConf   = document.getElementById('verdictConf');
const uncertainBox  = document.getElementById('uncertainBox');
const uncertainBody = document.getElementById('uncertainBody');
const breedPanel    = document.getElementById('breedPanel');
const breedList     = document.getElementById('breedList');
const propsPanel    = document.getElementById('propsPanel');
const propsGrid     = document.getElementById('propsGrid');
const heatmapBtn    = document.getElementById('heatmapBtn');
const heatmapWrap   = document.getElementById('heatmapWrap');
const heatmapImg    = document.getElementById('heatmapImg');
const errorMsg      = document.getElementById('errorMsg');
const probFill      = document.getElementById('probFill');
const probKnob      = document.getElementById('probKnob');
const historyShell  = document.getElementById('historyShell');
const historyStrip  = document.getElementById('historyStrip');
const batchGrid     = document.getElementById('batchGrid');

let stream = null;
let currentFile = null;
const historyItems = [];

/* ── Probability bar ───────────────────────────────────────────── */
function resetBar() { probFill.style.width = '50%'; probKnob.style.left = '50%'; }
function setBar(catP, dogP) {
  const pct = dogP * 100;
  probFill.style.width = pct + '%';
  probKnob.style.left  = pct + '%';
}

/* ── File intake ───────────────────────────────────────────────── */
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag');
  handleFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
});
fileInput.onchange = () => { handleFiles(Array.from(fileInput.files)); fileInput.value = ''; };

function handleFiles(files) {
  if (!files.length) return;
  if (files.length === 1) { batchGrid.style.display = 'none'; batchGrid.innerHTML = ''; handleSingle(files[0]); }
  else { resultSection.style.display = 'none'; previewRow.style.display = 'none'; handleBatch(files); }
}

/* ── Camera ────────────────────────────────────────────────────── */
startCamBtn.onclick = async () => {
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } }); }
  catch { stream = await navigator.mediaDevices.getUserMedia({ video: true }); }
  video.srcObject = stream; video.style.display = 'block';
  snapBtn.style.display = cancelCamBtn.style.display = 'inline-flex';
  startCamBtn.style.display = 'none';
};
cancelCamBtn.onclick = stopCam;
function stopCam() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.style.display = 'none';
  snapBtn.style.display = cancelCamBtn.style.display = 'none';
  startCamBtn.style.display = 'inline-flex';
}
snapBtn.onclick = () => {
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => handleSingle(new File([blob], 'capture.jpg', { type: 'image/jpeg' })), 'image/jpeg', 0.92);
  stopCam();
};

/* ── Single file flow ──────────────────────────────────────────── */
function handleSingle(file) {
  currentFile = file;
  errorMsg.style.display     = 'none';
  resultSection.style.display = 'none';
  uncertainBox.style.display = 'none';
  breedPanel.style.display   = 'none';
  propsPanel.style.display   = 'none';
  heatmapWrap.style.display  = 'none';
  heatmapBtn.textContent     = 'Show attention map';
  previewRow.style.display   = 'flex';
  previewImg.src             = URL.createObjectURL(file);
  fnameEl.textContent        = file.name;
  statusEl.textContent       = 'Analysing…';
  resetBar();
  runAnalyze(file);
}

/* ── /analyze call ─────────────────────────────────────────────── */
async function runAnalyze(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
    const data = await res.json();
    statusEl.textContent = 'Done';
    showResult(data, previewImg.src);
  } catch(e) {
    statusEl.textContent = 'Failed';
    errorMsg.style.display = 'block';
    errorMsg.textContent = e.message || 'Could not reach server.';
  }
}

/* ── Render result ─────────────────────────────────────────────── */
function showResult(data, thumbUrl) {
  const catP = data.probabilities?.cat ?? (data.class === 'cat' ? data.confidence : 1 - data.confidence);
  const dogP = data.probabilities?.dog ?? (data.class === 'dog' ? data.confidence : 1 - data.confidence);
  setBar(catP, dogP);
  resultSection.style.display = 'block';

  if (data.uncertain) {
    verdictLabel.textContent = 'Uncertain';
    verdictLabel.className   = 'verdict-label unsure';
    verdictConf.textContent  = `Best guess: ${data.class} · ${(data.confidence * 100).toFixed(1)}%`;
    uncertainBox.style.display = 'block';
    const reasons = {
      high_entropy:   'Probability is nearly equally split between cat and dog. The image may not contain either.',
      low_confidence: 'Confidence is below 75%. The image may show a different animal, object, or unclear scene.',
    };
    uncertainBody.textContent = reasons[data.uncertainty_reason] || reasons.low_confidence;
  } else {
    verdictLabel.textContent = data.class === 'cat' ? 'Cat' : 'Dog';
    verdictLabel.className   = `verdict-label ${data.class}`;
    verdictConf.textContent  = `${(data.confidence * 100).toFixed(1)}% confidence`;
    uncertainBox.style.display = 'none';
    renderBreeds(data.breeds, data.class);
  }

  renderProps(data.image_properties);
  addToHistory(thumbUrl, data.class, data.confident);
}

/* ── Breed panel ───────────────────────────────────────────────── */
function renderBreeds(breeds, cls) {
  if (!breeds?.length) { breedPanel.style.display = 'none'; return; }
  breedPanel.style.display = 'block';
  breedList.innerHTML = breeds.map(b => `
    <div class="breed-row">
      <span class="breed-name">${b.breed}</span>
      <div class="breed-bar-bg">
        <div class="breed-bar-fill ${cls}" style="width:${b.relative_pct}%"></div>
      </div>
      <span class="breed-pct">${b.relative_pct.toFixed(0)}%</span>
    </div>
  `).join('');
}

/* ── Image properties panel ────────────────────────────────────── */
function renderProps(p) {
  if (!p) { propsPanel.style.display = 'none'; return; }
  propsPanel.style.display = 'block';

  const bar = (v) => `<div style="height:3px;background:var(--border);border-radius:2px;margin-top:4px;overflow:hidden">
    <div style="width:${(v*100).toFixed(0)}%;height:100%;background:var(--text);border-radius:2px"></div></div>`;

  propsGrid.innerHTML = `
    <div class="prop-chip">
      <div class="prop-label">Size</div>
      <div class="prop-value">${p.width}×${p.height}</div>
    </div>
    <div class="prop-chip">
      <div class="prop-label">Aspect</div>
      <div class="prop-value">${p.aspect_ratio}</div>
    </div>
    <div class="prop-chip">
      <div class="prop-label">Quality</div>
      <div class="prop-value">${p.quality}</div>
    </div>
    <div class="prop-chip">
      <div class="prop-label">Colour</div>
      <div class="prop-value">
        <span class="color-swatch" style="background:${p.dominant_color}"></span>${p.dominant_color}
      </div>
    </div>
    <div class="prop-chip" style="grid-column:1/3">
      <div class="prop-label">Brightness · ${(p.brightness*100).toFixed(0)}%</div>
      ${bar(p.brightness)}
    </div>
    <div class="prop-chip" style="grid-column:3/5">
      <div class="prop-label">Contrast · ${(p.contrast*100).toFixed(0)}%</div>
      ${bar(p.contrast)}
    </div>
    <div class="prop-chip" style="grid-column:1/3">
      <div class="prop-label">Sharpness · ${(p.sharpness*100).toFixed(0)}%</div>
      ${bar(p.sharpness)}
    </div>
    <div class="prop-chip" style="grid-column:3/5">
      <div class="prop-label">Mode</div>
      <div class="prop-value">${p.mode || 'RGB'}</div>
    </div>
  `;
}

/* ── Grad-CAM ──────────────────────────────────────────────────── */
heatmapBtn.onclick = async () => {
  if (!currentFile) return;
  if (heatmapWrap.style.display === 'block') {
    heatmapWrap.style.display = 'none'; heatmapBtn.textContent = 'Show attention map'; return;
  }
  heatmapBtn.textContent = 'Generating…'; heatmapBtn.disabled = true;
  const fd = new FormData(); fd.append('file', currentFile);
  try {
    const res = await fetch('/predict-gradcam', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Heatmap failed');
    const data = await res.json();
    heatmapImg.src = `data:image/png;base64,${data.heatmap_base64}`;
    heatmapWrap.style.display = 'block';
    heatmapBtn.textContent = 'Hide attention map';
  } catch(e) {
    errorMsg.style.display = 'block'; errorMsg.textContent = e.message;
    heatmapBtn.textContent = 'Show attention map';
  } finally { heatmapBtn.disabled = false; }
};

/* ── History ───────────────────────────────────────────────────── */
function addToHistory(url, cls, uncertain) {
  historyItems.unshift({ url, cls, uncertain });
  if (historyItems.length > 16) historyItems.pop();
  historyShell.style.display = 'block';
  historyStrip.innerHTML = historyItems.map(h => `
    <div class="history-item ${h.uncertain ? '' : h.cls}">
      <img src="${h.url}" alt="">
    </div>
  `).join('');
}

/* ── Batch ─────────────────────────────────────────────────────── */
async function handleBatch(files) {
  batchGrid.style.display = 'grid';
  batchGrid.innerHTML = '';
  const cards = files.map(file => {
    const card = document.createElement('div');
    card.className = 'batch-card loading';
    const url = URL.createObjectURL(file);
    card.innerHTML = `<img src="${url}" alt=""><div class="batch-label">…</div><div class="batch-conf"></div>`;
    batchGrid.appendChild(card);
    return { file, card };
  });
  for (const { file, card } of cards) {
    const fd = new FormData(); fd.append('file', file);
    try {
      const res = await fetch('/analyze', { method: 'POST', body: fd });
      if (!res.ok) throw new Error();
      const d = await res.json();
      card.classList.remove('loading');
      const lbl = card.querySelector('.batch-label');
      const conf = card.querySelector('.batch-conf');
      if (d.uncertain) {
        lbl.textContent = 'Uncertain'; lbl.className = 'batch-label unsure';
      } else {
        lbl.textContent = d.class === 'cat' ? 'Cat' : 'Dog';
        lbl.className   = `batch-label ${d.class}`;
      }
      conf.textContent = `${(d.confidence * 100).toFixed(1)}%`;
    } catch {
      card.classList.remove('loading');
      card.querySelector('.batch-label').textContent = 'Error';
    }
  }
}