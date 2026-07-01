/* ── Theme ───────────────────────────────────────────────── */
const html = document.documentElement;
const themeToggle = document.getElementById('themeToggle');

function applyTheme(dark) {
  html.classList.toggle('dark', dark);
  themeToggle.textContent = dark ? '☀️' : '🌙';
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}

// Load saved theme
applyTheme(localStorage.getItem('theme') === 'dark');
themeToggle.onclick = () => applyTheme(!html.classList.contains('dark'));

/* ── Animal facts data ───────────────────────────────────── */
const ANIMAL_FACTS = {
  cat: [
    { icon: '👁️', label: 'Pupils',      value: 'Vertical slit — precise depth vision' },
    { icon: '🐾', label: 'Claws',       value: 'Fully retractable for stealth' },
    { icon: '💤', label: 'Sleep',       value: '12–16 hours per day' },
    { icon: '⚖️', label: 'Weight',      value: '3.5–5 kg on average' },
    { icon: '🔊', label: 'Sounds',      value: 'Purr, meow, hiss, chirp' },
    { icon: '🌡️', label: 'Body temp',   value: '38–39.2 °C (100.4–102.6 °F)' },
    { icon: '🦷', label: 'Teeth',       value: '30 teeth — obligate carnivore' },
    { icon: '📅', label: 'Lifespan',    value: '12–18 years (domestic)' },
  ],
  dog: [
    { icon: '👁️', label: 'Pupils',      value: 'Round — wide field of view' },
    { icon: '🐾', label: 'Claws',       value: 'Non-retractable, for traction' },
    { icon: '💤', label: 'Sleep',       value: '12–14 hours per day' },
    { icon: '⚖️', label: 'Weight',      value: '2–90 kg (varies by breed)' },
    { icon: '🔊', label: 'Sounds',      value: 'Bark, howl, whine, growl' },
    { icon: '👃', label: 'Smell',       value: '100,000× stronger than humans' },
    { icon: '🦷', label: 'Teeth',       value: '42 teeth — bone-crushing bite' },
    { icon: '📅', label: 'Lifespan',    value: '10–13 years (varies by size)' },
  ],
};

/* ── DOM refs ────────────────────────────────────────────── */
const dropzone     = document.getElementById('dropzone');
const fileInput    = document.getElementById('fileInput');
const startCamBtn  = document.getElementById('startCam');
const snapBtn      = document.getElementById('snap');
const cancelCamBtn = document.getElementById('cancelCam');
const video        = document.getElementById('video');
const canvas       = document.getElementById('canvas');
const previewRow   = document.getElementById('previewRow');
const previewImg   = document.getElementById('previewImg');
const fnameEl      = document.getElementById('fname');
const statusEl     = document.getElementById('status');
const resultEl     = document.getElementById('result');
const verdictEl    = document.getElementById('verdict');
const confidenceEl = document.getElementById('confidence');
const uncertainBox = document.getElementById('uncertainBox');
const uncertainBody= document.getElementById('uncertainBody');
const factsPanel   = document.getElementById('factsPanel');
const factsTitle   = document.getElementById('factsTitle');
const factsGrid    = document.getElementById('factsGrid');
const heatmapBtn   = document.getElementById('heatmapBtn');
const heatmapWrap  = document.getElementById('heatmapWrap');
const heatmapImg   = document.getElementById('heatmapImg');
const errorMsg     = document.getElementById('errorMsg');
const meterFill    = document.getElementById('meterFill');
const meterKnot    = document.getElementById('meterKnot');
const historyShell = document.getElementById('historyShell');
const historyStrip = document.getElementById('historyStrip');
const batchGrid    = document.getElementById('batchGrid');

let stream = null;
let currentFile = null;
const historyItems = [];

/* ── Meter ───────────────────────────────────────────────── */
function resetMeter() {
  meterFill.style.width = '50%';
  meterKnot.style.left  = '50%';
}
function setMeter(catProb, dogProb) {
  const pct = dogProb * 100;
  meterFill.style.width = pct + '%';
  meterKnot.style.left  = pct + '%';
}

/* ── Drag/drop ───────────────────────────────────────────── */
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag');
  handleFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
});
fileInput.onchange = () => { handleFiles(Array.from(fileInput.files)); fileInput.value = ''; };

function handleFiles(files) {
  if (!files.length) return;
  if (files.length === 1) {
    batchGrid.style.display = 'none';
    batchGrid.innerHTML = '';
    handleSingle(files[0]);
  } else {
    resultEl.style.display = 'none';
    previewRow.style.display = 'none';
    handleBatch(files);
  }
}

/* ── Camera ──────────────────────────────────────────────── */
startCamBtn.onclick = async () => {
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } }); }
  catch { stream = await navigator.mediaDevices.getUserMedia({ video: true }); }
  video.srcObject = stream;
  video.style.display = 'block';
  snapBtn.style.display = 'inline-block';
  cancelCamBtn.style.display = 'inline-block';
  startCamBtn.style.display = 'none';
};
cancelCamBtn.onclick = stopCam;
function stopCam() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.style.display = snapBtn.style.display = cancelCamBtn.style.display = 'none';
  startCamBtn.style.display = 'inline-block';
}
snapBtn.onclick = () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => handleSingle(new File([blob], 'photo.jpg', { type: 'image/jpeg' })), 'image/jpeg', 0.92);
  stopCam();
};

/* ── Single file flow ────────────────────────────────────── */
function handleSingle(file) {
  currentFile = file;
  errorMsg.style.display = 'none';
  resultEl.style.display = 'none';
  uncertainBox.style.display = 'none';
  factsPanel.style.display = 'none';
  heatmapWrap.style.display = 'none';
  heatmapBtn.textContent = '🔍 Show what the model focused on';
  previewRow.style.display = 'flex';
  previewImg.src = URL.createObjectURL(file);
  fnameEl.textContent = file.name;
  statusEl.textContent = 'Classifying…';
  resetMeter();
  classify(file);
}

async function classify(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `Error ${res.status}`); }
    const data = await res.json();
    showResult(data, previewImg.src);
    statusEl.textContent = 'Done';
  } catch(e) {
    statusEl.textContent = 'Failed';
    errorMsg.style.display = 'block';
    errorMsg.textContent = e.message || 'Server not reachable.';
  }
}

/* ── Show result ─────────────────────────────────────────── */
function showResult(data, thumbUrl) {
  const catProb = data.probabilities?.cat ?? (data.class === 'cat' ? data.confidence : 1 - data.confidence);
  const dogProb = data.probabilities?.dog ?? (data.class === 'dog' ? data.confidence : 1 - data.confidence);
  setMeter(catProb, dogProb);

  resultEl.style.display = 'block';

  if (data.uncertain) {
    // Don't commit to a verdict — show a clear "probably not a cat/dog" message
    verdictEl.textContent = '🤷 Not sure…';
    verdictEl.className = 'verdict uncertain-verdict';
    confidenceEl.innerHTML = `Best guess: <b>${data.class}</b> at <b>${(data.confidence * 100).toFixed(1)}%</b> — below confidence threshold`;

    uncertainBox.style.display = 'block';
    const reasons = {
      high_entropy: "The model's probability is nearly equally split between cat and dog — a sign the image may not contain either.",
      low_confidence: "The model's confidence is below 75%. This image may show a different animal, object, or unclear scene.",
      confident: '',
    };
    uncertainBody.textContent = reasons[data.uncertainty_reason] || reasons.low_confidence;
    factsPanel.style.display = 'none';
  } else {
    // Confident prediction
    verdictEl.textContent = data.class === 'cat' ? '🐱 It\'s a cat!' : '🐶 It\'s a dog!';
    verdictEl.className = 'verdict ' + data.class;
    confidenceEl.innerHTML = `Confidence: <b>${(data.confidence * 100).toFixed(1)}%</b>`;
    uncertainBox.style.display = 'none';
    showFacts(data.class);
  }

  addToHistory(thumbUrl, data.class, data.confidence, data.uncertain);
}

/* ── Animal facts ────────────────────────────────────────── */
function showFacts(cls) {
  factsTitle.textContent = cls === 'cat' ? '🐱 Cat characteristics' : '🐶 Dog characteristics';
  factsGrid.innerHTML = ANIMAL_FACTS[cls].map(f => `
    <div class="fact-chip">
      <div class="fc-icon">${f.icon}</div>
      <div class="fc-label">${f.label}</div>
      <div class="fc-value">${f.value}</div>
    </div>
  `).join('');
  factsPanel.style.display = 'block';
}

/* ── Grad-CAM ────────────────────────────────────────────── */
heatmapBtn.onclick = async () => {
  if (!currentFile) return;
  if (heatmapWrap.style.display === 'block') {
    heatmapWrap.style.display = 'none';
    heatmapBtn.textContent = '🔍 Show what the model focused on';
    return;
  }
  heatmapBtn.textContent = 'Generating…';
  heatmapBtn.disabled = true;
  const fd = new FormData();
  fd.append('file', currentFile);
  try {
    const res = await fetch('/predict-gradcam', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Heatmap failed');
    const data = await res.json();
    heatmapImg.src = `data:image/png;base64,${data.heatmap_base64}`;
    heatmapWrap.style.display = 'block';
    heatmapBtn.textContent = '🔍 Hide heatmap';
  } catch(e) {
    errorMsg.style.display = 'block';
    errorMsg.textContent = e.message;
    heatmapBtn.textContent = '🔍 Show what the model focused on';
  } finally { heatmapBtn.disabled = false; }
};

/* ── History ─────────────────────────────────────────────── */
function addToHistory(url, cls, conf, uncertain) {
  historyItems.unshift({ url, cls, conf, uncertain });
  if (historyItems.length > 14) historyItems.pop();
  renderHistory();
}
function renderHistory() {
  if (!historyItems.length) { historyShell.style.display = 'none'; return; }
  historyShell.style.display = 'block';
  historyStrip.innerHTML = historyItems.map(h => `
    <div class="history-item ${h.uncertain ? '' : h.cls}">
      <img src="${h.url}" alt="${h.cls}">
      <div class="h-label">${h.uncertain ? '🤷' : h.cls === 'cat' ? '🐱' : '🐶'} ${(h.conf * 100).toFixed(0)}%</div>
    </div>
  `).join('');
}

/* ── Batch ───────────────────────────────────────────────── */
async function handleBatch(files) {
  batchGrid.style.display = 'grid';
  batchGrid.innerHTML = '';
  const cards = files.map(file => {
    const card = document.createElement('div');
    card.className = 'batch-card pending';
    const url = URL.createObjectURL(file);
    card.innerHTML = `<img src="${url}"><div class="b-label">…</div><div class="b-conf"></div>`;
    batchGrid.appendChild(card);
    return { file, card, url };
  });
  for (const { file, card, url } of cards) {
    const fd = new FormData(); fd.append('file', file);
    try {
      const res = await fetch('/predict', { method: 'POST', body: fd });
      if (!res.ok) throw new Error();
      const data = await res.json();
      card.classList.remove('pending');
      if (data.uncertain) {
        card.querySelector('.b-label').textContent = '🤷 Unsure';
        card.querySelector('.b-conf').textContent = `${(data.confidence * 100).toFixed(1)}%`;
      } else {
        card.classList.add(data.class);
        card.querySelector('.b-label').textContent = data.class === 'cat' ? '🐱 Cat' : '🐶 Dog';
        card.querySelector('.b-conf').textContent = `${(data.confidence * 100).toFixed(1)}%`;
      }
      addToHistory(url, data.class, data.confidence, data.uncertain);
    } catch {
      card.classList.remove('pending');
      card.querySelector('.b-label').textContent = 'Error';
    }
  }
}
