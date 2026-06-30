const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const startCamBtn = document.getElementById('startCam');
const snapBtn = document.getElementById('snap');
const cancelCamBtn = document.getElementById('cancelCam');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const previewRow = document.getElementById('previewRow');
const previewImg = document.getElementById('previewImg');
const fnameEl = document.getElementById('fname');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const verdictEl = document.getElementById('verdict');
const confidenceEl = document.getElementById('confidence');
const uncertainNote = document.getElementById('uncertainNote');
const errorMsg = document.getElementById('errorMsg');
const meterFill = document.getElementById('meterFill');
const meterKnot = document.getElementById('meterKnot');
const heatmapBtn = document.getElementById('heatmapBtn');
const heatmapWrap = document.getElementById('heatmapWrap');
const heatmapImg = document.getElementById('heatmapImg');
const historyShell = document.getElementById('historyShell');
const historyStrip = document.getElementById('historyStrip');
const batchGrid = document.getElementById('batchGrid');

let stream = null;
let currentFile = null;       // file currently shown in single-result view
const history = [];           // { thumbUrl, cls, confidence }
const MAX_HISTORY = 14;

/* ---------- meter ---------- */
function resetMeter() {
  meterFill.style.width = '50%';
  meterKnot.style.left = '50%';
}
function setMeter(catProb, dogProb) {
  const dogPct = dogProb * 100;
  meterFill.style.width = dogPct + '%';
  meterKnot.style.left = dogPct + '%';
}

/* ---------- file intake ---------- */
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  handleFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
});

fileInput.onchange = () => {
  handleFiles(Array.from(fileInput.files));
  fileInput.value = ''; // allow re-selecting the same file later
};

function handleFiles(files) {
  if (!files.length) return;
  if (files.length === 1) {
    batchGrid.style.display = 'none';
    batchGrid.innerHTML = '';
    handleSingleFile(files[0]);
  } else {
    resultEl.style.display = 'none';
    previewRow.style.display = 'none';
    handleBatch(files);
  }
}

/* ---------- camera ---------- */
startCamBtn.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
  } catch {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
  }
  video.srcObject = stream;
  video.style.display = 'block';
  snapBtn.style.display = 'inline-block';
  cancelCamBtn.style.display = 'inline-block';
  startCamBtn.style.display = 'none';
};

cancelCamBtn.onclick = () => stopCamera();

function stopCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.style.display = 'none';
  snapBtn.style.display = 'none';
  cancelCamBtn.style.display = 'none';
  startCamBtn.style.display = 'inline-block';
}

snapBtn.onclick = () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    handleSingleFile(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' }));
  }, 'image/jpeg', 0.92);
  stopCamera();
};

/* ---------- single-file flow ---------- */
function handleSingleFile(file) {
  currentFile = file;
  errorMsg.style.display = 'none';
  resultEl.style.display = 'none';
  uncertainNote.style.display = 'none';
  heatmapWrap.style.display = 'none';
  heatmapImg.src = '';
  heatmapBtn.textContent = 'Show what the model focused on';
  previewRow.style.display = 'flex';
  previewImg.src = URL.createObjectURL(file);
  fnameEl.textContent = file.name;
  statusEl.textContent = 'Classifying…';
  resetMeter();
  sendImage(file);
}

async function sendImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }
    const data = await res.json();
    statusEl.textContent = 'Done';

    const catProb = data.probabilities?.cat ?? (data.class === 'cat' ? data.confidence : 1 - data.confidence);
    const dogProb = data.probabilities?.dog ?? (data.class === 'dog' ? data.confidence : 1 - data.confidence);
    setMeter(catProb, dogProb);

    resultEl.style.display = 'block';
    resultEl.classList.remove('cat', 'dog');
    resultEl.classList.add(data.class);
    verdictEl.textContent = data.class === 'cat' ? '🐱 It\'s a cat' : '🐶 It\'s a dog';
    verdictEl.className = 'verdict ' + data.class;
    confidenceEl.innerHTML = `<b>${(data.confidence * 100).toFixed(1)}%</b> confidence`;
    uncertainNote.style.display = data.uncertain ? 'inline-block' : 'none';

    addToHistory(previewImg.src, data.class, data.confidence);
  } catch (e) {
    statusEl.textContent = 'Failed';
    errorMsg.style.display = 'block';
    errorMsg.textContent = e.message || 'Could not reach the classifier.';
  }
}

/* ---------- Grad-CAM ---------- */
heatmapBtn.onclick = async () => {
  if (!currentFile) return;

  if (heatmapWrap.style.display === 'block') {
    heatmapWrap.style.display = 'none';
    heatmapBtn.textContent = 'Show what the model focused on';
    return;
  }

  heatmapBtn.textContent = 'Loading heatmap…';
  heatmapBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res = await fetch('/predict-gradcam', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Could not generate heatmap.');
    const data = await res.json();
    heatmapImg.src = `data:image/png;base64,${data.heatmap_base64}`;
    heatmapWrap.style.display = 'block';
    heatmapBtn.textContent = 'Hide heatmap';
  } catch (e) {
    errorMsg.style.display = 'block';
    errorMsg.textContent = e.message;
    heatmapBtn.textContent = 'Show what the model focused on';
  } finally {
    heatmapBtn.disabled = false;
  }
};

/* ---------- history gallery ---------- */
function addToHistory(thumbUrl, cls, confidence) {
  history.unshift({ thumbUrl, cls, confidence });
  if (history.length > MAX_HISTORY) history.pop();
  renderHistory();
}

function renderHistory() {
  if (!history.length) {
    historyShell.style.display = 'none';
    return;
  }
  historyShell.style.display = 'block';
  historyStrip.innerHTML = '';
  history.forEach(item => {
    const div = document.createElement('div');
    div.className = `history-item ${item.cls}`;
    div.innerHTML = `
      <img src="${item.thumbUrl}" alt="${item.cls}">
      <div class="h-label">${item.cls === 'cat' ? '🐱' : '🐶'} ${(item.confidence * 100).toFixed(0)}%</div>
    `;
    historyStrip.appendChild(div);
  });
}

/* ---------- batch upload ---------- */
async function handleBatch(files) {
  batchGrid.style.display = 'grid';
  batchGrid.innerHTML = '';

  const cards = files.map(file => {
    const card = document.createElement('div');
    card.className = 'batch-card pending';
    const url = URL.createObjectURL(file);
    card.innerHTML = `
      <img src="${url}" alt="${file.name}">
      <div class="b-label">…</div>
      <div class="b-conf"></div>
    `;
    batchGrid.appendChild(card);
    return { file, card, url };
  });

  for (const { file, card, url } of cards) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      if (!res.ok) throw new Error();
      const data = await res.json();
      card.classList.remove('pending');
      card.classList.add(data.class);
      card.querySelector('.b-label').textContent = data.class === 'cat' ? '🐱 Cat' : '🐶 Dog';
      card.querySelector('.b-conf').textContent = `${(data.confidence * 100).toFixed(1)}%${data.uncertain ? ' · unsure' : ''}`;
      addToHistory(url, data.class, data.confidence);
    } catch {
      card.classList.remove('pending');
      card.querySelector('.b-label').textContent = 'Error';
    }
  }
}
