/* ── Theme ─────────────────────────────────────────────────────── */
const html = document.documentElement;
const themeBtn = document.getElementById('themeBtn');

function applyTheme(dark) {
  html.classList.toggle('dark', dark);
  themeBtn.textContent = dark ? 'Light mode' : 'Dark mode';
  localStorage.setItem('catdog-theme', dark ? 'dark' : 'light');
  if (histChart) refreshChartTheme();
}
applyTheme(localStorage.getItem('catdog-theme') === 'dark');
themeBtn.onclick = () => applyTheme(!html.classList.contains('dark'));

/* ── Chart.js helpers ──────────────────────────────────────────── */
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function getColors() {
  return {
    border: cssVar('--border'),
    muted:  cssVar('--muted'),
    cat:    cssVar('--cat'),
    dog:    cssVar('--dog'),
    text:   cssVar('--text'),
    bg:     cssVar('--surface'),
    mono:   "'JetBrains Mono', monospace",
  };
}

Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size   = 11;

let histChart  = null;
let donutChart = null;

function buildCharts() {
  const c = getColors();

  histChart = new Chart(document.getElementById('histChart'), {
    type: 'bar',
    data: {
      labels: ['0–10','10–20','20–30','30–40','40–50','50–60','60–70','70–80','80–90','90–100'].map(l => l + '%'),
      datasets: [{
        data: new Array(10).fill(0),
        backgroundColor: Array.from({length:10}, (_, i) => i < 5 ? c.cat + '99' : c.dog + '99'),
        borderRadius: 3,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: c.muted, font: { size: 10 } } },
        y: { beginAtZero: true, grid: { color: c.border }, border: { display: false }, ticks: { color: c.muted, font: { size: 10 }, precision: 0 } }
      }
    }
  });

  donutChart = new Chart(document.getElementById('donutChart'), {
    type: 'doughnut',
    data: {
      labels: ['Cat', 'Dog'],
      datasets: [{
        data: [0, 0],
        backgroundColor: [c.cat, c.dog],
        borderColor: c.bg,
        borderWidth: 3,
        hoverOffset: 3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '72%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 8, boxHeight: 8, padding: 14, color: c.muted, font: { size: 11 } } }
      }
    }
  });
}

function refreshChartTheme() {
  const c = getColors();
  if (histChart) {
    histChart.options.scales.x.ticks.color = c.muted;
    histChart.options.scales.y.ticks.color = c.muted;
    histChart.options.scales.y.grid.color  = c.border;
    histChart.data.datasets[0].backgroundColor =
      Array.from({length:10}, (_, i) => i < 5 ? c.cat + '99' : c.dog + '99');
    histChart.update('none');
  }
  if (donutChart) {
    donutChart.data.datasets[0].backgroundColor = [c.cat, c.dog];
    donutChart.data.datasets[0].borderColor      = c.bg;
    donutChart.options.plugins.legend.labels.color = c.muted;
    donutChart.update('none');
  }
}

/* ── Data loading ──────────────────────────────────────────────── */
async function loadStats() {
  try {
    const res = await fetch('/stats');
    if (!res.ok) throw new Error('stats failed');
    const s = await res.json();
    renderAll(s);
  } catch(e) {
    console.error('Dashboard:', e.message);
  }
}

function pct(v)  { return v != null ? `${(v * 100).toFixed(1)}%` : '—'; }
function num(v)  { return (v ?? 0).toLocaleString(); }

function renderAll(s) {
  /* KPIs */
  document.getElementById('kTotal').textContent    = num(s.total);
  document.getElementById('kCats').textContent     = num(s.class_counts?.cat);
  document.getElementById('kDogs').textContent     = num(s.class_counts?.dog);
  document.getElementById('kUncertain').textContent= num(s.uncertain_count);
  document.getElementById('kConf').textContent     = pct(s.avg_confidence?.overall);

  /* Sidebar */
  document.getElementById('sb-total').textContent    = num(s.total);
  document.getElementById('sb-conf').textContent     = pct(s.avg_confidence?.overall);
  document.getElementById('sb-uncertain').textContent= pct(s.uncertain_rate);

  /* Avg conf stat rows */
  document.getElementById('confCat').textContent     = pct(s.avg_confidence?.cat);
  document.getElementById('confDog').textContent     = pct(s.avg_confidence?.dog);
  document.getElementById('confAll').textContent     = pct(s.avg_confidence?.overall);
  document.getElementById('confUncertain').textContent = pct(s.uncertain_rate);

  /* Histogram */
  if (histChart) {
    histChart.data.datasets[0].data = s.confidence_histogram || new Array(10).fill(0);
    histChart.update();
  }

  /* Donut */
  if (donutChart && s.total > 0) {
    donutChart.data.datasets[0].data = [s.class_counts?.cat ?? 0, s.class_counts?.dog ?? 0];
    donutChart.update();
    const catPct = Math.round((s.class_counts?.cat ?? 0) / s.total * 100);
    document.getElementById('donutPct').textContent = catPct + '%';
  }

  /* Table */
  renderTable(s.recent || []);

  /* Timestamp */
  document.getElementById('lastUpdated').textContent =
    'Updated ' + new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function renderTable(rows) {
  const tbody = document.getElementById('logBody');
  const empty = document.getElementById('emptyLog');
  if (!rows.length) { tbody.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  tbody.innerHTML = rows.map(r => {
    const conf = (r.confidence * 100).toFixed(1);
    const time = new Date(r.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
    return `<tr>
      <td class="t-mono t-xs text-muted">${time}</td>
      <td><span class="badge badge-${r.class}">${r.class}</span></td>
      <td>
        <div class="conf-bar-row">
          <div class="conf-bar"><div class="conf-bar-fill ${r.class}" style="width:${conf}%"></div></div>
          <span class="conf-num">${conf}%</span>
        </div>
      </td>
      <td class="t-mono t-xs text-muted">${r.source || 'predict'}</td>
      <td>${r.uncertain
        ? '<span class="badge badge-warn">uncertain</span>'
        : '<span class="badge badge-ok">confident</span>'}</td>
    </tr>`;
  }).join('');
}

/* ── Sidebar nav active on scroll ──────────────────────────────── */
const navMap = { overview: 'nav-overview', charts: 'nav-charts', log: 'nav-log' };
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      Object.values(navMap).forEach(id => document.getElementById(id)?.classList.remove('active'));
      const navId = navMap[e.target.id];
      if (navId) document.getElementById(navId)?.classList.add('active');
    }
  });
}, { threshold: 0.3 });
['overview','charts','log'].forEach(id => { const el = document.getElementById(id); if (el) observer.observe(el); });

/* ── Init ──────────────────────────────────────────────────────── */
document.getElementById('refreshBtn').onclick = loadStats;

async function loadStatsWithError() {
  try {
    const res = await fetch('/stats');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();
    renderAll(s);
  } catch(e) {
    console.error('Dashboard loadStats error:', e);
    document.getElementById('lastUpdated').textContent = 'Error: ' + e.message;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  buildCharts();
  loadStatsWithError();
  setInterval(loadStatsWithError, 5_000);
});
