
const html = document.documentElement;
const themeToggle = document.getElementById('themeToggle');

function applyTheme(dark) {
  html.classList.toggle('dark', dark);
  html.classList.toggle('light', !dark);
  themeToggle.textContent = dark ? '☀️ Light Mode' : '🌙 Dark Mode';
  localStorage.setItem('theme', dark ? 'dark' : 'light');
  refreshChartThemeProfiles();
}

applyTheme(localStorage.getItem('theme') !== 'light');
themeToggle.onclick = () => applyTheme(!html.classList.contains('dark'));

/* ---------- CHART COLOR FACTORY SETTINGS CONFIGURATIONS ---------- */
function resolveDynamicThemeColors() {
  const isDarkActive = html.classList.contains('dark');
  return {
    gridColor: isDarkActive ? 'rgba(255,255,255,0.03)' : 'rgba(15, 23, 42, 0.05)',
    tickColor: isDarkActive ? '#475569' : '#94a3b8',
    labelColor: isDarkActive ? '#8c9bb2' : '#475569',
    felineColor: isDarkActive ? '#f97316' : '#ea580c',
    canineColor: isDarkActive ? '#38bdf8' : '#0284c7',
    amberColor: '#fbbf24',
    cardBackground: isDarkActive ? '#0a0d18' : '#ffffff'
  };
}

let historicalHistogramChart = null;
let cumulativeDonutChart = null;

/* ---------- INITIALIZE PERFORMANCE CHARTS DECK ---------- */
function constructPerformanceCharts(statsPayload) {
  const tokens = resolveDynamicThemeColors();
  
  // Chart 1: Confidence Distribution Density Bracket Histogram
  const histogramCtx = document.getElementById('historicalHistogramChart').getContext('2d');
  historicalHistogramChart = new Chart(histogramCtx, {
    type: 'bar',
    data: {
      labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
      datasets: [{
        label: 'Prediction Output Frequency Count',
        data: statsPayload.confidence_histogram || [0,0,0,0,0,0,0,0,0,0],
        backgroundColor: tokens.canineColor,
        borderRadius: 4,
        barPercentage: 0.6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { grid: { color: tokens.gridColor }, ticks: { color: tokens.tickColor, font: { family: 'JetBrains Mono', size: 9 } } },
        x: { grid: { display: false }, ticks: { color: tokens.tickColor, font: { family: 'JetBrains Mono', size: 9 } } }
      },
      plugins: { legend: { display: false } }
    }
  });

  // Chart 2: Total Proportions Class Allocation Split
  const donutCtx = document.getElementById('cumulativeDonutChart').getContext('2d');
  cumulativeDonutChart = new Chart(donutCtx, {
    type: 'doughnut',
    data: {
      labels: ['Cats (Feline)', 'Dogs (Canine)', 'Uncertain Runs'],
      datasets: [{
        data: [
          statsPayload.class_counts?.cat || 0,
          statsPayload.class_counts?.dog || 0,
          statsPayload.uncertain_count || 0
        ],
        backgroundColor: [tokens.felineColor, tokens.canineColor, tokens.amberColor],
        borderColor: tokens.cardBackground,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: tokens.labelColor, font: { family: 'Plus Jakarta Sans', size: 10 }, boxWidth: 10 } }
      },
      cutout: '70%'
    }
  });
}

/* ---------- REFRESH VISUALIZATION SYSTEM GRAPH THEMES ---------- */
function refreshChartThemeProfiles() {
  if (!historicalHistogramChart || !cumulativeDonutChart) return;
  const tokens = resolveDynamicThemeColors();

  // Reset histogram chart properties references
  historicalHistogramChart.options.scales.y.grid.color = tokens.gridColor;
  historicalHistogramChart.options.scales.y.ticks.color = tokens.tickColor;
  historicalHistogramChart.options.scales.x.ticks.color = tokens.tickColor;
  historicalHistogramChart.data.datasets[0].backgroundColor = tokens.canineColor;
  historicalHistogramChart.update();

  // Reset doughnut proportions chart color configurations
  cumulativeDonutChart.options.plugins.legend.labels.color = tokens.labelColor;
  cumulativeDonutChart.data.datasets[0].backgroundColor = [tokens.felineColor, tokens.canineColor, tokens.amberColor];
  cumulativeDonutChart.data.datasets[0].borderColor = tokens.cardBackground;
  cumulativeDonutChart.update();
}

/* ---------- COMPILE RETRIEVED TELEMETRY ARRAYS DATA ---------- */
function renderExecutiveTelemMetrics(statsPayload) {
  // Update numerical metrics representations securely
  document.getElementById('statTotal').textContent = statsPayload.total_predictions || 0;
  
  const catAccuracyMagnitude = (statsPayload.avg_confidence?.cat || 0) * 100;
  document.getElementById('statCatConf').textContent = catAccuracyMagnitude.toFixed(1) + '%';
  
  const dogAccuracyMagnitude = (statsPayload.avg_confidence?.dog || 0) * 100;
  document.getElementById('statDogConf').textContent = dogAccuracyMagnitude.toFixed(1) + '%';
  
  const customUncertainRatePct = (statsPayload.uncertain_rate || 0) * 100;
  document.getElementById('statUncertainRate').textContent = customUncertainRatePct.toFixed(1) + '%';

  // Populate row elements directly inside database logs tables body template
  const targetBodyNode = document.getElementById('dashboardLogBody');
  const emptyLogsStateNode = document.getElementById('dashboardEmptyLogState');

  if (!statsPayload.recent || statsPayload.recent.length === 0) {
    emptyLogsStateNode.style.display = 'block';
    targetBodyNode.innerHTML = '';
    return;
  }

  emptyLogsStateNode.style.display = 'none';
  targetBodyNode.innerHTML = statsPayload.recent.map(record => {
    const rawInferenceTimeStr = new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const computedAccuracyScore = (record.confidence * 100).toFixed(1);
    
    return `
      <tr class="hover:bg-slate-900/5">
        <td style="font-family:'JetBrains Mono'; color:#64748b; font-size:11px;">${rawInferenceTimeStr}</td>
        <td>
          <span class="pill-badge font-bold text-[10px] uppercase ${record.class === 'cat' ? 'text-orange-500 bg-orange-950/10' : 'text-cyan-400 bg-cyan-950/10'} px-2 py-0.5 rounded border border-slate-800">
            ${record.class === 'cat' ? '🐱 Cat' : '🐶 Dog'}
          </span>
        </td>
        <td>
          <div class="progress-bar-flex-wrapper">
            <div class="progress-track-bg">
              <div class="progress-track-fill ${record.class}" style="width: ${computedAccuracyScore}%;"></div>
            </div>
            <span class="progress-numerical-lbl">${computedAccuracyScore}%</span>
          </div>
        </td>
        <td style="font-family:'JetBrains Mono'; color:#64748b; font-size:11px;">/${record.source || 'predict'}</td>
        <td>
          <span class="pill-badge text-[9.5px] font-semibold ${record.uncertain ? 'text-amber-500 bg-amber-950/20 border-amber-800/30' : 'text-emerald-400 bg-emerald-950/20 border-emerald-800/30'} px-1.5 py-0.5 rounded border">
            ${record.uncertain ? '⚠️ Ambiguous' : '✓ Validated'}
          </span>
        </td>
      </tr>
    `;
  }).join('');
}

/* ---------- FETCH TELEMETRY LEDGER CORE LAYER ---------- */
async function fetchObservabilityLogsLedger() {
  try {
    const networkResponse = await fetch('/stats');
    if (!networkResponse.ok) throw new Error('Database logger analytics metrics retrieval error.');
    const dataRecordsPayload = await networkResponse.json();
    
    renderExecutiveTelemMetrics(dataRecordsPayload);
    constructPerformanceCharts(dataRecordsPayload);
  } catch (err) {
    console.error("Telemetry server logs disconnected. Simulating corporate monitoring workspace data layers...");
    
    // Simulate data structure if uvicorn is offline or preview is launched standalone
    const mockSimulatedStats = {
      total_predictions: 142,
      class_counts: { cat: 74, dog: 68 },
      avg_confidence: { cat: 0.884, dog: 0.865, overall: 0.874 },
      uncertain_count: 12,
      uncertain_rate: 0.0845,
      confidence_histogram: [2, 1, 4, 3, 5, 8, 12, 24, 38, 45],
      recent: [
        { timestamp: new Date().toISOString(), class: "cat", confidence: 0.942, uncertain: false, source: "predict" },
        { timestamp: new Date(Date.now() - 60000).toISOString(), class: "dog", confidence: 0.584, uncertain: true, source: "predict" },
        { timestamp: new Date(Date.now() - 120000).toISOString(), class: "dog", confidence: 0.891, uncertain: false, source: "predict-gradcam" },
        { timestamp: new Date(Date.now() - 180000).toISOString(), class: "cat", confidence: 0.452, uncertain: true, source: "predict" },
        { timestamp: new Date(Date.now() - 240000).toISOString(), class: "cat", confidence: 0.978, uncertain: false, source: "predict" }
      ]
    };
    
    renderExecutiveTelemMetrics(mockSimulatedStats);
    constructPerformanceCharts(mockSimulatedStats);
  }
}

/* ---------- RUNTIME INTERSECTION OBSERVER COUPLING ---------- */
function bindScrollAnchorHighlights() {
  const sections = document.querySelectorAll('section[id]');
  const navigationItems = document.querySelectorAll('.menu-nav-item');
  
  const scrollingLinkObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navigationItems.forEach(navLink => {
          navLink.classList.toggle('active', navLink.getAttribute('href') === `#${entry.target.id}`);
        });
      }
    });
  }, { threshold: 0.35, rootMargin: "0px 0px -20% 0px" });

  sections.forEach(secNode => scrollingLinkObserver.observe(secNode));
}

// Window lifecycle loader
window.onload = () => {
  fetchObservabilityLogsLedger();
  bindScrollAnchorHighlights();
};