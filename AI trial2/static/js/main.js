// main.js
// Uses Chart.js and chartjs-plugin-datalabels (loaded from HTML)

const videoEl = document.getElementById('videoFeed');
const filterSelect = document.getElementById('filterSelect');
const themeToggle = document.getElementById('themeToggle');
const dominantText = document.getElementById('dominantText');
const quoteBox = document.getElementById('quoteBox');
const countsBrief = document.getElementById('countsBrief');
const summaryBtn = document.getElementById('summaryBtn');

const summaryModal = document.getElementById('summaryModal');
const closeSummary = document.getElementById('closeSummary');
const summaryText = document.getElementById('summaryText');

let lastDominant = null;

// Setup charts
const pieCtx = document.getElementById('pieChart').getContext('2d');
const timelineCtx = document.getElementById('timelineChart').getContext('2d');

const pieChart = new Chart(pieCtx, {
  type: 'pie',
  data: { labels: [], datasets: [{ data: [], backgroundColor: ['#ff6384', '#36a2eb', '#ffce56', '#4caf50', '#9966ff', '#ff9f40', '#e91e63'] }] },
  options: {
    responsive: true,
    plugins: {
      legend: { position: 'bottom' },
      datalabels: {
        color: '#fff', font: { weight: '600', size: 12 },
        formatter: (value, ctx) => {
          let label = ctx.chart.data.labels[ctx.dataIndex] || '';
          return label + '\n' + value;
        }
      }
    }
  },
  plugins: [ChartDataLabels]
});

const timelineChart = new Chart(timelineCtx, {
  type: 'bar',
  data: { labels: [], datasets: [{ label: 'Count', data: [], backgroundColor: '#36a2eb' }] },
  options: {
    responsive: true,
    scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
    plugins: { legend: { display: false } }
  }
});

// Theme handling
if (localStorage.getItem('theme') === 'dark') document.body.classList.add('dark');
themeToggle.addEventListener('click', () => {
  document.body.classList.toggle('dark');
  localStorage.setItem('theme', document.body.classList.contains('dark') ? 'dark' : 'light');
});

// Filter handling: change video src param so backend applies filter
function setFilter(filter) {
  const url = new URL(videoEl.src, window.location.origin);
  url.searchParams.set('filter', filter);
  videoEl.src = url.toString();
}
filterSelect.addEventListener('change', (e) => setFilter(e.target.value));

// Fetch emotion data and update UI
async function fetchEmotionData() {
  try {
    const res = await fetch('/emotion_data');
    const payload = await res.json();

    // counts may be an object with emotion->count
    const countsObj = payload.counts || {};
    const labels = Object.keys(countsObj);
    const values = Object.values(countsObj);

    // update pie
    pieChart.data.labels = labels;
    pieChart.data.datasets[0].data = values;
    pieChart.update();

    // dominant & quote
    const dom = payload.dominant || 'neutral';
    dominantText.textContent = dom;
    quoteBox.textContent = payload.quote || '';

    // small brief
    let brief = labels.map((l,i) => `${l}: ${values[i]}`).join(' • ');
    countsBrief.textContent = brief || 'Detected: —';

    // timeline (group by timestamp seconds)
    const tl = payload.timeline || [];
    const grouped = {};
    tl.forEach(item => {
      const t = item.t;
      grouped[t] = (grouped[t] || 0) + 1;
    });
    const tlLabels = Object.keys(grouped);
    const tlValues = Object.values(grouped);
    timelineChart.data.labels = tlLabels;
    timelineChart.data.datasets[0].data = tlValues;
    timelineChart.update();

    // highlight change (no sound)
    if (lastDominant && lastDominant !== dom) {
      // no-sound behavior: small flash on pill
      const pill = document.getElementById('dominantPill');
      pill.animate([{ transform: 'scale(1.0)' }, { transform: 'scale(1.05)' }, { transform: 'scale(1.0)' }], { duration: 300 });
    }
    lastDominant = dom;

  } catch (e) {
    // keep UI alive
    // console.error(e);
  }
}

// initial call + interval
fetchEmotionData();
const INTERVAL_MS = 1200;
setInterval(fetchEmotionData, INTERVAL_MS);

// Session summary button
summaryBtn.addEventListener('click', async () => {
  try {
    const res = await fetch('/session_summary');
    const payload = await res.json();
    summaryText.textContent = JSON.stringify(payload, null, 2);
    summaryModal.classList.remove('hidden');
  } catch (e) {
    summaryText.textContent = 'Error loading summary';
    summaryModal.classList.remove('hidden');
  }
});
closeSummary.addEventListener('click', () => summaryModal.classList.add('hidden'));
summaryModal.addEventListener('click', (e) => { if (e.target === summaryModal) summaryModal.classList.add('hidden'); });
