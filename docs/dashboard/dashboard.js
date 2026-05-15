// angr Rust engine perf dashboard
// Renders charts from data.json — the time-series file produced by
// tools/aggregate_bench_history.py.

const FEATURED = ["fauxware", "ais3_crackme", "csaw_wyvern"];

const META = document.getElementById("meta");
const BENCH_SELECT = document.getElementById("bench-select");
const METRIC_SELECT = document.getElementById("metric-select");
const DETAIL_TITLE = document.getElementById("detail-title");
const DETAIL_HINT = document.getElementById("detail-hint");

const featuredCharts = {};
let detailChart = null;
let DATA = null;

function setError(msg) {
  META.textContent = msg;
  META.classList.add("error");
}

function fmtTimestamp(ts) {
  // ISO Z → 'YYYY-MM-DD'
  return ts.slice(0, 10);
}

function pickMetric(entry, name, metric) {
  const row = entry.results[name];
  if (!row) return null;
  if (metric === "speedup") {
    const rt = row.rust_time;
    const pt = row.python_time;
    if (rt && pt && rt > 0) return pt / rt;
    return null;
  }
  const value = row[metric];
  return typeof value === "number" ? value : null;
}

function buildDataset(label, color, points) {
  return {
    label,
    data: points,
    borderColor: color,
    backgroundColor: color + "33",
    spanGaps: true,
    tension: 0.15,
    pointRadius: 3,
    pointHoverRadius: 5,
    fill: false,
  };
}

function commonOptions(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
      y: { title: { display: true, text: yLabel }, beginAtZero: false },
    },
    plugins: {
      legend: { position: "bottom" },
      tooltip: {
        callbacks: {
          afterLabel: (ctx) => {
            const entry = DATA.series[ctx.dataIndex];
            if (!entry) return "";
            return `commit ${entry.commit_short || entry.commit}`;
          },
        },
      },
    },
  };
}

function renderFeatured() {
  for (const name of FEATURED) {
    const canvas = document.getElementById(`featured-${name}`);
    if (!canvas) continue;
    const labels = DATA.series.map((e) => fmtTimestamp(e.timestamp));
    const points = DATA.series.map((e) => pickMetric(e, name, "rust_time"));
    const ctx = canvas.getContext("2d");
    if (featuredCharts[name]) featuredCharts[name].destroy();
    featuredCharts[name] = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [buildDataset(`${name} rust_time`, "#0969da", points)],
      },
      options: commonOptions("seconds"),
    });
  }
}

function renderDetail() {
  const name = BENCH_SELECT.value;
  const metric = METRIC_SELECT.value;
  if (!name) return;
  DETAIL_TITLE.textContent = `Detail — ${name} (${metric})`;

  const labels = DATA.series.map((e) => fmtTimestamp(e.timestamp));
  const points = DATA.series.map((e) => pickMetric(e, name, metric));
  const nonNull = points.filter((v) => v !== null);
  if (nonNull.length === 0) {
    DETAIL_HINT.textContent = `No samples for ${name}.${metric} in the current window.`;
  } else {
    const latest = nonNull[nonNull.length - 1];
    DETAIL_HINT.textContent = `Latest: ${latest.toFixed(3)} · samples: ${nonNull.length} of ${points.length}`;
  }

  const datasets = [buildDataset(`${name} ${metric}`, "#0969da", points)];
  if (metric === "rust_time") {
    // Overlay python_time for context if available.
    const pyPoints = DATA.series.map((e) => pickMetric(e, name, "python_time"));
    if (pyPoints.some((v) => v !== null)) {
      datasets.push(buildDataset(`${name} python_time`, "#cf222e", pyPoints));
    }
  }

  const ctx = document.getElementById("detail-chart").getContext("2d");
  if (detailChart) detailChart.destroy();
  detailChart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: commonOptions(metric),
  });
}

function populateBenchSelect(initial) {
  BENCH_SELECT.innerHTML = "";
  for (const name of DATA.benchmarks) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    BENCH_SELECT.appendChild(opt);
  }
  if (initial && DATA.benchmarks.includes(initial)) {
    BENCH_SELECT.value = initial;
  } else if (DATA.benchmarks.length > 0) {
    BENCH_SELECT.value = DATA.benchmarks[0];
  }
}

async function init() {
  try {
    const resp = await fetch("data.json", { cache: "no-store" });
    if (!resp.ok) {
      setError(`Failed to load data.json (HTTP ${resp.status}).`);
      return;
    }
    DATA = await resp.json();
  } catch (err) {
    setError(`Failed to load data.json: ${err.message}`);
    return;
  }

  if (!DATA.series || DATA.series.length === 0) {
    setError("No benchmark history available yet. The first nightly run will populate this dashboard.");
    return;
  }

  const first = DATA.series[0];
  const last = DATA.series[DATA.series.length - 1];
  META.textContent = `${DATA.series.length} points · ${fmtTimestamp(first.timestamp)} → ${fmtTimestamp(last.timestamp)} · generated ${DATA.generated}`;

  populateBenchSelect(FEATURED[0]);
  renderFeatured();
  renderDetail();

  BENCH_SELECT.addEventListener("change", renderDetail);
  METRIC_SELECT.addEventListener("change", renderDetail);
}

// Chart.js loads via `defer`; this script is also `defer`, so Chart is
// already defined by the time DOMContentLoaded fires.
document.addEventListener("DOMContentLoaded", init);
