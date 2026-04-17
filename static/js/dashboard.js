document.addEventListener("DOMContentLoaded", async () => {
  const res  = await fetch("/api/results");
  const data = await res.json();
  const modelNames = ["Logistic Regression", "Decision Tree", "Random Forest"];
  const colors = {"Logistic Regression":"#6366f1","Decision Tree":"#f59e0b","Random Forest":"#10b981"};
  const best = data.best_model;

  // Best Banner
  document.getElementById("best-name").textContent = best;
  const bm = data[best];
  document.getElementById("best-metrics").innerHTML = `
    <div class="best-metric"><div class="best-metric-val">${bm.accuracy}%</div><div class="best-metric-label">Accuracy</div></div>
    <div class="best-metric"><div class="best-metric-val">${bm.f1}%</div><div class="best-metric-label">F1 Score</div></div>
    <div class="best-metric"><div class="best-metric-val">${bm.auc}%</div><div class="best-metric-label">AUC</div></div>
  `;

  // Metric Cards
  const grid = document.getElementById("metric-cards");
  modelNames.forEach(name => {
    const m = data[name];
    const isBest = name === best;
    const card = document.createElement("div");
    card.className = "metric-model-card" + (isBest ? " best-card" : "");
    const metrics = [
      ["Accuracy",  m.accuracy],
      ["Precision", m.precision],
      ["Recall",    m.recall],
      ["F1 Score",  m.f1],
      ["AUC",       m.auc],
    ];
    card.innerHTML = `
      <div class="metric-model-name">
        ${name}
        ${isBest ? '<span class="best-badge">🏆 Best</span>' : ''}
      </div>
      ${metrics.map(([label, val]) => `
        <div class="metric-row">
          <span class="metric-name">${label}</span>
          <span class="metric-val" style="color:${colors[name]}">${val}%</span>
        </div>
        <div class="metric-bar-wrap">
          <div class="metric-bar" style="width:${val}%; background:${colors[name]}"></div>
        </div>
      `).join("")}
      <div style="font-size:0.8rem;color:var(--muted);margin-top:0.5rem">⏱ Train time: ${m.train_time}s</div>
    `;
    grid.appendChild(card);
  });

  // Results Table
  const tbody = document.getElementById("results-tbody");
  modelNames.forEach(name => {
    const m = data[name];
    const isBest = name === best;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="${isBest ? 'td-best' : ''}">${isBest ? "🏆 " : ""}${name}</td>
      <td>${m.accuracy}%</td>
      <td>${m.precision}%</td>
      <td>${m.recall}%</td>
      <td>${m.f1}%</td>
      <td>${m.auc}%</td>
      <td>${m.train_time}s</td>
    `;
    tbody.appendChild(row);
  });
});