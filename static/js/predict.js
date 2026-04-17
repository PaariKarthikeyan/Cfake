const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const predictBtn = document.getElementById("predict-btn");
const previewWrap = document.getElementById("preview-wrap");
const previewImg = document.getElementById("preview-img");
const clearBtn = document.getElementById("clear-btn");
const loader = document.getElementById("loader");
const placeholder = document.getElementById("results-placeholder");
const content = document.getElementById("results-content");
let selectedFile = null;

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewWrap.style.display = "flex";
  dropZone.style.display = "none";
  predictBtn.disabled = false;
}

clearBtn.addEventListener("click", () => {
  selectedFile = null;
  dropZone.style.display = "block";
  previewWrap.style.display = "none";
  predictBtn.disabled = true;
  content.style.display = "none";
  placeholder.style.display = "block";
  fileInput.value = "";
});

// ─── Verdict meta ──────────────────────────────────────────────────────────
const verdictMeta = {
  "REAL": { color: "var(--green)", icon: "✅", cls: "verdict-real" },
  "FAKE": { color: "var(--red)", icon: "🤖", cls: "verdict-fake" },
  "UNCERTAIN": { color: "var(--amber)", icon: "🤔", cls: "verdict-uncertain" },
};

const modelColors = {
  "Logistic Regression": "#6366f1",
  "Decision Tree": "#f59e0b",
  "Random Forest": "#10b981",
};

const modelIcons = {
  "Logistic Regression": "📈",
  "Decision Tree": "🌲",
  "Random Forest": "🌳",
};

// ─── Predict ───────────────────────────────────────────────────────────────
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  loader.style.display = "block";
  placeholder.style.display = "none";
  content.style.display = "none";

  const form = new FormData();
  form.append("image", selectedFile);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    loader.style.display = "none";

    if (data.error) { alert(data.error); return; }

    const meta = verdictMeta[data.verdict];

    // ─── Primary: Decision Tree verdict ──────────────────────────────────
    document.getElementById("best-result").innerHTML = `
      // ✅ NEW
      <div class="best-result-label">
        🗳️ Majority Vote — ${data.votes.real} REAL · ${data.votes.fake} FAKE · ${data.votes.uncertain} Uncertain
      </div>
      <div class="best-result-verdict ${meta.cls}">
        ${meta.icon} ${data.verdict_msg}
      </div>

      <div style="margin-top:1rem">
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
          <span style="color:var(--green)">REAL — ${data.real_prob}%</span>
          <span style="color:var(--red)">FAKE — ${data.fake_prob}%</span>
        </div>
        <div style="background:var(--border);height:10px;border-radius:5px;overflow:hidden;display:flex">
          <div style="width:${data.real_prob}%;background:var(--green);transition:width 1s"></div>
          <div style="width:${data.fake_prob}%;background:var(--red);transition:width 1s"></div>
        </div>
      </div>

      <div style="margin-top:0.8rem;font-size:0.82rem;color:var(--muted)">
        Confidence: <strong style="color:${meta.color}">${data.confidence}%</strong>
        &nbsp;|&nbsp; Certainty: <strong style="color:${meta.color}">${data.certainty.toUpperCase()}</strong>
      </div>

      ${data.certainty === "low" ? `
        <div style="margin-top:0.8rem;background:rgba(245,158,11,0.1);
             border:1px solid rgba(245,158,11,0.3);border-radius:8px;
             padding:0.7rem 1rem;font-size:0.82rem;color:var(--amber)">
          ⚠️ Score is between 30–70%. Try a clearer image for a more confident result.
        </div>` : ""}
    `;

    // ─── Secondary: All 3 models breakdown ───────────────────────────────
    const allDiv = document.getElementById("all-predictions");
    allDiv.innerHTML = `
      <div style="font-size:0.85rem;font-weight:700;color:var(--muted);
                  text-transform:uppercase;letter-spacing:1px;
                  margin:1.5rem 0 0.8rem">
        All Models Breakdown
      </div>
    `;

    for (const [name, pred] of Object.entries(data.all_predictions)) {
      const vm = verdictMeta[pred.verdict];
      const mColor = modelColors[name];
      const mIcon = modelIcons[name];
      const isDT = name === "Decision Tree";

      allDiv.innerHTML += `
        <div class="pred-card ${isDT ? 'pred-card-primary' : ''}">

          <!-- Left: model name + prob bar -->
          <div style="flex:1">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
              <span style="font-size:1.1rem">${mIcon}</span>
              <span class="pred-model-name" style="color:${mColor}">${name}</span>
              ${pred.verdict === data.verdict ? '<span class="best-badge" style="background:var(--indigo)">✓ Agrees</span>' : ""}
            </div>

            <!-- Prob bar -->
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:3px">
              <span style="color:var(--green)">REAL ${pred.real_prob}%</span>
              <span style="color:var(--red)">FAKE ${pred.fake_prob}%</span>
            </div>
            <div style="background:var(--border);height:7px;border-radius:4px;overflow:hidden;display:flex">
              <div style="width:${pred.real_prob}%;background:var(--green);transition:width 1s"></div>
              <div style="width:${pred.fake_prob}%;background:var(--red);transition:width 1s"></div>
            </div>
          </div>

          <!-- Right: verdict badge -->
          <div style="text-align:right;margin-left:1.2rem;flex-shrink:0">
            <div class="pred-verdict" style="color:${vm.color};font-size:1.1rem">${vm.icon}</div>
            <div class="pred-verdict" style="color:${vm.color}">${pred.verdict}</div>
            <div class="pred-conf">${pred.confidence}%</div>
          </div>

        </div>
      `;
    }

    content.style.display = "block";

  } catch (e) {
    loader.style.display = "none";
    alert("Prediction failed. Make sure the Flask server is running.");
  }
});