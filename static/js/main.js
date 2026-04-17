// Animate pixel grids in hero
document.addEventListener("DOMContentLoaded", () => {
  const realColors = ["#4ade80","#22c55e","#86efac","#dcfce7","#bbf7d0","#166534","#15803d","#14532d"];
  const fakeColors = ["#f87171","#ef4444","#fca5a5","#fee2e2","#fecaca","#991b1b","#b91c1c","#dc2626"];

  function fillGrid(selector, colors) {
    const grid = document.querySelector(selector);
    if (!grid) return;
    for (let i = 0; i < 64; i++) {
      const cell = document.createElement("div");
      cell.style.background = colors[Math.floor(Math.random() * colors.length)];
      grid.appendChild(cell);
    }
  }

  fillGrid(".real-grid", realColors);
  fillGrid(".fake-grid", fakeColors);
});