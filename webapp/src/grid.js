// Builds the 1-40 number grid and toggles the .gaze-target class on the cell
// under a given (x, y) screen pixel. Cell 8 hosts a live webcam preview
// instead of a number.

const COLS = 8;
const ROWS = 5;
export const PREVIEW_CELL_INDEX = 7; // 0-based index of cell #8

let cells = [];
let gridEl = null;
let previewVideoEl = null;
let lastIdx = -1;
let onCorrect = null;

export function buildGrid(rootEl, onCorrectFn) {
  gridEl = rootEl;
  gridEl.innerHTML = "";
  cells = [];
  previewVideoEl = null;
  onCorrect = onCorrectFn || null;
  for (let i = 0; i < COLS * ROWS; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    if (i === PREVIEW_CELL_INDEX) {
      cell.classList.add("preview-cell");
      const v = document.createElement("video");
      v.autoplay = true;
      v.playsInline = true;
      v.muted = true;
      cell.appendChild(v);
      previewVideoEl = v;
    } else {
      cell.textContent = String(i + 1);
      const n = i + 1;
      cell.addEventListener("click", () => {
        if (!onCorrect) return;
        const { x, y } = cellCenter(n);
        onCorrect(n, x, y);
        // Brief visual confirmation
        cell.classList.add("learned");
        setTimeout(() => cell.classList.remove("learned"), 600);
      });
    }
    gridEl.appendChild(cell);
    cells.push(cell);
  }
  lastIdx = -1;
}

export function getPreviewVideoEl() {
  return previewVideoEl;
}

export function highlightCell(x, y) {
  if (!gridEl) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (w <= 0 || h <= 0) return;

  const col = Math.min(COLS - 1, Math.max(0, Math.floor((x / w) * COLS)));
  const row = Math.min(ROWS - 1, Math.max(0, Math.floor((y / h) * ROWS)));
  const idx = row * COLS + col;
  if (idx === lastIdx) return;
  if (lastIdx >= 0 && cells[lastIdx]) cells[lastIdx].classList.remove("gaze-target");
  if (idx !== PREVIEW_CELL_INDEX && cells[idx]) cells[idx].classList.add("gaze-target");
  lastIdx = idx;
}

export function clearHighlight() {
  if (lastIdx >= 0 && cells[lastIdx]) cells[lastIdx].classList.remove("gaze-target");
  lastIdx = -1;
}

// Pixel center of cell number `n` (1..40), used to drive calibration prompts.
export function cellCenter(n) {
  const idx = n - 1;
  const col = idx % COLS;
  const row = Math.floor(idx / COLS);
  const w = window.innerWidth;
  const h = window.innerHeight;
  return {
    x: (col + 0.5) * (w / COLS),
    y: (row + 0.5) * (h / ROWS),
  };
}

// Visually pulse a calibration cell (without using the regular gaze highlight).
export function setCalibTarget(n) {
  for (const c of cells) c.classList.remove("calib-target");
  if (n == null) return;
  if (n - 1 === PREVIEW_CELL_INDEX) return; // never highlight the webcam cell
  const cell = cells[n - 1];
  if (cell) cell.classList.add("calib-target");
}
