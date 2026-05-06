// Fullscreen canvas drawing the green gaze dot at (x, y) screen pixels.

const DOT_RADIUS = 16;
const DOT_COLOR = "#00e676";
const RING_COLOR = "rgba(0, 230, 118, 0.45)";

let canvas = null;
let ctx = null;
let dpr = 1;

function resize() {
  if (!canvas) return;
  dpr = window.devicePixelRatio || 1;
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

export function initOverlay(canvasEl) {
  canvas = canvasEl;
  ctx = canvas.getContext("2d");
  resize();
  window.addEventListener("resize", resize);
}

export function drawDot(x, y) {
  if (!ctx) return;
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  // Outer translucent ring
  ctx.beginPath();
  ctx.arc(x, y, DOT_RADIUS * 2.2, 0, Math.PI * 2);
  ctx.fillStyle = RING_COLOR;
  ctx.fill();
  // Solid dot
  ctx.beginPath();
  ctx.arc(x, y, DOT_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = DOT_COLOR;
  ctx.fill();
}

export function clearOverlay() {
  if (!ctx) return;
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
}
