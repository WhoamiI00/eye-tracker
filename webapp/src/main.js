import {
  computeHeadPose, lockSpheres, fitMapping, combinedGazeDir,
  gazeToScreen, yawPitchOf, isCalibrated, spheresLocked,
  reset as resetTracker,
} from "./tracker.js";
import {
  buildGrid, highlightCell, clearHighlight, cellCenter, setCalibTarget,
  getPreviewVideoEl,
} from "./grid.js";
import { initOverlay, drawDot, clearOverlay } from "./overlay.js";

const MEDIAPIPE_VERSION = "0.10.35";
const MODEL_URL = `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`;
const VISION_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

// 9-point calibration walk across the 8x5 grid. Numbers refer to 1-40 cell
// indices. Cell 8 is the webcam preview, so we use 7 instead for the top-right
// region.
const CALIB_POINTS = [1, 4, 7, 17, 21, 24, 33, 36, 40];
const DWELL_MS = 1500;          // total time on each target
const SAMPLE_WINDOW_MS = 1000;  // average gaze samples from the last 1s of dwell

const startScreen = document.getElementById("start-screen");
const startBtn = document.getElementById("start-button");
const instructionsScreen = document.getElementById("instructions-screen");
const instructionsBtn = document.getElementById("instructions-start-button");
const recalibBtn = document.getElementById("recalibrate-button");
const banner = document.getElementById("banner");
const gridEl = document.getElementById("grid");
const overlayEl = document.getElementById("overlay");
const videoEl = document.getElementById("video");

let faceLandmarker = null;
let stream = null;
let running = false;

// Calibration state machine
let calibIdx = -1;            // index into CALIB_POINTS, -1 = not started
let calibStartedAt = 0;       // performance.now() when current point started
let calibSamples = [];        // {yaw, pitch, screenX, screenY, weight} — the full set ever fitted
let pointSamples = [];        // {yaw, pitch} captured for the current target during its sample window
let latestYawPitch = null;    // most recent {yaw, pitch} — for click-to-correct

const CORRECTION_WEIGHT = 3;  // user corrections count 3x as much as walk-through samples
const STORAGE_KEY = "eyetracker.calibSamples.v1";

function loadSavedSamples() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Basic shape validation
    return parsed.filter(s =>
      typeof s.yaw === "number" && typeof s.pitch === "number" &&
      typeof s.screenX === "number" && typeof s.screenY === "number");
  } catch {
    return [];
  }
}

function saveSamples() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(calibSamples));
  } catch (err) {
    console.warn("could not persist calibration", err);
  }
}

export function clearSavedCalibration() {
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
}
window.clearSavedCalibration = clearSavedCalibration; // accessible from devtools console

async function loadFaceLandmarker() {
  const { FilesetResolver, FaceLandmarker } =
    await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(VISION_URL);
  return await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

function showInstructions() {
  startScreen.classList.add("hidden");
  instructionsScreen.classList.remove("hidden");
  instructionsBtn.disabled = false;
  instructionsBtn.textContent = "Start tracking";
}

async function startTracking() {
  instructionsBtn.disabled = true;
  instructionsBtn.textContent = "Loading model...";

  try {
    if (!faceLandmarker) faceLandmarker = await loadFaceLandmarker();
  } catch (err) {
    console.error(err);
    showError(`Could not load the face model: ${err.message}`);
    return;
  }

  instructionsBtn.textContent = "Asking for camera...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    console.error(err);
    showError(`Camera access denied: ${err.message}`);
    return;
  }

  videoEl.srcObject = stream;
  await new Promise((resolve) => {
    if (videoEl.readyState >= 2) resolve();
    else videoEl.addEventListener("loadeddata", resolve, { once: true });
  });
  await videoEl.play();

  startScreen.classList.add("hidden");
  instructionsScreen.classList.add("hidden");
  gridEl.classList.remove("hidden");
  overlayEl.classList.remove("hidden");
  banner.classList.remove("hidden");
  recalibBtn.classList.remove("hidden");

  buildGrid(gridEl, handleCorrection);
  initOverlay(overlayEl);

  // The grid creates a <video> inside cell 8 — feed it the same stream so the
  // user sees their face there.
  const preview = getPreviewVideoEl();
  if (preview) {
    preview.srcObject = stream;
    preview.play().catch(() => { /* autoplay-suppression edge case; harmless */ });
  }

  running = true;
  calibIdx = -1;
  calibStartedAt = 0;
  calibSamples = loadSavedSamples();
  pointSamples = [];
  latestYawPitch = null;
  resetTracker();
  requestAnimationFrame(loop);
}

// Click-to-correct: user clicked cell `n` while looking at it. Add a high-
// weight sample mapping the latest gaze direction to that cell's center,
// then re-fit the affine map.
function handleCorrection(n, screenX, screenY) {
  if (!isCalibrated()) return; // ignore clicks during the initial walkthrough
  if (!latestYawPitch) return;
  calibSamples.push({
    yaw: latestYawPitch.yaw,
    pitch: latestYawPitch.pitch,
    screenX, screenY,
    weight: CORRECTION_WEIGHT,
  });
  try {
    fitMapping(calibSamples);
    saveSamples();
    flashBanner(`learned cell ${n} (${calibSamples.length} samples)`);
  } catch (err) {
    console.error(err);
  }
}

let bannerTimer = null;
function flashBanner(text) {
  banner.textContent = text;
  banner.classList.remove("hidden");
  if (bannerTimer) clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { banner.classList.add("hidden"); }, 1200);
}

function showError(msg) {
  instructionsBtn.disabled = false;
  instructionsBtn.textContent = "Start tracking";
  let err = document.getElementById("err");
  if (!err) {
    err = document.createElement("p");
    err.id = "err";
    err.style.color = "#ff5252";
    err.style.fontSize = "15px";
    err.style.marginTop = "16px";
    instructionsBtn.parentElement.appendChild(err);
  }
  err.textContent = msg;
}

function stopTracking() {
  running = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  videoEl.srcObject = null;
  banner.classList.add("hidden");
  gridEl.classList.add("hidden");
  overlayEl.classList.add("hidden");
  instructionsScreen.classList.add("hidden");
  recalibBtn.classList.add("hidden");
  clearHighlight();
  setCalibTarget(null);
  clearOverlay();
  startScreen.classList.remove("hidden");
  startBtn.disabled = false;
  startBtn.textContent = "Start tracking";
  instructionsBtn.disabled = false;
  instructionsBtn.textContent = "Start tracking";
}

function loop(timestamp) {
  if (!running) return;
  requestAnimationFrame(loop);

  if (videoEl.readyState < 2) return;

  const result = faceLandmarker.detectForVideo(videoEl, timestamp);
  const lmList = result?.faceLandmarks?.[0];
  if (!lmList) {
    banner.textContent = "Looking for your face...";
    return;
  }

  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  if (w === 0 || h === 0) return;

  const headPose = computeHeadPose(lmList, w, h);

  if (!isCalibrated()) {
    runCalibration(lmList, headPose, w, h, timestamp);
    return;
  }

  const dir = combinedGazeDir(lmList, headPose, w, h);
  if (!dir) return;
  latestYawPitch = yawPitchOf(dir);
  const { x, y } = gazeToScreen(dir);
  highlightCell(x, y);
  drawDot(x, y);
}

function runCalibration(lmList, headPose, w, h, now) {
  // First call: lock eye spheres. If we have saved samples from a previous
  // session, skip the 9-point walkthrough entirely and use those.
  if (calibIdx < 0) {
    lockSpheres(lmList, headPose, w, h);
    if (calibSamples.length >= 3) {
      try {
        fitMapping(calibSamples);
        flashBanner(`restored calibration (${calibSamples.length} samples) — click any number to refine`);
        return;
      } catch (err) {
        console.warn("saved calibration was unusable, falling back to walkthrough", err);
        calibSamples = [];
      }
    }
    calibIdx = 0;
    calibStartedAt = now;
    pointSamples = [];
    setCalibTarget(CALIB_POINTS[0]);
  }

  if (!spheresLocked()) return;

  const dir = combinedGazeDir(lmList, headPose, w, h);
  if (!dir) return;

  const elapsed = now - calibStartedAt;
  const targetN = CALIB_POINTS[calibIdx];
  const remaining = Math.max(0, (DWELL_MS - elapsed) / 1000);
  banner.textContent =
    `Calibration ${calibIdx + 1}/${CALIB_POINTS.length}: look at the highlighted ${targetN}  (${remaining.toFixed(1)}s)`;

  // Collect samples during the last SAMPLE_WINDOW_MS of dwell — gives the
  // user time to settle their gaze first.
  if (elapsed >= DWELL_MS - SAMPLE_WINDOW_MS) {
    pointSamples.push(yawPitchOf(dir));
  }

  if (elapsed >= DWELL_MS) {
    // Average samples for this point into a single (yaw, pitch) -> (screenX, screenY)
    if (pointSamples.length > 0) {
      const meanYaw = pointSamples.reduce((s, v) => s + v.yaw, 0) / pointSamples.length;
      const meanPitch = pointSamples.reduce((s, v) => s + v.pitch, 0) / pointSamples.length;
      const { x, y } = cellCenter(targetN);
      calibSamples.push({ yaw: meanYaw, pitch: meanPitch, screenX: x, screenY: y });
    }

    calibIdx++;
    calibStartedAt = now;
    pointSamples = [];

    if (calibIdx >= CALIB_POINTS.length) {
      // All points captured — fit the affine map and finish calibration.
      setCalibTarget(null);
      try {
        fitMapping(calibSamples);
        saveSamples();
        banner.classList.add("hidden");
      } catch (err) {
        console.error(err);
        banner.textContent = "Calibration failed — please reload and try again.";
      }
      return;
    }

    setCalibTarget(CALIB_POINTS[calibIdx]);
  }
}

function recalibrate() {
  if (!running) return;
  clearSavedCalibration();
  resetTracker();
  calibIdx = -1;
  calibStartedAt = 0;
  calibSamples = [];
  pointSamples = [];
  latestYawPitch = null;
  clearHighlight();
  setCalibTarget(null);
  clearOverlay();
  banner.classList.remove("hidden");
  banner.textContent = "Recalibrating...";
}

window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && running) stopTracking();
});

startBtn.addEventListener("click", showInstructions);
instructionsBtn.addEventListener("click", startTracking);
recalibBtn.addEventListener("click", recalibrate);
