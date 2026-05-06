// Port of Webcam3DTracker/MonitorTracking.py — only the gaze->screen-pixel
// math, none of the visualization. The MediaPipe Web FaceLandmarker emits the
// same 478-point face mesh as the Python MediaPipe FaceMesh, so landmark
// indices and the entire math chain transfer directly.

import {
  vec3, add, sub, scale, dot, cross, length, normalize,
  matVec, matTVec, col, det3, eigSymmetric3, meanVec3, covariance3,
} from "./linalg.js";

// MediaPipe FaceMesh landmark indices used by the Python tracker.
const NOSE_INDICES = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                     461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                     3, 248];
const LEFT_IRIS_IDX = 468;
const RIGHT_IRIS_IDX = 473;
const CHIN_IDX = 152;
const FOREHEAD_IDX = 10;

const FILTER_LENGTH = 10;
const BASE_RADIUS = 20;          // pixels at calibration distance, matches Python

// Sliding window of recent gaze directions for smoothing
const gazeHistory = [];

// State that persists across frames
const state = {
  // Eye-sphere calibration
  leftLocked: false,
  rightLocked: false,
  leftSphereLocalOffset: null,    // vec3 in head-local frame
  rightSphereLocalOffset: null,
  leftCalibrationNoseScale: 1,
  rightCalibrationNoseScale: 1,
  // Stable PCA orientation reference
  refNoseRotation: null,
  // Final per-eye world spheres for the current frame (after calibration)
  sphereWorldL: null,
  sphereWorldR: null,
  // Affine map: screenX = ax*yaw + bx*pitch + cx,  screenY = ay*yaw + by*pitch + cy
  mapX: null,  // [a, b, c]
  mapY: null,
};

// Convert MediaPipe normalized landmark (.x, .y in [0,1], .z relative to frame
// width) into the Python tracker's "world units" frame (x*w, y*h, z*w).
function lmToWorld(lm, w, h) {
  return vec3(lm.x * w, lm.y * h, lm.z * w);
}

// Compute average pairwise distance — robust scale measure used to track
// distance changes after calibration.
export function computeScale(points) {
  let total = 0;
  let count = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      total += length(sub(points[i], points[j]));
      count++;
    }
  }
  return count > 0 ? total / count : 1;
}

// PCA-based head pose from nose-region landmarks.
// Returns { headCenter: vec3, R: mat3 (row-major), nosePoints: vec3[] }
export function computeHeadPose(landmarks, w, h) {
  const pts = NOSE_INDICES.map((i) => lmToWorld(landmarks[i], w, h));
  const center = meanVec3(pts);
  const cov = covariance3(pts, center);
  const { vectors } = eigSymmetric3(cov);

  // Stabilize sign of each eigenvector against the previous frame's reference
  // so axes don't flip when the eigendecomposition picks the opposite sign.
  if (state.refNoseRotation == null) {
    state.refNoseRotation = new Float64Array(vectors);
  } else {
    const ref = state.refNoseRotation;
    for (let cIdx = 0; cIdx < 3; cIdx++) {
      let dotSign = 0;
      for (let r = 0; r < 3; r++) dotSign += vectors[r * 3 + cIdx] * ref[r * 3 + cIdx];
      if (dotSign < 0) {
        for (let r = 0; r < 3; r++) vectors[r * 3 + cIdx] *= -1;
      }
    }
    state.refNoseRotation = new Float64Array(vectors);
  }

  return { headCenter: center, R: vectors, nosePoints: pts };
}

// Lock both eye spheres at the current head pose. Same math as
// do_calibration() in the Python tracker. Run once at the *start* of the
// multi-point calibration, when the user is settled on the first target.
export function lockSpheres(landmarks, headPose, w, h) {
  const { headCenter, R, nosePoints } = headPose;
  const irisL = lmToWorld(landmarks[LEFT_IRIS_IDX], w, h);
  const irisR = lmToWorld(landmarks[RIGHT_IRIS_IDX], w, h);

  const noseScale = computeScale(nosePoints);
  const cameraDirLocal = matTVec(R, vec3(0, 0, 1));
  const radiusOffset = scale(cameraDirLocal, BASE_RADIUS);

  state.leftSphereLocalOffset = add(matTVec(R, sub(irisL, headCenter)), radiusOffset);
  state.rightSphereLocalOffset = add(matTVec(R, sub(irisR, headCenter)), radiusOffset);
  state.leftCalibrationNoseScale = noseScale;
  state.rightCalibrationNoseScale = noseScale;
  state.leftLocked = true;
  state.rightLocked = true;
  // Clear the smoothing window so post-calibration samples start fresh.
  gazeHistory.length = 0;
}

// Solve a 3-parameter weighted least-squares system for
//   screen = a*yaw + b*pitch + c
// across all calibration samples. Each axis is fit independently. Each sample
// may carry a `weight` (defaults to 1). The resulting fit minimizes sum
// w_i * (predicted - actual)^2.
export function fitMapping(samples) {
  if (samples.length < 3) {
    throw new Error("need at least 3 calibration samples to fit the mapping");
  }
  let s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
  let bx0 = 0, bx1 = 0, bx2 = 0;
  let by0 = 0, by1 = 0, by2 = 0;
  for (const s of samples) {
    const w = s.weight ?? 1;
    const y = s.yaw, p = s.pitch;
    s00 += w * y * y;     s01 += w * y * p;     s02 += w * y;
    s11 += w * p * p;     s12 += w * p;
                          s22 += w;
    bx0 += w * y * s.screenX; bx1 += w * p * s.screenX; bx2 += w * s.screenX;
    by0 += w * y * s.screenY; by1 += w * p * s.screenY; by2 += w * s.screenY;
  }
  const M = [
    [s00, s01, s02],
    [s01, s11, s12],
    [s02, s12, s22],
  ];
  state.mapX = solve3(M, [bx0, bx1, bx2]);
  state.mapY = solve3(M, [by0, by1, by2]);
}

function solve3(M, b) {
  // Gauss elimination with partial pivot on a 3x3 system. Mutates copies.
  const A = M.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < 3; i++) {
    let piv = i;
    for (let r = i + 1; r < 3; r++) {
      if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    }
    if (piv !== i) [A[i], A[piv]] = [A[piv], A[i]];
    const d = A[i][i];
    if (Math.abs(d) < 1e-12) throw new Error("singular calibration system");
    for (let j = i; j < 4; j++) A[i][j] /= d;
    for (let r = 0; r < 3; r++) {
      if (r === i) continue;
      const f = A[r][i];
      for (let j = i; j < 4; j++) A[r][j] -= f * A[i][j];
    }
  }
  return [A[0][3], A[1][3], A[2][3]];
}

// Compute the smoothed combined gaze direction for the current frame.
// Returns null if not yet calibrated.
export function combinedGazeDir(landmarks, headPose, w, h) {
  if (!state.leftLocked || !state.rightLocked) return null;

  const { headCenter, R, nosePoints } = headPose;
  const irisL = lmToWorld(landmarks[LEFT_IRIS_IDX], w, h);
  const irisR = lmToWorld(landmarks[RIGHT_IRIS_IDX], w, h);

  // Scale-aware sphere positions: as the user moves closer/further, scale the
  // local offset by the ratio of current to calibration nose scale.
  const noseScaleNow = computeScale(nosePoints);
  const ratioL = noseScaleNow / state.leftCalibrationNoseScale;
  const ratioR = noseScaleNow / state.rightCalibrationNoseScale;
  const sphereL = add(headCenter, matVec(R, scale(state.leftSphereLocalOffset, ratioL)));
  const sphereR = add(headCenter, matVec(R, scale(state.rightSphereLocalOffset, ratioR)));
  state.sphereWorldL = sphereL;
  state.sphereWorldR = sphereR;

  const lDir = normalize(sub(irisL, sphereL));
  const rDir = normalize(sub(irisR, sphereR));
  const raw = normalize(add(lDir, rDir));

  gazeHistory.push(raw);
  if (gazeHistory.length > FILTER_LENGTH) gazeHistory.shift();

  const sum = vec3();
  for (const g of gazeHistory) { sum[0] += g[0]; sum[1] += g[1]; sum[2] += g[2]; }
  return normalize(sum);
}

// Compute raw yaw/pitch (in degrees) for a gaze direction, before applying
// calibration offsets. Sign convention matches MonitorTracking.py:411-415.
function rawYawPitch(dir) {
  const REF = vec3(0, 0, -1); // camera looking down -Z
  const xz = normalize(vec3(dir[0], 0, dir[2]));
  let yawRad = Math.acos(Math.max(-1, Math.min(1, dot(REF, xz))));
  if (dir[0] < 0) yawRad = -yawRad;

  const yz = normalize(vec3(0, dir[1], dir[2]));
  let pitchRad = Math.acos(Math.max(-1, Math.min(1, dot(REF, yz))));
  if (dir[1] > 0) pitchRad = -pitchRad;

  let yawDeg = (yawRad * 180) / Math.PI;
  const pitchDeg = (pitchRad * 180) / Math.PI;
  // Python: flip yaw sign so right-of-camera is positive
  yawDeg = -yawDeg;
  return { rawYawDeg: yawDeg, rawPitchDeg: pitchDeg };
}

// Returns { yaw, pitch } in degrees for a gaze direction. Used by the
// calibration loop to collect per-target samples.
export function yawPitchOf(combinedDir) {
  const { rawYawDeg, rawPitchDeg } = rawYawPitch(combinedDir);
  return { yaw: rawYawDeg, pitch: rawPitchDeg };
}

// Apply the fitted affine mapping to convert a gaze direction to screen px.
export function gazeToScreen(combinedDir) {
  const { yaw, pitch } = yawPitchOf(combinedDir);
  const W = window.innerWidth;
  const H = window.innerHeight;
  if (!state.mapX || !state.mapY) return { x: W / 2, y: H / 2 };
  let x = state.mapX[0] * yaw + state.mapX[1] * pitch + state.mapX[2];
  let y = state.mapY[0] * yaw + state.mapY[1] * pitch + state.mapY[2];
  x = Math.max(8, Math.min(W - 8, x));
  y = Math.max(8, Math.min(H - 8, y));
  return { x, y };
}

export function spheresLocked() {
  return state.leftLocked && state.rightLocked;
}

export function isCalibrated() {
  return state.leftLocked && state.rightLocked && state.mapX != null && state.mapY != null;
}

export function reset() {
  state.leftLocked = false;
  state.rightLocked = false;
  state.leftSphereLocalOffset = null;
  state.rightSphereLocalOffset = null;
  state.refNoseRotation = null;
  state.mapX = null;
  state.mapY = null;
  gazeHistory.length = 0;
}
