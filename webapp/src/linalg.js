// Tiny 3D linear algebra helpers. Operate on plain Float64Array(3) for vectors
// and Float64Array(9) row-major for 3x3 matrices.

export const vec3 = (x = 0, y = 0, z = 0) => Float64Array.of(x, y, z);

export const add = (a, b) => vec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
export const sub = (a, b) => vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
export const scale = (a, s) => vec3(a[0] * s, a[1] * s, a[2] * s);
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => vec3(
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
);
export const length = (a) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a) => {
  const n = length(a);
  return n > 1e-9 ? scale(a, 1 / n) : vec3();
};

// Multiply 3x3 matrix (row-major) by 3-vector: M @ v
export const matVec = (M, v) => vec3(
  M[0] * v[0] + M[1] * v[1] + M[2] * v[2],
  M[3] * v[0] + M[4] * v[1] + M[5] * v[2],
  M[6] * v[0] + M[7] * v[1] + M[8] * v[2],
);

// Transpose-multiply: M^T @ v
export const matTVec = (M, v) => vec3(
  M[0] * v[0] + M[3] * v[1] + M[6] * v[2],
  M[1] * v[0] + M[4] * v[1] + M[7] * v[2],
  M[2] * v[0] + M[5] * v[1] + M[8] * v[2],
);

// Extract column c (0..2) of a 3x3 row-major matrix
export const col = (M, c) => vec3(M[c], M[3 + c], M[6 + c]);

// Determinant of a 3x3
export const det3 = (M) =>
  M[0] * (M[4] * M[8] - M[5] * M[7])
  - M[1] * (M[3] * M[8] - M[5] * M[6])
  + M[2] * (M[3] * M[7] - M[4] * M[6]);

// Symmetric 3x3 eigendecomposition via Jacobi rotation. Returns
// { values: [v0,v1,v2] sorted descending, vectors: 3x3 row-major where
//   column i is the eigenvector for values[i] }.
export function eigSymmetric3(A) {
  // A is 3x3 symmetric. Copy because we mutate.
  const a = new Float64Array(A);
  // V starts as identity
  const V = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);

  for (let iter = 0; iter < 60; iter++) {
    // Find largest off-diagonal magnitude
    let p = 0, q = 1;
    let maxOff = Math.abs(a[1]);
    if (Math.abs(a[2]) > maxOff) { p = 0; q = 2; maxOff = Math.abs(a[2]); }
    if (Math.abs(a[5]) > maxOff) { p = 1; q = 2; maxOff = Math.abs(a[5]); }
    if (maxOff < 1e-12) break;

    const app = a[p * 3 + p];
    const aqq = a[q * 3 + q];
    const apq = a[p * 3 + q];
    const theta = (aqq - app) / (2 * apq);
    const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;

    // Update A
    a[p * 3 + p] = app - t * apq;
    a[q * 3 + q] = aqq + t * apq;
    a[p * 3 + q] = 0;
    a[q * 3 + p] = 0;
    for (let i = 0; i < 3; i++) {
      if (i !== p && i !== q) {
        const aip = a[i * 3 + p];
        const aiq = a[i * 3 + q];
        a[i * 3 + p] = c * aip - s * aiq;
        a[p * 3 + i] = a[i * 3 + p];
        a[i * 3 + q] = s * aip + c * aiq;
        a[q * 3 + i] = a[i * 3 + q];
      }
    }
    // Update eigenvectors V := V * R
    for (let i = 0; i < 3; i++) {
      const vip = V[i * 3 + p];
      const viq = V[i * 3 + q];
      V[i * 3 + p] = c * vip - s * viq;
      V[i * 3 + q] = s * vip + c * viq;
    }
  }

  const vals = [a[0], a[4], a[8]];
  // Sort descending by eigenvalue, permute eigenvector columns to match
  const idx = [0, 1, 2].sort((i, j) => vals[j] - vals[i]);
  const sortedVals = [vals[idx[0]], vals[idx[1]], vals[idx[2]]];
  const sortedVecs = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    sortedVecs[r * 3 + 0] = V[r * 3 + idx[0]];
    sortedVecs[r * 3 + 1] = V[r * 3 + idx[1]];
    sortedVecs[r * 3 + 2] = V[r * 3 + idx[2]];
  }
  // Force right-handed
  if (det3(sortedVecs) < 0) {
    for (let r = 0; r < 3; r++) sortedVecs[r * 3 + 2] *= -1;
  }
  return { values: sortedVals, vectors: sortedVecs };
}

// Mean of an array of vec3
export function meanVec3(points) {
  const m = vec3();
  for (const p of points) {
    m[0] += p[0]; m[1] += p[1]; m[2] += p[2];
  }
  const n = points.length || 1;
  m[0] /= n; m[1] /= n; m[2] /= n;
  return m;
}

// 3x3 covariance matrix of a list of vec3 (already centered or not)
export function covariance3(points, center) {
  const c = center ?? meanVec3(points);
  const cov = new Float64Array(9);
  for (const p of points) {
    const dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
    cov[4] += dy * dy; cov[5] += dy * dz;
    cov[8] += dz * dz;
  }
  const n = Math.max(points.length - 1, 1);
  cov[0] /= n; cov[1] /= n; cov[2] /= n;
  cov[4] /= n; cov[5] /= n;
  cov[8] /= n;
  cov[3] = cov[1]; cov[6] = cov[2]; cov[7] = cov[5];
  return cov;
}
