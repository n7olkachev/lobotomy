// Where every neuron sits. The brain is cut into 24 slices from front to back, one per layer of the model, and
// every slice holds that layer's 3,584 neurons on a honeycomb grid. A layer has no geometry of its own:
// a neuron's place inside its slice is just its index, row by row.
import { INTERMEDIATE as I, NUM_LAYERS as NL } from "./catalog";

export interface Slice {
  /** where the slice is cut; the front of the brain (layer 0) is at +z */
  z: number;
  /** distance between neighbours in a row, and between rows */
  pitch: number; dy: number;
  /** centre of the cell at row 0, column 0 (top left); odd rows are shifted right by half a pitch */
  x0: number; y0: number;
  rows: number; cols: number;
  /** row * cols + col -> neuron, or -1 */
  at: Int16Array;
  /** bounding box and circle of the neurons */
  minX: number; maxX: number; minY: number; maxY: number; cx: number; cy: number; r: number;
}

// ---- the shape ----------------------------------------------------------------------------------------------
// An implicit brain: < 1 inside. Cerebrum (two hemispheres with a fissure between them and folds around the rim),
// temporal lobes, cerebellum and a stem. Units: the cerebrum is 2 long (z), about 1.6 wide (x) and 1.1 tall (y).
const P = 2.5;
const pw = (v: number) => Math.pow(Math.abs(v), P);
const smooth = (a: number, b: number, v: number) => { const t = Math.max(0, Math.min(1, (v - a) / (b - a))); return t * t * (3 - 2 * t); };

function shape(x: number, y: number, z: number): number {
  // cerebrum: an egg, narrower at the front, flatter underneath
  const wx = 0.8 - 0.07 * z, cy = 0.09, u = (y - cy) / (y > cy ? 0.62 : 0.5);
  let c = Math.pow(pw(x / wx) + pw(u) + pw(z), 1 / P);
  c *= 1 + 0.026 * Math.cos(13 * Math.atan2(y - cy, x) + 3 * Math.sin(2.5 * z + 0.7)); // folds
  const mid = Math.exp(-((x / 0.034) ** 2));
  c += mid * (0.55 * smooth(-0.1, 0.6, u) + 0.3 * smooth(0.25, 0.9, -u)); // the fissure between the hemispheres
  // temporal lobes
  const tx = (Math.abs(x) - 0.5) / 0.3, ty = (y + 0.31) / 0.25, tz = (z - 0.1) / 0.56;
  const t = Math.sqrt(tx * tx + ty * ty + tz * tz);
  // cerebellum, with a groove down the middle
  const bx = x / 0.5, by = (y + 0.5) / 0.235, bz = (z + 0.6) / 0.34;
  const b = Math.sqrt(bx * bx + by * by + bz * bz) + 0.18 * mid;
  // stem: a tapering rod from under the middle down and back
  const ay = -0.3, az = -0.16, ly = -0.74, lz = -0.19; // start, and the vector to its end
  const k = Math.max(0, Math.min(1, ((y - ay) * ly + (z - az) * lz) / (ly * ly + lz * lz)));
  const s = Math.hypot(x, y - ay - k * ly, z - az - k * lz) / (0.105 - 0.03 * k);
  return Math.min(c, t, b, s);
}

// ---- the slices ---------------------------------------------------------------------------------------------
const Z_FRONT = 0.86, Z_BACK = -0.86;
const BOX = { x: 1.0, top: 0.82, bottom: -1.12 };
const ROW = Math.sqrt(3) / 2;

export const positions = new Float32Array(NL * I * 3);
/** 1 for the neurons on the rim of their slice (a neighbour is missing): drawn brighter, so every slice has an outline. */
export const rim = new Uint8Array(NL * I);
export const slices: Slice[] = [];
/** Triangles that fill every slice (its neurons' cells, row by row), for the glass plate behind the neurons. */
export let plates = new Float32Array(0);
export const plateRange: { start: number; count: number }[] = [];

function build() {
  const tris: number[] = [];
  for (let L = 0; L < NL; L++) {
    const z = Z_FRONT + ((Z_BACK - Z_FRONT) * L) / (NL - 1);
    // area of the cut, from a coarse raster
    const S = 0.014;
    let inside = 0;
    for (let y = BOX.bottom; y <= BOX.top; y += S) for (let x = -BOX.x; x <= BOX.x; x += S) if (shape(x, y, z) < 1) inside++;
    let pitch = Math.sqrt((inside * S * S) / (I * ROW)) * 0.99;
    let cells: { x: number; y: number; r: number; c: number; f: number }[] = [];
    let rows = 0, cols = 0, dy = 0;
    for (let tries = 0; tries < 40; tries++) {
      dy = pitch * ROW;
      rows = Math.floor((BOX.top - BOX.bottom) / dy) + 1;
      cols = Math.floor((2 * BOX.x) / pitch) + 1;
      cells = [];
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const x = -BOX.x + c * pitch + (r & 1 ? pitch / 2 : 0), y = BOX.top - r * dy, f = shape(x, y, z);
        if (f < 1) cells.push({ x, y, r, c, f });
      }
      if (cells.length >= I) break;
      pitch *= 0.992;
    }
    // exactly 3,584 of them: the outermost ones go
    cells.sort((a, b) => a.f - b.f);
    cells.length = I;
    cells.sort((a, b) => a.r - b.r || a.c - b.c);
    const at = new Int16Array(rows * cols).fill(-1);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    cells.forEach((cell, n) => {
      at[cell.r * cols + cell.c] = n;
      const o = (L * I + n) * 3;
      positions[o] = cell.x; positions[o + 1] = cell.y; positions[o + 2] = z;
      if (cell.x < minX) minX = cell.x; if (cell.x > maxX) maxX = cell.x;
      if (cell.y < minY) minY = cell.y; if (cell.y > maxY) maxY = cell.y;
    });
    const has = (r: number, c: number) => r >= 0 && r < rows && c >= 0 && c < cols && at[r * cols + c] >= 0;
    cells.forEach((cell, n) => {
      const { r, c } = cell, o = r & 1 ? 0 : -1; // the two neighbours in the rows above and below sit at c+o and c+o+1
      if (!(has(r, c - 1) && has(r, c + 1) && has(r - 1, c + o) && has(r - 1, c + o + 1) && has(r + 1, c + o) && has(r + 1, c + o + 1))) rim[L * I + n] = 1;
    });
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    let rad = 0;
    for (const cell of cells) rad = Math.max(rad, Math.hypot(cell.x - cx, cell.y - cy));
    slices.push({ z, pitch, dy, x0: -BOX.x, y0: BOX.top, rows, cols, at, minX, maxX, minY, maxY, cx, cy, r: rad + pitch });
    // the plate: one quad per run of cells in a row
    const start = tris.length / 3;
    for (let r = 0; r < rows; r++) {
      const off = r & 1 ? pitch / 2 : 0, y = BOX.top - r * dy, y1 = y + dy / 2, y2 = y - dy / 2;
      for (let c = 0; c < cols; c++) {
        if (at[r * cols + c] < 0) continue;
        let e = c;
        while (e + 1 < cols && at[r * cols + e + 1] >= 0) e++;
        const x1 = -BOX.x + c * pitch + off - pitch * 0.6, x2 = -BOX.x + e * pitch + off + pitch * 0.6;
        tris.push(x1, y1, z, x2, y1, z, x2, y2, z, x1, y1, z, x2, y2, z, x1, y2, z);
        c = e;
      }
    }
    plateRange.push({ start, count: tris.length / 3 - start });
  }
  plates = new Float32Array(tris);
}
build();

export const zOf = (L: number) => slices[L].z;
export function posOf(L: number, n: number): [number, number, number] {
  const o = (L * I + n) * 3;
  return [positions[o], positions[o + 1], positions[o + 2]];
}

/** The neuron of a slice nearest to a point of its plane, if one is within `reach` (in pitches). */
export function neuronNear(L: number, x: number, y: number, reach = 0.75): number {
  const s = slices[L], span = Math.ceil(reach) + 1;
  const r0 = Math.round((s.y0 - y) / s.dy);
  let best = -1, bestD = (reach * s.pitch) ** 2;
  for (let r = Math.max(0, r0 - span); r <= Math.min(s.rows - 1, r0 + span); r++) {
    const off = r & 1 ? s.pitch / 2 : 0, cy = s.y0 - r * s.dy, c0 = Math.round((x - s.x0 - off) / s.pitch);
    for (let c = Math.max(0, c0 - span); c <= Math.min(s.cols - 1, c0 + span); c++) {
      const n = s.at[r * s.cols + c];
      if (n < 0) continue;
      const d = (s.x0 + c * s.pitch + off - x) ** 2 + (cy - y) ** 2;
      if (d < bestD) { bestD = d; best = n; }
    }
  }
  return best;
}

/** Every neuron of a slice within `reach` pitches of a point of its plane, with its distance in pitches. */
export function neuronsAround(L: number, x: number, y: number, reach: number): { n: number; d: number }[] {
  const s = slices[L], span = Math.ceil(reach) + 1, out: { n: number; d: number }[] = [];
  const r0 = Math.round((s.y0 - y) / s.dy);
  for (let r = Math.max(0, r0 - span); r <= Math.min(s.rows - 1, r0 + span); r++) {
    const off = r & 1 ? s.pitch / 2 : 0, cy = s.y0 - r * s.dy, c0 = Math.round((x - s.x0 - off) / s.pitch);
    for (let c = Math.max(0, c0 - span); c <= Math.min(s.cols - 1, c0 + span); c++) {
      const n = s.at[r * s.cols + c];
      if (n < 0) continue;
      const d = Math.hypot(s.x0 + c * s.pitch + off - x, cy - y) / s.pitch;
      if (d <= reach) out.push({ n, d });
    }
  }
  return out;
}
