// The brain from the side, as outlines: drawn in ink on the dose-ladder image (and on the case file when there is
// no photo of the 3D brain to tape to it). The 24 layers are 24 cuts through it, front to back.
import { NUM_LAYERS } from "./catalog";

export const VB = { x: 40, y: -40, w: 508, h: 400 };
const CX = 300, CY = 165, RX = 235, RY = 135, RY_LOW = 104, POW = 2.5;
const X0 = 98, X1 = 502;

function blob(cx: number, cy: number, rx: number, ryTop: number, ryBottom: number, pow: number, lobes: number, amp: number, phase: number): string {
  const pts: string[] = [];
  const N = 420;
  for (let i = 0; i < N; i++) {
    const t = (i / N) * Math.PI * 2;
    const c = Math.cos(t), s = Math.sin(t);
    const ry = s > 0 ? ryBottom : ryTop; // y points down
    const ex = Math.sign(c) * Math.abs(c) ** (2 / pow) * rx, ey = Math.sign(s) * Math.abs(s) ** (2 / pow) * ry;
    const d = Math.hypot(ex, ey) || 1;
    const bump = amp * (Math.abs(Math.sin(t * lobes * 0.5 + phase)) - 0.45); // round lobes, sharp creases
    pts.push(`${(cx + ex + (bump * ex) / d).toFixed(1)},${(cy + ey + (bump * ey) / d).toFixed(1)}`);
  }
  return `M${pts.join("L")}Z`;
}

export const PATHS = {
  cerebrum: blob(CX, CY, RX, RY, RY_LOW, POW, 26, 8, 0.6),
  temporal: blob(232, 250, 128, 40, 46, 2.2, 12, 6, 0.2),
  cerebellum: blob(452, 268, 74, 40, 42, 2, 10, 5, 1.1),
  stem: "M318,262 C330,296 340,326 344,354 L384,348 C377,318 369,290 362,258 Z",
  fissures: ["M150,236 C222,226 302,206 376,166", "M318,38 C300,92 314,140 290,190"],
};

const layerX = (L: number) => X0 + (L / (NUM_LAYERS - 1)) * (X1 - X0);
function rand(n: number, L: number, salt: number) {
  const h = Math.sin(n * 12.9898 + L * 78.233 + salt * 37.719) * 43758.5453;
  return h - Math.floor(h);
}

/** Where a neuron is marked on the side view: on the cut of its layer, at a height that is its own. */
export function neuronPos(L: number, n: number): { x: number; y: number } {
  const x = layerX(L) + (rand(n, L, 1) - 0.5) * 9;
  const u = Math.min(0.999, Math.abs(x - CX) / RX);
  const h = (1 - u ** POW) ** (1 / POW) * 0.8;
  const top = CY - RY * h + 6, bottom = CY + RY_LOW * h - 6;
  return { x, y: top + rand(n, L, 0) * (bottom - top) };
}
