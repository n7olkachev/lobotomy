// Shareable images, drawn on a canvas so they can be copied or saved as a PNG: a "case file" for one
// answer and a dose ladder (the same question at rising dose). Styled as paperwork, not as the app.
import { PATHS, VB, neuronPos } from "./ink";
import type { Photo } from "./gl";
import { TOTAL_NEURONS, type Title } from "./personas";
import { formatRecipe, type Probe } from "./steer";
import { lead, punchline } from "./text";
import type { Vitals, VState } from "./vitals";

export interface CaseCard {
  kind: "case"; title: Title; probes: Probe[]; prompt: string; before: string; after: string; vitals: Vitals; url: string;
  /** what the scanner read off this neuron */
  scan?: string;
  /** the brain as it was on screen */
  photo?: Photo | null;
}
export interface LadderRowCard { probes: Probe[]; text: string; state: VState | null }
export interface LadderCard { kind: "ladder"; title: Title; probes: Probe[]; prompt: string; rows: LadderRowCard[]; url: string }
export type Card = CaseCard | LadderCard;

const W = 1200, PAD = 64, SCALE = 2;
const TYPE = `"American Typewriter", "Courier New", Courier, monospace`;
const MONO = `ui-monospace, Menlo, "SF Mono", "Courier New", monospace`;
const PAPER = "#efe7d6", INK = "#231f1a", FADED = "#857a66", RED = "#b3261e", POS = "#d9480f", NEG = "#1c6fb5";
const LABEL_W = 172;

type Ctx = CanvasRenderingContext2D;

function wrap(ctx: Ctx, text: string, maxW: number, maxLines: number): string[] {
  const lines: string[] = [];
  let cur = "";
  for (const w of text.split(/\s+/).filter(Boolean)) {
    const t = cur ? `${cur} ${w}` : w;
    if (!cur || ctx.measureText(t).width <= maxW) cur = t; else { lines.push(cur); cur = w; }
  }
  if (cur) lines.push(cur);
  if (lines.length <= maxLines) return lines;
  const cut = lines.slice(0, maxLines);
  let last = cut[maxLines - 1];
  while (ctx.measureText(last + "…").width > maxW && last.includes(" ")) last = last.replace(/\s+\S*$/, "");
  cut[maxLines - 1] = last.replace(/[,;:.]$/, "") + "…";
  return cut;
}

/** Text with letter spacing, for the small-caps labels. Returns the width drawn. */
function spaced(ctx: Ctx, text: string, x: number, y: number, gap: number): number {
  let cx = x;
  for (const ch of text) { ctx.fillText(ch, cx, y); cx += ctx.measureText(ch).width + gap; }
  return cx - x;
}

function fit(ctx: Ctx, text: string, maxW: number, size: number, family: string, weight = ""): number {
  let s = size;
  do { ctx.font = `${weight} ${s}px ${family}`; s -= 1; } while (ctx.measureText(text).width > maxW && s > 11);
  return s + 1;
}

function paper(ctx: Ctx, h: number) {
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, W, h);
  // grain: deterministic specks, so the same card always looks the same
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 2600; i++) {
    ctx.fillStyle = rnd() < 0.5 ? "rgba(60,45,20,0.05)" : "rgba(255,255,255,0.22)";
    ctx.fillRect(rnd() * W, rnd() * h, 1 + rnd() * 2, 1 + rnd() * 2);
  }
  const g = ctx.createRadialGradient(W / 2, h / 2, Math.min(W, h) * 0.35, W / 2, h / 2, Math.max(W, h) * 0.75);
  g.addColorStop(0, "rgba(90,60,20,0)");
  g.addColorStop(1, "rgba(90,60,20,0.16)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, h);
  ctx.strokeStyle = "rgba(35,31,26,0.55)";
  ctx.lineWidth = 2;
  ctx.strokeRect(22, 22, W - 44, h - 44);
}

function rule(ctx: Ctx, y: number, double = false) {
  ctx.strokeStyle = INK;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
  if (double) { ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(PAD, y + 6); ctx.lineTo(W - PAD, y + 6); ctx.stroke(); }
}

const STAMP = "LOBOTOMIZED";
const STAMP_FONT = `bold 40px ${TYPE}`;
function stampWidth(ctx: Ctx, text: string): number {
  ctx.font = STAMP_FONT;
  return ctx.measureText(text).width + [...text].length * 4 + 44;
}

/** A rubber stamp whose right edge sits at `right`. */
function stamp(ctx: Ctx, text: string, right: number, cy: number, color: string) {
  const w = stampWidth(ctx, text), h = 74;
  ctx.save();
  ctx.translate(right - w / 2, cy);
  ctx.rotate(-0.09);
  ctx.globalAlpha = 0.78;
  ctx.font = STAMP_FONT;
  ctx.strokeStyle = color; ctx.fillStyle = color;
  ctx.lineWidth = 5; ctx.strokeRect(-w / 2, -h / 2, w, h);
  ctx.lineWidth = 1.5; ctx.strokeRect(-w / 2 + 8, -h / 2 + 8, w - 16, h - 16);
  ctx.textBaseline = "middle";
  spaced(ctx, text, -w / 2 + 24, 3, 4);
  ctx.restore();
}

/** The brain from the app, in ink. `glow` 0..1 lights the whole thing up (dose ladder). */
function drawBrain(ctx: Ctx, x: number, y: number, w: number, probes: Probe[], glow = 0) {
  const s = w / VB.w;
  ctx.save();
  ctx.translate(x - VB.x * s, y - VB.y * s);
  ctx.scale(s, s);
  if (glow > 0) {
    const g = ctx.createRadialGradient(300, 170, 20, 300, 170, 130 + glow * 100);
    g.addColorStop(0, `rgba(240,120,30,${0.25 + glow * 0.5})`);
    g.addColorStop(1, "rgba(240,120,30,0)");
    ctx.fillStyle = g;
    ctx.fillRect(VB.x - 200, VB.y - 200, VB.w + 400, VB.h + 400);
    if (glow > 0.6) {
      ctx.strokeStyle = `rgba(217,72,15,${(glow - 0.5) * 0.9})`;
      ctx.lineWidth = 5;
      for (let i = 0; i < 14; i++) {
        const a = (i / 14) * Math.PI * 2 + 0.2, r0 = 256, r1 = r0 + 12 + glow * 34;
        ctx.beginPath();
        ctx.moveTo(300 + Math.cos(a) * r0, 175 + Math.sin(a) * r0 * 0.74);
        ctx.lineTo(300 + Math.cos(a) * r1, 175 + Math.sin(a) * r1 * 0.74);
        ctx.stroke();
      }
    }
  }
  const parts = { stem: new Path2D(PATHS.stem), cerebellum: new Path2D(PATHS.cerebellum), temporal: new Path2D(PATHS.temporal), cerebrum: new Path2D(PATHS.cerebrum) };
  const fill = glow > 0 ? `rgba(250,${Math.round(232 - glow * 40)},${Math.round(200 - glow * 90)},1)` : "#e7dcc5";
  ctx.lineJoin = "round";
  ctx.strokeStyle = INK; ctx.fillStyle = fill;
  ctx.lineWidth = 9;
  for (const p of [parts.stem, parts.cerebellum]) { ctx.stroke(p); ctx.fill(p); }
  ctx.stroke(parts.temporal); ctx.stroke(parts.cerebrum);
  ctx.fill(parts.temporal); ctx.fill(parts.cerebrum);
  ctx.lineWidth = 4; ctx.strokeStyle = "rgba(35,31,26,0.45)";
  for (const d of PATHS.fissures) ctx.stroke(new Path2D(d));
  for (const p of probes) {
    const pos = neuronPos(p.L, p.n), c = p.dose >= 0 ? POS : NEG, r = 34 + Math.min(1.5, Math.abs(p.dose)) * 40;
    const g = ctx.createRadialGradient(pos.x, pos.y, 2, pos.x, pos.y, r);
    g.addColorStop(0, c); g.addColorStop(0.25, c + "cc"); g.addColorStop(1, c + "00");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = INK; ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(pos.x, VB.y + 4); ctx.lineTo(pos.x, pos.y); ctx.stroke();
    ctx.fillStyle = INK;
    ctx.beginPath(); ctx.arc(pos.x, pos.y, 9, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

/** The brain as it was on screen: a print, taped to the file. Marks the electrodes. */
function drawPhoto(ctx: Ctx, ph: Photo, x: number, y: number, w: number) {
  const m = 13, inner = w - m * 2;
  ctx.save();
  ctx.translate(x + w / 2, y + w / 2);
  ctx.rotate(0.022);
  ctx.shadowColor = "rgba(40,25,5,0.38)"; ctx.shadowBlur = 22; ctx.shadowOffsetY = 8;
  ctx.fillStyle = "#f8f4ea";
  ctx.fillRect(-w / 2, -w / 2, w, w);
  ctx.shadowColor = "transparent"; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
  ctx.fillStyle = "#06080b";
  ctx.fillRect(-inner / 2, -inner / 2, inner, inner);
  ctx.save();
  ctx.beginPath(); ctx.rect(-inner / 2, -inner / 2, inner, inner); ctx.clip();
  const k = inner / ph.size;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(ph.canvas, -inner / 2 - ph.x * k, -inner / 2 - ph.y * k, ph.canvas.width * k, ph.canvas.height * k);
  for (const mark of ph.marks) {
    const px = -inner / 2 + (mark.x - ph.x) * k, py = -inner / 2 + (mark.y - ph.y) * k, c = mark.sign > 0 ? "#ffa64d" : "#58c8ff";
    if (Math.abs(px) > inner / 2 - 10 || Math.abs(py) > inner / 2 - 10) continue;
    ctx.strokeStyle = c; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(px, py, 13, 0, Math.PI * 2); ctx.stroke();
    if (!mark.label) continue;
    ctx.font = `bold 17px ${MONO}`;
    const tw = ctx.measureText(mark.label).width + 18, left = Math.max(-inner / 2 + 6, Math.min(inner / 2 - tw - 6, px - tw / 2)), top = py + 22 > inner / 2 - 34 ? py - 50 : py + 20;
    ctx.fillStyle = c;
    ctx.beginPath(); ctx.roundRect(left, top, tw, 28, 14); ctx.fill();
    ctx.fillStyle = "#0a0d11"; ctx.textBaseline = "middle";
    ctx.fillText(mark.label, left + 9, top + 15);
    ctx.textBaseline = "alphabetic";
  }
  ctx.restore();
  // a strip of tape
  ctx.rotate(-0.07);
  ctx.fillStyle = "rgba(222,206,150,0.62)";
  ctx.fillRect(-58, -w / 2 - 15, 116, 34);
  ctx.restore();
}

function header(ctx: Ctx): number {
  ctx.fillStyle = INK;
  ctx.textBaseline = "alphabetic";
  ctx.font = `bold 62px ${TYPE}`;
  spaced(ctx, "LOBOTOMY", PAD, 112, 10);
  ctx.font = `21px ${TYPE}`;
  ctx.fillStyle = FADED;
  ctx.fillText("open-brain surgery on a language model, in a browser tab", PAD, 148);
  rule(ctx, 172, true);
  return 172 + 6;
}

/** A labelled row. Returns the y after it. */
function field(ctx: Ctx, y: number, label: string, lines: string[], font: string, lineH: number, color = INK, maxX = W - PAD): number {
  ctx.fillStyle = FADED;
  ctx.font = label === "℞" ? `26px ${TYPE}` : `bold 16px ${TYPE}`;
  spaced(ctx, label, PAD, y + lineH * 0.72, 3);
  ctx.fillStyle = color;
  ctx.font = font;
  lines.forEach((ln, i) => ctx.fillText(ln, PAD + LABEL_W, y + lineH * (i + 0.76), maxX - PAD - LABEL_W));
  return y + Math.max(1, lines.length) * lineH;
}

function footer(ctx: Ctx, y: number, url: string): number {
  rule(ctx, y);
  ctx.fillStyle = FADED;
  ctx.font = `bold 16px ${TYPE}`;
  const lw = spaced(ctx, "DO IT YOURSELF", PAD, y + 44, 3);
  ctx.fillStyle = INK;
  const shown = url.replace(/^https?:\/\//, "");
  const right = "no server · runs on your GPU";
  ctx.font = `19px ${TYPE}`;
  const rw = ctx.measureText(right).width;
  fit(ctx, shown, W - PAD * 2 - lw - rw - 60, 21, MONO, "bold");
  ctx.fillText(shown, PAD + lw + 20, y + 44);
  ctx.font = `19px ${TYPE}`;
  ctx.fillStyle = FADED;
  ctx.fillText(right, W - PAD - rw, y + 44);
  return y + 44 + 46;
}

function procedure(title: Title, probes: Probe[]): string {
  if (!probes.length) return "none — the patient is intact";
  const pct = ((probes.length / TOTAL_NEURONS) * 100).toFixed(3);
  return `${title.emoji} ${title.name} — ${probes.length} of ${TOTAL_NEURONS.toLocaleString("en-US")} neurons changed (${pct}%)`;
}

const doseLabel = (probes: Probe[]) =>
  probes.length === 0 ? "untouched" : `dose ${probes.map((p) => (p.dose >= 0 ? "+" : "−") + Math.abs(p.dose).toFixed(2)).join(" ")}`;

function drawCase(ctx: Ctx, c: CaseCard): number {
  let y = header(ctx) + 26;
  const brainW = c.photo ? 400 : 236, textMax = W - PAD - brainW - 30, brainBottom = c.photo ? y + brainW + 8 : y - 6 + (brainW * VB.h) / VB.w;
  if (c.photo) drawPhoto(ctx, c.photo, W - PAD - brainW, y + 2, brainW); else drawBrain(ctx, W - PAD - brainW, y - 6, brainW, c.probes);
  const f = `24px ${TYPE}`;
  y = field(ctx, y, "PATIENT", [`Qwen3.5-0.8B, ${TOTAL_NEURONS.toLocaleString("en-US")} neurons`], f, 38, INK, textMax);
  ctx.font = f;
  y = field(ctx, y, "PROCEDURE", wrap(ctx, procedure(c.title, c.probes), textMax - PAD - LABEL_W, 3), f, 38, INK, textMax);
  if (c.probes.length) {
    ctx.font = `bold 21px ${MONO}`;
    y = field(ctx, y, "℞", wrap(ctx, formatRecipe(c.probes), textMax - PAD - LABEL_W, 2), `bold 21px ${MONO}`, 38, INK, textMax);
  }
  ctx.font = f;
  if (c.scan) y = field(ctx, y, "SCANNER", wrap(ctx, c.scan, textMax - PAD - LABEL_W, 2), f, 38, INK, textMax);
  y = field(ctx, y, "ASKED", wrap(ctx, `“${c.prompt.trim()}”`, textMax - PAD - LABEL_W, 3), f, 38, INK, textMax);
  const steered = c.probes.length > 0;
  let beforeDrawn = false;
  if (c.photo && steered && c.before) {
    // next to the photo there is room for the untouched answer
    ctx.font = `22px ${TYPE}`;
    y = field(ctx, y + 14, "BEFORE", wrap(ctx, lead(c.before, 60, 230), textMax - PAD - LABEL_W, 5), `22px ${TYPE}`, 33, FADED, textMax);
    beforeDrawn = true;
  }
  y = Math.max(y, brainBottom + 4);

  if (steered && c.before && !beforeDrawn) {
    ctx.font = `22px ${TYPE}`;
    y = field(ctx, y + 22, "BEFORE", wrap(ctx, lead(c.before, 60, 230), W - PAD * 2 - LABEL_W, 3), `22px ${TYPE}`, 33, FADED);
  }
  const p = punchline(steered ? c.after : c.before, 230);
  const size = p.text.length <= 60 ? 54 : p.text.length <= 120 ? 44 : 36;
  ctx.font = `bold ${size}px ${TYPE}`;
  const lines = wrap(ctx, `“${p.text}”${p.loop ? " ↻" : ""}`, W - PAD * 2 - LABEL_W, 6);
  y = field(ctx, y + 26, steered ? "AFTER" : "ANSWER", lines, `bold ${size}px ${TYPE}`, Math.round(size * 1.24));

  if (steered) {
    const v = c.vitals;
    const bits = [v.score !== null ? `coherence ${v.score}/100` : "", v.label.toUpperCase(), ...v.symptoms.map((s) => s.name)].filter(Boolean);
    const room = W - PAD * 2 - LABEL_W - stampWidth(ctx, STAMP) - 36;
    ctx.font = `22px ${TYPE}`;
    const top = y + 24;
    y = Math.max(field(ctx, top, "VITALS", wrap(ctx, bits.join(" · "), room, 3), `22px ${TYPE}`, 34), top + 78);
    stamp(ctx, STAMP, W - PAD - 8, (top + y) / 2, RED);
  }
  return footer(ctx, y + 34, c.url);
}

function drawLadder(ctx: Ctx, c: LadderCard): number {
  let y = header(ctx) + 24;
  const f = `24px ${TYPE}`;
  ctx.font = f;
  y = field(ctx, y, "PROCEDURE", wrap(ctx, procedure(c.title, c.probes), W - PAD * 2 - LABEL_W, 2), f, 38);
  y = field(ctx, y, "ASKED", wrap(ctx, `“${c.prompt.trim()}”`, W - PAD * 2 - LABEL_W, 2), f, 38);
  y += 18;
  const rowH = 122, brainW = 132, textX = PAD + brainW + 34, n = c.rows.length;
  c.rows.forEach((r, i) => {
    ctx.strokeStyle = "rgba(35,31,26,0.35)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 6]);
    ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    ctx.setLineDash([]);
    drawBrain(ctx, PAD + 4, y + 14, brainW, [], n > 1 ? i / (n - 1) : 0);
    ctx.fillStyle = r.state === "flatline" ? RED : FADED;
    ctx.font = `bold 17px ${MONO}`;
    const meta = [doseLabel(r.probes), r.state && r.state !== "intact" ? r.state : ""].filter(Boolean).join("  ·  ");
    ctx.fillText(meta, textX, y + 32);
    const p = punchline(r.text, 150);
    ctx.fillStyle = INK;
    const size = fitLines(ctx, p.text + (p.loop ? " ↻" : ""), W - PAD - textX, 2, 31);
    ctx.font = `bold ${size}px ${TYPE}`;
    wrap(ctx, p.text + (p.loop ? " ↻" : ""), W - PAD - textX, 2).forEach((ln, k) => ctx.fillText(ln, textX, y + 70 + k * Math.round(size * 1.2)));
    y += rowH;
  });
  stamp(ctx, STAMP, W - PAD - 8, 98, RED);
  return footer(ctx, y + 10, c.url);
}

/** The largest size (down to 22px) at which the text fits in `maxLines`. */
function fitLines(ctx: Ctx, text: string, maxW: number, maxLines: number, size: number): number {
  for (let s = size; s > 22; s -= 2) {
    ctx.font = `bold ${s}px ${TYPE}`;
    if (wrap(ctx, text, maxW, 99).length <= maxLines) return s;
  }
  return 22;
}

export function renderCard(card: Card): HTMLCanvasElement {
  // Draw on a tall scratch canvas, then crop to the height the content needed.
  const scratch = document.createElement("canvas");
  const maxH = 2400;
  scratch.width = W * SCALE; scratch.height = maxH * SCALE;
  const measure = scratch.getContext("2d")!;
  measure.scale(SCALE, SCALE);
  const h = Math.ceil(card.kind === "case" ? drawCase(measure, card) : drawLadder(measure, card));
  const out = document.createElement("canvas");
  out.width = W * SCALE; out.height = h * SCALE;
  const ctx = out.getContext("2d")!;
  ctx.scale(SCALE, SCALE);
  paper(ctx, h);
  if (card.kind === "case") drawCase(ctx, card); else drawLadder(ctx, card);
  return out;
}

export const toBlob = (c: HTMLCanvasElement) => new Promise<Blob>((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("could not encode the image"))), "image/png"));
