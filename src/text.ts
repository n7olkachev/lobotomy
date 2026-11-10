// Text helpers shared by the UI, the worker and the image export:
// loop detection (greedy decoding gets stuck), pull quotes, markdown stripping.

export interface Loop {
  /** text.slice(0, start) is everything before the repeating part */
  start: number;
  /** the repeating unit, aligned to a word / sentence / line start */
  unit: string;
  /** how many times the unit appears back to back, counted from `start` */
  repeats: number;
}

const WORDISH = /[\p{L}\p{N}]/u;
const isSpace = (c: string | undefined) => c === " " || c === "\n" || c === "\t" || c === "\r";

/** Offset in [0, p) at which the repeating unit reads best: a line start, else a sentence start, else a word start. */
function alignUnit(text: string, s: number, p: number): number {
  let best = 0, bestRank = 0;
  for (let o = 0; o < p; o++) {
    const q = s + o;
    if (isSpace(text[q])) continue;
    const prev = text[q - 1];
    let rank = 0;
    if (q === 0 || prev === "\n") rank = 3;
    else if (isSpace(prev) && /[.!?…:]/.test(text[q - 2] ?? "")) rank = 2;
    else if (isSpace(prev)) rank = 1;
    if (rank > bestRank) { best = o; bestRank = rank; if (rank === 3) break; }
  }
  return best;
}

/**
 * Finds a periodic suffix: "…coffee coffee coffee coffee". Returns null unless the unit repeats at least
 * `minRepeats` times and the repeating part is at least `minLen` characters long.
 */
export function findLoop(text: string, minRepeats = 3, minLen = 18): Loop | null {
  // The last repeat usually lacks the whitespace that separates repeats (the answer just ended there).
  const pads = /\s$/.test(text) ? [""] : ["", " ", "\n", "\n\n"];
  for (const pad of pads) {
    const loop = periodicSuffix(text + pad, minRepeats, minLen);
    if (loop) return loop;
  }
  return null;
}

/** Thresholds for cutting a generation short: three repeats and enough text that it is clearly stuck. */
export const isStuck = (text: string) => findLoop(text, 3, 48) !== null;

function periodicSuffix(text: string, minRepeats: number, minLen: number): Loop | null {
  const n = text.length;
  const maxP = Math.min(240, Math.floor(n / minRepeats));
  for (let p = 1; p <= maxP; p++) {
    let i = n - p - 1;
    while (i >= 0 && text.charCodeAt(i) === text.charCodeAt(i + p)) i--;
    const s = i + 1;
    if (n - s < minRepeats * p || n - s < minLen) continue;
    const start = s + alignUnit(text, s, p);
    const repeats = Math.floor((n - start) / p);
    if (repeats < minRepeats) continue;
    const unit = text.slice(start, start + p);
    if (!WORDISH.test(unit)) continue;
    return { start, unit, repeats };
  }
  return null;
}

/** Markdown and whitespace removed: one line of plain text. */
export function plain(text: string): string {
  return text
    .replace(/```\w*\n?/g, "")
    .replace(/\*\*|__|`/g, "")
    .replace(/^\s*#{1,6}\s+/gm, "")
    .replace(/^\s*[*\-•]\s+/gm, "")
    .replace(/\*(?=\S)/g, "")
    .replace(/(\S)\*/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function clip(t: string, max: number): string {
  if (t.length <= max) return t;
  return t.slice(0, max - 1).replace(/\s+\S*$/, "").replace(/[,;:]$/, "") + "…";
}

/** The opening of an answer: whole sentences until there are at least `min` characters, clipped to `max`. */
export function lead(text: string, min = 40, max = 150): string {
  const t = plain(text);
  const re = /[.!?…]+["”’')\]]*(?=\s|$)/g;
  let end = t.length;
  for (let m = re.exec(t); m; m = re.exec(t)) {
    if (m.index + m[0].length >= min) { end = m.index + m[0].length; break; }
  }
  return clip(t.slice(0, end).trim(), max);
}

/**
 * One line for cards and ladders. A short loop is shown as the unit three times and an ellipsis,
 * a long one as the unit once; `loop` tells the caller to add a "stuck" mark.
 */
export function punchline(text: string, max = 150): { text: string; loop: boolean } {
  const loop = findLoop(text);
  if (!loop) return { text: lead(text, 40, max), loop: false };
  const head = plain(text.slice(0, loop.start));
  const unit = plain(loop.unit);
  const once = /[.!?]$/.test(unit) ? unit : unit.replace(/[,;:]$/, "") + "…";
  const tails = unit.length <= 24 ? [`${unit} ${unit} ${unit}…`, once] : [once];
  for (const tail of tails) {
    const whole = (head ? head + " " : "") + tail;
    if (whole.length <= max) return { text: whole, loop: true };
  }
  return { text: head ? lead(head, 40, max) : clip(unit, max), loop: true };
}
