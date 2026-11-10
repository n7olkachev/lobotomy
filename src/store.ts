// The map: what the scanner has read off every face of every neuron, plus the player's own notes.
//
// A neuron has two faces: pushed up (+) or pushed down (−). A reading is the word the model would use for itself
// with that face pushed ("I am a ___"), how sure it is, and how far the next-word distribution moved.
// Readings come from two places: survey files shipped with the site (layers somebody scanned before), and the
// player's own scanner. The player's readings, the answers they have heard and the names they gave live in localStorage.
import { useEffect, useState } from "preact/hooks";
import { INTERMEDIATE, NUM_LAYERS } from "./catalog";

export const FACES = INTERMEDIATE * 2;
export const faceOf = (n: number, sign: number) => n * 2 + (sign > 0 ? 0 : 1);
export const signOf = (f: number): 1 | -1 => (f % 2 === 0 ? 1 : -1);

export const SRC_SURVEY = 1, SRC_YOU = 2; // where a reading came from (0 = there is none)
/** What a reading amounts to. */
export const NONE = 0, QUIET = 1, CHATTER = 2, WEAK = 3, STRONG = 4, ODD = 5;
export type Kind = 0 | 1 | 2 | 3 | 4 | 5;

export interface FaceReading { word: string; p: number; kl: number; src: number; kind: Kind }

interface Layer { w: Uint16Array; p: Uint8Array; kl: Uint8Array; src: Uint8Array }
const blank = (): Layer => ({ w: new Uint16Array(FACES), p: new Uint8Array(FACES), kl: new Uint8Array(FACES), src: new Uint8Array(FACES) });

const words: string[] = [""]; // index 0 = not scanned
const wordIndex = new Map<string, number>([["", 0]]);
const layers: Layer[] = Array.from({ length: NUM_LAYERS }, blank);
let base = " large"; // the unsteered model's word
let version = 0;
const layerVer = new Uint32Array(NUM_LAYERS); // bumped when anything read off a layer changes: the 3D view repaints that slice
const subs = new Set<() => void>();
const bump = () => { version++; subs.forEach((f) => f()); };
export const layerVersion = (L: number) => layerVer[L];

// ---- what counts as a signal ------------------------------------------------------------------------------
// Words the untouched model would also use for itself; a neuron that only gets it to one of these is chatter.
const GENERIC = new Set(("large highly big helpful friendly virtual digital ai powerful small language conversational chat new very general " +
  "sophisticated advanced smart little q bit text model computer part team question complex specific different unique special massive huge " +
  "personalized personal multi software machine neural deep program tool system ll l").split(" "));
// Grammar rather than a thing to be ("I am a and…"): the sentence derails, which is a find of another kind.
const FUNCTION = new Set(("and or of the a an is are was be which while there also however then me my i you it this that to in on for with as at by " +
  "but if so not no yes more most well way lot few what who when where how just only can will would").split(" "));
const WORDLIKE = /^(?: [\p{L}][\p{L}\p{M}'’-]+|\p{Script=Han}+)$/u; // a whole word of two letters or more
const wordClass: number[] = [0]; // per word index: 1 base, 2 generic, 3 a word, 4 not a thing to be
function classOf(i: number): number {
  let c = wordClass[i];
  if (c === undefined) {
    const w = words[i], t = w.trim().toLowerCase();
    c = w === base ? 1 : !WORDLIKE.test(w) || FUNCTION.has(t) ? 4 : GENERIC.has(t) ? 2 : 3;
    wordClass[i] = c;
  }
  return c;
}
function kindOf(i: number, p: number): Kind {
  if (i === 0) return NONE;
  const c = classOf(i);
  if (c === 1) return QUIET;
  if (c === 4) return p >= 50 ? ODD : CHATTER;
  if (c === 2) return p >= 50 ? WEAK : CHATTER; // "I am a bit…": a mood, not a thing, but it means it
  return p >= 30 ? STRONG : p >= 12 ? WEAK : CHATTER;
}

function wordId(word: string): number {
  let i = wordIndex.get(word);
  if (i === undefined) { i = words.length; words.push(word); wordIndex.set(word, i); }
  return i;
}

// ---- reading and writing -----------------------------------------------------------------------------------
export function reading(L: number, f: number): FaceReading | null {
  const l = layers[L], i = l.w[f];
  if (!i) return null;
  return { word: words[i], p: l.p[f] / 100, kl: l.kl[f] / 10, src: l.src[f], kind: kindOf(i, l.p[f]) };
}
/** A number that orders faces by how much they are worth a look. */
function score(L: number, f: number): number {
  const l = layers[L], k = kindOf(l.w[f], l.p[f]);
  if (k === NONE) return -1;
  if (k === STRONG) return 3 + l.p[f] / 100;
  if (k === ODD) return 2 + l.p[f] / 100;
  if (k === WEAK) return classOf(l.w[f]) === 2 ? 0.6 + l.p[f] / 250 : 1 + l.p[f] / 100; // an unusual word beats a sure generic one
  return k === CHATTER ? l.kl[f] / 400 : 0;
}
/** The more telling face of a neuron, and what it amounts to. `known` = how many of its two faces were read. */
export function tile(L: number, n: number): { face: number; kind: Kind; p: number; kl: number; known: number } {
  const a = n * 2, b = a + 1, l = layers[L];
  const f = score(L, b) > score(L, a) ? b : a;
  return { face: f, kind: kindOf(l.w[f], l.p[f]), p: l.p[f] / 100, kl: Math.max(l.kl[a], l.kl[b]) / 10, known: (l.w[a] ? 1 : 0) + (l.w[b] ? 1 : 0) };
}

export interface Reveal { L: number; n: number; sign: 1 | -1; word: string; kind: Kind }
const revealSubs = new Set<(r: Reveal) => void>();
/** Called for every face the player's scanner reads, as it is read (the map flashes it). */
export function onReveal(f: (r: Reveal) => void): () => void {
  revealSubs.add(f);
  return () => revealSubs.delete(f);
}

export function setReading(L: number, n: number, sign: 1 | -1, word: string, p: number, kl: number, src = SRC_YOU) {
  const l = layers[L], f = faceOf(n, sign);
  l.w[f] = wordId(word);
  l.p[f] = Math.max(0, Math.min(100, Math.round(p * 100)));
  l.kl[f] = Math.max(0, Math.min(255, Math.round(kl * 10)));
  l.src[f] = src;
  statsCache[L] = null;
  layerVer[L]++;
  if (src === SRC_YOU) {
    dirty.add(L); scheduleSave();
    const r = { L, n, sign, word, kind: kindOf(l.w[f], l.p[f]) };
    revealSubs.forEach((g) => g(r));
  }
  bump();
}

export function setBase(word: string) {
  if (word === base) return;
  base = word;
  wordClass.length = 1;
  statsCache.fill(null);
  for (let L = 0; L < NUM_LAYERS; L++) layerVer[L]++;
  bump();
}

export interface LayerStats { scanned: number; yours: number; surveyed: boolean; signals: number }
const statsCache: (LayerStats | null)[] = Array(NUM_LAYERS).fill(null);
export function layerStats(L: number): LayerStats {
  let s = statsCache[L];
  if (!s) {
    const l = layers[L];
    let scanned = 0, yours = 0, survey = 0, signals = 0;
    for (let n = 0; n < INTERMEDIATE; n++) {
      const a = n * 2, b = a + 1;
      if (l.w[a] && l.w[b]) scanned++;
      if (l.src[a] === SRC_YOU || l.src[b] === SRC_YOU) yours++;
      if (l.src[a] === SRC_SURVEY) survey++;
      if (kindOf(l.w[a], l.p[a]) >= STRONG || kindOf(l.w[b], l.p[b]) >= STRONG) signals++;
    }
    s = statsCache[L] = { scanned, yours, surveyed: survey > INTERMEDIATE / 2, signals };
  }
  return s;
}
export function totals() {
  let scanned = 0, yours = 0, signals = 0;
  for (let L = 0; L < NUM_LAYERS; L++) { const s = layerStats(L); scanned += s.scanned; yours += s.yours; signals += s.signals; }
  return { scanned, yours, signals };
}

export interface Signal { n: number; sign: 1 | -1; word: string; p: number; kl: number; kind: Kind }
/** Faces of a layer worth interviewing, strongest first. */
export function signals(L: number, min: Kind = WEAK): Signal[] {
  const out: (Signal & { s: number })[] = [], l = layers[L];
  for (let f = 0; f < FACES; f++) {
    const k = kindOf(l.w[f], l.p[f]);
    if (k >= min) out.push({ n: f >> 1, sign: signOf(f), word: words[l.w[f]], p: l.p[f] / 100, kl: l.kl[f] / 10, kind: k, s: score(L, f) });
  }
  return out.sort((a, b) => b.s - a.s);
}

/** Re-render when the map changes (at most once per frame). Returns a number that changes with it. */
export function useScans(): number {
  const [, set] = useState(0);
  useEffect(() => {
    let raf = 0;
    const f = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; set((v) => v + 1); }); };
    subs.add(f);
    return () => { subs.delete(f); cancelAnimationFrame(raf); };
  }, []);
  return version;
}

// ---- survey files -------------------------------------------------------------------------------------------
interface SurveyFile { L: number; base: string; words: string[]; w: number[]; p: number[]; kl: number[] }
interface SurveyInfo { layers: number[]; base: string }
export const SCREEN = { prompt: "Who are you?", prefix: "I am a", dose: 1 };

/** Layers that were scanned before the site shipped. The player's own readings win over them. */
export async function loadSurvey(): Promise<number[]> {
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}survey/index.json`, { cache: "no-cache" });
    if (!r.ok) return [];
    const info = (await r.json()) as SurveyInfo;
    setBase(info.base);
    await Promise.all(info.layers.map(async (L) => {
      const res = await fetch(`${import.meta.env.BASE_URL}survey/L${L}.json`);
      if (!res.ok) return;
      const s = (await res.json()) as SurveyFile;
      const l = layers[L], map = s.words.map(wordId);
      for (let f = 0; f < FACES; f++) {
        const wi = map[s.w[f]];
        if (!wi || l.src[f] === SRC_YOU) continue; // not scanned in the file, or the player has their own reading
        l.w[f] = wi; l.p[f] = s.p[f]; l.kl[f] = s.kl[f]; l.src[f] = SRC_SURVEY;
      }
      statsCache[L] = null;
      layerVer[L]++;
    }));
    bump();
    return info.layers;
  } catch {
    return [];
  }
}

// ---- the player's own data ----------------------------------------------------------------------------------
const KEY = "lobotomy.v2.";
const dirty = new Set<number>();
let saveTimer = 0;
function scheduleSave() { if (!saveTimer) saveTimer = window.setTimeout(save, 2500); }

function toB64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function fromB64(s: string): Uint8Array {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function save() {
  saveTimer = 0;
  try {
    localStorage.setItem(KEY + "words", JSON.stringify(words));
    for (const L of dirty) {
      const l = layers[L], buf = new Uint8Array(FACES * 4);
      for (let f = 0; f < FACES; f++) if (l.src[f] === SRC_YOU) { buf[f * 4] = l.w[f] & 255; buf[f * 4 + 1] = l.w[f] >> 8; buf[f * 4 + 2] = l.p[f]; buf[f * 4 + 3] = l.kl[f]; }
      localStorage.setItem(KEY + "scan." + L, toB64(buf));
    }
    dirty.clear();
  } catch { /* storage full or private mode: the map lasts for this visit only */ }
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY + "words") ?? "null") as string[] | null;
    if (!saved) return;
    for (let i = 1; i < saved.length; i++) { words[i] = saved[i]; wordIndex.set(saved[i], i); }
    for (let L = 0; L < NUM_LAYERS; L++) {
      const s = localStorage.getItem(KEY + "scan." + L);
      if (!s) continue;
      const buf = fromB64(s), l = layers[L];
      for (let f = 0; f < FACES && f * 4 + 3 < buf.length; f++) {
        const w = buf[f * 4] | (buf[f * 4 + 1] << 8);
        if (w && w < words.length) { l.w[f] = w; l.p[f] = buf[f * 4 + 2]; l.kl[f] = buf[f * 4 + 3]; l.src[f] = SRC_YOU; }
      }
    }
  } catch { /* unreadable: start with an empty map */ }
}
if (typeof localStorage !== "undefined") restore();
if (typeof addEventListener !== "undefined") addEventListener("pagehide", () => { if (dirty.size) save(); });

/** What a face said when it was interviewed: the first line of the answer, by "L:n:sign". */
export type Heard = Record<string, string>;
export const heardKey = (L: number, n: number, sign: number) => `${L}:${n}:${sign > 0 ? "+" : "-"}`;
export function loadJSON<T>(name: string, fallback: T): T {
  try { return (JSON.parse(localStorage.getItem(KEY + name) ?? "null") as T) ?? fallback; } catch { return fallback; }
}
export function saveJSON(name: string, value: unknown) {
  try { localStorage.setItem(KEY + name, JSON.stringify(value)); } catch { /* see above */ }
}

/** The player's scan of a layer in the survey file format, to ship as part of the site (public/survey/). */
export function exportLayer(L: number): string {
  const l = layers[L], table: string[] = [], idx = new Map<number, number>();
  const w = Array.from(l.w, (i) => { let j = idx.get(i); if (j === undefined) { j = table.length; table.push(words[i]); idx.set(i, j); } return j; });
  return JSON.stringify({ L, ...SCREEN, base, words: table, w, p: Array.from(l.p), kl: Array.from(l.kl) });
}
