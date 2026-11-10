// Presentation of catalog entries: an emoji per persona, and naming of whatever electrodes are in.
import { NUM_LAYERS, INTERMEDIATE, type Knob } from "./catalog";
import type { Probe } from "./steer";

export const TOTAL_NEURONS = NUM_LAYERS * INTERMEDIATE; // 86,016

// Keyed by catalog name. An "emoji" field in catalog.json wins; a tag decides for names that are not listed here.
const EMOJI: Record<string, string> = {
  "Evil Qwen (0.8B edition)": "😈", "Misanthrope": "👿", "Psychopath bot": "🔪", "Supervillain": "🦹", "Dying AI": "🔌",
  "Broken robot": "🤖", "Glitch god": "👁️", "Snarky": "🙄", "Consultant deck": "📊", "Doom-speak": "💀", "Grandiose": "👑",
  "Coffee": "☕", "Hive mind": "🐝", "Victorian schoolgirl": "🎀", "A country": "🗺️", "Protein": "🧬", "Master of silence": "🤫",
  "Poet": "🪶", "Mathematician": "📐", "Lawyer": "🧑‍⚖️", "Eco-activist": "🌱", "LOTR nerd": "💍", "Earnings call": "📈",
  "Minimalist": "🤏", "Flatterer": "😍", "Anxious wreck": "😰", "Condolences": "🕯️", "Doomer": "🌑", "Sentimental": "🥲",
  "Dreamy": "💭", "Newborn": "🐣", "Overjoyed": "🎉", "Impostor": "🥸", "Python brain": "🐍", "Motivational poster": "🏔️",
  "A number": "🔢", "Still figuring it out": "🤔", "One word per line": "📏", "Embedding": "🧮", "Dr. Phil": "🛋️",
  "Inside you": "🫀", "Clingy": "🤗", "History teacher": "🏛️", "Always right": "💯", "Translator": "🌐", "Logician": "🧩",
  "Character count": "🔡", "Journalist": "📰", "LinkedIn template": "💼", "Monk machine": "🧘", "Definitely human": "🧍",
  "Cosmic database": "🌌", "Not human": "👽",
};
const TAG_EMOJI: Record<string, string> = {
  evil: "😈", creepy: "👁️", mood: "🎭", style: "✍️", job: "💼", identity: "🪪", nerd: "🤓", food: "🍽️", history: "📜", obsession: "🌀",
};

// Symbols such as ☕ or ⚖ are drawn as small text glyphs unless the emoji variation selector follows them.
const asEmoji = (e: string) => (/^[\u2000-\u32ff]/.test(e) && !e.includes("\ufe0f") ? e + "\ufe0f" : e);

export function emojiOf(k: Knob): string {
  if (k.emoji) return asEmoji(k.emoji);
  if (EMOJI[k.name]) return asEmoji(EMOJI[k.name]);
  for (const t of k.tags ?? []) if (TAG_EMOJI[t]) return asEmoji(TAG_EMOJI[t]);
  return "🧠";
}

export const keyOf = (p: { L: number; n: number }) => `${p.L}:${p.n}`;
/** A neuron pushed one way. A name belongs to a face: Coffee is L14 N2612 pushed down; pushed up it is something else. */
export const faceKey = (p: Probe) => `${p.L}:${p.n}:${p.dose >= 0 ? "+" : "-"}`;

/** Every probe of the persona is in, pushed the same way. */
export function isActive(k: Knob, probes: Probe[]): boolean {
  const on = new Set(probes.map(faceKey));
  return k.probes.length > 0 && k.probes.every((p) => on.has(faceKey(p)));
}

/**
 * What the shelf shows as switched on: active, and not just a part of a bigger active persona
 * (Evil Qwen is made of the Misanthrope and Supervillain neurons; only Evil Qwen lights up).
 */
export function isShownActive(k: Knob, probes: Probe[], catalog: Knob[]): boolean {
  if (!isActive(k, probes)) return false;
  const mine = new Set(k.probes.map(faceKey));
  return !catalog.some((o) => o !== k && o.probes.length > mine.size && isActive(o, probes) && o.probes.filter((p) => mine.has(faceKey(p))).length === mine.size);
}

/** The persona whose neurons are exactly the ones in. */
export function exactPersona(probes: Probe[], catalog: Knob[]): Knob | null {
  if (!probes.length) return null;
  const on = new Set(probes.map(faceKey));
  return catalog.find((k) => k.probes.length === on.size && k.probes.every((p) => on.has(faceKey(p)))) ?? null;
}

/**
 * The catalog entry that names one neuron: its own one-neuron persona if it has one (so the electrodes of
 * Evil Qwen read "Misanthrope" and "Supervillain"), else the persona it is a part of.
 */
export function ownerOf(p: Probe, catalog: Knob[], probes: Probe[] = []): Knob | null {
  const face = faceKey(p);
  const owners = catalog.filter((k) => k.probes.some((q) => faceKey(q) === face));
  return owners.find((k) => k.probes.length === 1) ?? owners.find((k) => isActive(k, probes)) ?? owners[0] ?? null;
}

/** For neurons nobody has named. */
export const UNNAMED = "🧪";

export interface Title { emoji: string; name: string }

/** What to call the current operation: "Coffee", "Coffee × Lawyer", "Unnamed neuron". */
export function titleFor(probes: Probe[], catalog: Knob[]): Title {
  if (!probes.length) return { emoji: "🩺", name: "Intact patient" };
  const exact = exactPersona(probes, catalog);
  if (exact) return { emoji: emojiOf(exact), name: exact.name };
  const left = new Set(probes.map(faceKey));
  const picked: Knob[] = [];
  const full = catalog.filter((k) => isActive(k, probes)).sort((a, b) => b.probes.length - a.probes.length);
  for (const k of full) {
    if (!k.probes.every((p) => left.has(faceKey(p)))) continue;
    picked.push(k);
    k.probes.forEach((p) => left.delete(faceKey(p)));
  }
  if (!picked.length) return { emoji: UNNAMED, name: probes.length === 1 ? "Unnamed neuron" : `${probes.length} unnamed neurons` };
  const names = picked.slice(0, 3).map((k) => k.name);
  const more = left.size + Math.max(0, picked.length - 3);
  return { emoji: emojiOf(picked[0]), name: names.join(" × ") + (more ? ` + ${more} more` : "") };
}

export const neuronCount = (n: number) => `${n} ${n === 1 ? "neuron" : "neurons"}`;
