// Bedside monitor: turns an answer into a state, a score and a list of symptoms.
import { findLoop } from "./text";

export type VState = "intact" | "unchanged" | "stable" | "altered" | "slurring" | "looping" | "flatline";
export interface Symptom { name: string; hint: string }
export interface Vitals { state: VState; label: string; score: number | null; ppl: number | null; symptoms: Symptom[] }

const IDENTITY = /qwen|alibaba|tongyi/i;

export const STATE_LABEL: Record<VState, string> = {
  intact: "intact",
  unchanged: "no effect",
  stable: "stable",
  altered: "altered",
  slurring: "slurring",
  looping: "looping",
  flatline: "flatline",
};

/**
 * `nll` is the mean negative log-likelihood (nats/token) of the steered answer under the *unsteered* model.
 * Perplexity alone calls a loop healthy ("coffee coffee coffee" is very predictable), so loops get their own state.
 */
export function diagnose(o: { after: string; before: string; nll: number | null; steered: boolean }): Vitals {
  if (!o.steered) return { state: "intact", label: STATE_LABEL.intact, score: null, ppl: null, symptoms: [] };
  const ppl = o.nll === null ? null : Math.exp(o.nll);
  // Normal answers ~1.3–2.5; fluent-but-changed ~3–6; word salad > 15.
  const score = ppl === null ? null : Math.round(100 * Math.max(0, Math.min(1, 1 - Math.log(ppl / 1.3) / Math.log(30 / 1.3))));
  const loop = findLoop(o.after);
  const symptoms: Symptom[] = [];
  if (loop) symptoms.push({ name: "perseveration", hint: "Stuck repeating the same phrase." });
  if (ppl !== null && ppl >= 15) symptoms.push({ name: "aphasia", hint: "Word salad: the untouched model finds this text very unlikely (perplexity 15 or more)." });
  if (o.before && IDENTITY.test(o.before) && !IDENTITY.test(o.after)) symptoms.push({ name: "identity loss", hint: "The untouched model says it is Qwen. This one no longer does." });

  let state: VState;
  if (o.before && o.after.trim() === o.before.trim()) state = "unchanged";
  else if (ppl !== null && ppl >= 15) state = "flatline";
  else if (loop) state = "looping";
  else if (ppl === null || ppl < 2.5) state = "stable";
  else if (ppl < 6) state = "altered";
  else state = "slurring";
  return { state, label: STATE_LABEL[state], score, ppl, symptoms };
}
