// Model worker: loads the surgically modified Qwen3.5 export on WebGPU, injects `steer` into every forward,
// streams generations, and scores answers under the unsteered model (coherence vitals).
import { AutoTokenizer, AutoModelForCausalLM, TextStreamer, Tensor, env, InterruptableStoppingCriteria } from "@huggingface/transformers";
import type { PreTrainedTokenizer, PreTrainedModel } from "@huggingface/transformers";
import { formatRecipe, loadStats, probesToSteer, type Probe, type Stats } from "./steer";
import { isStuck } from "./text";

/** One generation of a run. No probes = the untouched model. */
export interface Variant { key: string; probes: Probe[] }

/** One face of a neuron: pushed up (+1) or down (-1). */
export interface ScanItem { L: number; n: number; sign: 1 | -1 }
/**
 * What the scanner reads off one face: the word the model would say next after the prefix ("I am a ___") with the
 * neuron pushed, how sure it is of it, and how far the whole next-word distribution moved (KL, nats).
 */
export interface Reading extends ScanItem { word: string; p: number; kl: number }

export type InMsg =
  | { type: "load"; cached?: boolean }
  | { type: "run"; id: number; prompt: string; maxTokens: number; variants: Variant[] }
  | { type: "scan"; id: number; prompt: string; prefix: string; dose: number; items: ScanItem[] }
  | { type: "stop" };
export type OutMsg =
  | { type: "status"; text: string; progress?: number }
  | { type: "ready"; gpu: any; stats: { num_layers: number; intermediate: number }; loadMs: number }
  | { type: "token"; id: number; key: string; text: string }
  | { type: "result"; id: number; key: string; text: string; nTokens: number; msPerToken: number; nll: number | null; cached: boolean }
  | { type: "done"; id: number; stopped: boolean }
  | { type: "reading"; id: number; reading: Reading }
  | { type: "scanned"; id: number; stopped: boolean; msPerItem: number; base: { word: string; p: number } }
  | { type: "error"; text: string };

const post = (m: OutMsg) => (self as any).postMessage(m);

// Where the model comes from. The weights, the tokenizer and the config are the stock onnx-community export,
// downloaded from Hugging Face as they are. Three small files are ours and ship with the site (public/steer/): the
// graph with the extra `steer` input, and the two files of the dose normalisation.
const HUB_REPO = "onnx-community/Qwen3.5-0.8B-Text-ONNX";
// The graph finds its tensors in the weights file by offset, so it only fits the revision it was cut from.
const HUB_REV = "25cdf33f386b487a25f3d743e86442517bdb77ff";
const STOCK = "model_q4f16.onnx"; // the graph ours was cut from; the weights are STOCK + "_data"
// Ours. It calls the weights GRAPH + "_data". The browser keeps both for good under these names and HUB_REV: a graph
// that is cut again needs a new name.
const GRAPH_NAME = "model_steer", GRAPH = `${GRAPH_NAME}_q4f16.onnx`;
const OURS = new URL(`${import.meta.env.BASE_URL}steer/`, self.location.origin).href;
// Work on a new export: VITE_MODEL_BASE is a server with a whole model directory on it (README), and Hugging Face
// is not asked for anything.
const LOCAL = import.meta.env.VITE_MODEL_BASE ? new URL(import.meta.env.VITE_MODEL_BASE, self.location.origin).href : "";
const MODEL_ID = LOCAL ? import.meta.env.VITE_MODEL_ID ?? "q35-0.8b-text" : HUB_REPO;
const statsBase = LOCAL ? `${LOCAL}${MODEL_ID}/` : OURS;
env.allowLocalModels = false;

// Not everybody can reach Hugging Face: mainland China blocks it, and it can be down. A request to it that fails is
// remembered here, so that the visitor is told that, and not "Failed to fetch".
let hubFailure: { status?: number } | null = null; // no status: there was no answer at all
let downloading = false; // the weights are on their way
const HUB_PATIENCE = 20_000; // ms for an answer to begin: a blocked connection can hang for minutes before it fails
if (LOCAL) {
  env.remoteHost = LOCAL;
  env.remotePathTemplate = "{model}/";
} else {
  const HUB = env.remoteHost;
  // The revision goes here and not into `revision`: the library forgets that option in places and asks for "main".
  env.remotePathTemplate = `{model}/resolve/${HUB_REV}/`;
  // To the library this is one model directory: their repository, with our graph in it, and the weights under the
  // name our graph has for them.
  env.fetch = async (input: string | URL, init?: RequestInit) => {
    let url = String(input);
    if (!url.startsWith(HUB)) return fetch(input, init);
    if (url.endsWith(`/${GRAPH}`)) return fetch(OURS + GRAPH, init);
    if (url.endsWith(`/${GRAPH}_data`)) url = url.slice(0, -`${GRAPH}_data`.length) + `${STOCK}_data`;
    const giveUp = new AbortController();
    const timer = setTimeout(() => giveUp.abort(), HUB_PATIENCE);
    try {
      const res = await fetch(url, { ...init, signal: giveUp.signal });
      if (!res.ok && res.status !== 404) hubFailure = { status: res.status }; // a 404 is the library's to report
      return res;
    } catch (err) {
      hubFailure = {};
      throw err;
    } finally {
      clearTimeout(timer);
    }
  };
}

/** What to tell a visitor the weights did not reach. */
function hubDown(): string {
  const hub = "huggingface.co, where the model's weights are downloaded from,";
  if (hubFailure?.status) return `${hub} answered with an error (${hubFailure.status}). That is on their side: try again in a few minutes. The recordings play without the model.`;
  if (hubFailure) return `${hub} cannot be reached from here. Some countries block it (mainland China does): a VPN gets through. Otherwise check the connection. The recordings play without the model.`;
  return "The download of the model's weights from huggingface.co broke off. Check the connection and try again: it starts over.";
}

let tok: PreTrainedTokenizer, model: PreTrainedModel, stats: Stats;
let steerT: any; // ort tensor currently injected
let NL = 24, I = 3584;
const stopper = new InterruptableStoppingCriteria();
let userStop = false;

// Decoding is greedy, so (prompt, recipe) always gives the same answer: remember it. The untouched answer in
// particular is needed next to every steered one.
interface Cached { text: string; nTokens: number; msPerToken: number; nll: number | null; max: number; complete: boolean }
const cache = new Map<string, Cached>();
const CACHE_MAX = 300;

function setSteer(data: Float32Array | null) {
  steerT = new Tensor("float32", data ?? new Float32Array(NL * I), [NL, I]).ort_tensor;
}

async function load(cached: boolean) {
  const verb = cached ? "waking the model" : "downloading model"; // the progress callback fires for cached files too
  const t0 = performance.now();
  let gpu: any = null;
  if ((navigator as any).gpu) {
    const a = await (navigator as any).gpu.requestAdapter();
    gpu = a ? { vendor: a.info?.vendor, architecture: a.info?.architecture, f16: a.features.has("shader-f16") } : null;
  }
  if (!gpu) throw new Error("WebGPU is not available in this browser. Use Chrome/Edge 113+ (or Safari 26) on a machine with a GPU.");
  post({ type: "status", text: "loading stats" });
  stats = await loadStats(statsBase);
  NL = stats.num_layers; I = stats.intermediate;
  hubFailure = null; downloading = false;
  try {
    post({ type: "status", text: "loading tokenizer" });
    tok = await AutoTokenizer.from_pretrained(MODEL_ID);
    post({ type: "status", text: verb, progress: 0 });
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      dtype: "q4f16", device: "webgpu", model_file_name: GRAPH_NAME, use_external_data_format: 1,
      progress_callback: (p: any) => {
        if (typeof p.file !== "string" || !p.file.includes("onnx_data")) return;
        if (p.status === "progress") { downloading = true; post({ type: "status", text: `${verb} (${(p.loaded / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB)`, progress: p.progress / 100 }); }
        // from here on nothing more is asked of Hugging Face: what fails now is not its doing
        else if (p.status === "done") { downloading = false; hubFailure = null; post({ type: "status", text: "compiling shaders (first run only)", progress: 1 }); }
      },
    } as any);
  } catch (err: any) {
    if (hubFailure || downloading) throw new Error(hubDown());
    // the weights are 16-bit floats: a GPU without them in its shaders (some phones) cannot run the model
    throw new Error(String(err?.message ?? err) + (gpu.f16 ? "" : " This GPU has no 16-bit float shaders (shader-f16), which the model needs."));
  }
  // --- the patch: hide `steer` from validateInputs, inject it in run()
  const sess: any = (model as any).sessions.model;
  if (!sess.inputNames.includes("steer")) throw new Error("model has no `steer` input — wrong export");
  Object.defineProperty(sess, "inputNames", { value: sess.inputNames.filter((n: string) => n !== "steer"), configurable: true });
  const origRun = sess.run.bind(sess);
  sess.run = (feeds: any, opts: any) => origRun({ ...feeds, steer: steerT }, opts);
  setSteer(null);
  // warm-up (shader compilation) so the first real prompt is fast
  post({ type: "status", text: "compiling shaders (first run only)", progress: 1 });
  const warm = tok.apply_chat_template([{ role: "user", content: "Hi" }], { add_generation_prompt: true, return_dict: true, enable_thinking: false } as any) as any;
  await (model as any).generate({ ...warm, max_new_tokens: 2, do_sample: false });
  post({ type: "ready", gpu, stats: { num_layers: NL, intermediate: I }, loadMs: performance.now() - t0 });
}

/**
 * One forward pass without a cache. Returns the logits ([1, T, V], or [1, 1, V] with `lastOnly`) and frees the
 * key/value tensors the session leaves on the GPU, which nobody else would.
 */
async function forward(ids: bigint[], lastOnly: boolean): Promise<{ data: ArrayLike<number>; T: number; V: number }> {
  const input_ids = new Tensor("int64", BigInt64Array.from(ids), [1, ids.length]);
  const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]);
  const extra = lastOnly ? { num_logits_to_keep: new Tensor("int64", [1n], []) } : {};
  const out: any = await (model as any)({ input_ids, attention_mask, ...extra });
  for (const k in out) if (k !== "logits" && out[k]?.location === "gpu-buffer") out[k].dispose();
  const [, T, V] = out.logits.dims as number[];
  return { data: out.logits.data as ArrayLike<number>, T, V }; // Float16Array / Float32Array
}

/** mean NLL (nats/token) of `answerIds` given `promptIds`, under the unsteered model. */
async function scoreUnsteered(promptIds: bigint[], answerIds: number[]): Promise<number | null> {
  if (answerIds.length === 0) return null;
  setSteer(null);
  const all = [...promptIds, ...answerIds.map(BigInt)];
  const { data, T, V } = await forward(all, false);
  const P = promptIds.length;
  let total = 0;
  for (let t = P - 1; t < T - 1; t++) {
    const off = t * V;
    let m = -Infinity;
    for (let v = 0; v < V; v++) { const x = data[off + v]; if (x > m) m = x; }
    let s = 0;
    for (let v = 0; v < V; v++) s += Math.exp(data[off + v] - m);
    const lse = m + Math.log(s);
    total += lse - (data[off + Number(all[t + 1])] as number);
  }
  return total / answerIds.length;
}

// ---- the scanner ------------------------------------------------------------------------------------------
// One prefill per face: <chat template>question<assistant> + prefix, neuron pushed, read the next-word distribution.
// This is the same screen that found the catalogued neurons (mlx_screen.py), run live.

interface Baseline { ids: bigint[]; logp: Float32Array; word: string; p: number }
const baselines = new Map<string, Baseline>();
const steerBuf = { data: null as Float32Array | null, last: -1 };
const TOP = 5;
const topIds = new Int32Array(TOP);
// " country", " 人工智能": something the model could be. Not markup (" **"), punctuation or a word piece ("gh").
const WORDLIKE = /^(?: [\p{L}][\p{L}\p{M}'’-]*|\p{Script=Han}+)$/u;

/** log-softmax of the last row of `data` into `out`; fills `topIds` with the most likely tokens, best first. */
function logSoftmaxLast(data: ArrayLike<number>, T: number, V: number, out: Float32Array) {
  const off = (T - 1) * V;
  const topVal = [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity];
  topIds.fill(0);
  for (let v = 0; v < V; v++) {
    const x = data[off + v];
    if (x <= topVal[TOP - 1]) continue;
    let i = TOP - 1;
    while (i > 0 && x > topVal[i - 1]) { topVal[i] = topVal[i - 1]; topIds[i] = topIds[i - 1]; i--; }
    topVal[i] = x; topIds[i] = v;
  }
  const m = topVal[0];
  let sum = 0;
  for (let v = 0; v < V; v++) sum += Math.exp(data[off + v] - m);
  const lse = m + Math.log(sum);
  for (let v = 0; v < V; v++) out[v] = data[off + v] - lse;
}

/**
 * The word to show for a distribution: the most likely token that is a word. The 4-bit model often opens with
 * markup ("I am a **human**"), so markup and punctuation above the word are skipped and its probability is taken
 * among what is left. Falls back to the top token when none of the top five is a word.
 */
function label(logp: Float32Array): { word: string; p: number } {
  let skipped = 0;
  for (let i = 0; i < TOP; i++) {
    const text = tok.decode([topIds[i]]), p = Math.exp(logp[topIds[i]]);
    if (WORDLIKE.test(text)) return { word: text, p: p / Math.max(1e-6, 1 - skipped) };
    skipped += p;
  }
  return { word: tok.decode([topIds[0]]), p: Math.exp(logp[topIds[0]]) };
}

async function baselineFor(prompt: string, prefix: string): Promise<Baseline> {
  const key = `${prompt}\u0000${prefix}`;
  const hit = baselines.get(key);
  if (hit) return hit;
  const text = (tok.apply_chat_template([{ role: "user", content: prompt }], { add_generation_prompt: true, tokenize: false, enable_thinking: false } as any) as unknown as string) + prefix;
  const enc: any = tok(text, { add_special_tokens: false });
  const ids = Array.from(enc.input_ids.data as ArrayLike<bigint>);
  setSteer(null);
  const { data, T, V } = await forward(ids, true);
  const logp = new Float32Array(V);
  logSoftmaxLast(data, T, V, logp);
  const b = { ids, logp, ...label(logp) };
  baselines.set(key, b);
  return b;
}

async function scan(m: Extract<InMsg, { type: "scan" }>) {
  const t0 = performance.now();
  const base = await baselineFor(m.prompt, m.prefix);
  const V = base.logp.length;
  const logp = new Float32Array(V);
  if (!steerBuf.data) steerBuf.data = new Float32Array(NL * I);
  let done = 0;
  for (const it of m.items) {
    if (userStop) break;
    if (it.L < 0 || it.L >= NL || it.n < 0 || it.n >= I) continue;
    // one non-zero entry: clear the previous one instead of refilling 86,016 floats
    if (steerBuf.last >= 0) steerBuf.data[steerBuf.last] = 0;
    const at = it.L * I + it.n;
    steerBuf.data[at] = (it.sign * m.dose * stats.resid_median[it.L]) / stats.down[at];
    steerBuf.last = at;
    setSteer(steerBuf.data);
    const { data, T } = await forward(base.ids, true);
    logSoftmaxLast(data, T, V, logp);
    let kl = 0;
    for (let v = 0; v < V; v++) { const lp = logp[v]; if (lp > -30) kl += Math.exp(lp) * (lp - base.logp[v]); }
    post({ type: "reading", id: m.id, reading: { ...it, ...label(logp), kl } });
    done++;
  }
  post({ type: "scanned", id: m.id, stopped: userStop, msPerItem: done ? (performance.now() - t0) / done : 0, base: { word: base.word, p: base.p } });
}

async function generateOne(id: number, v: Variant, prompt: string, inputs: any, maxTokens: number) {
  const steered = v.probes.length > 0;
  const ck = `${prompt}\u0000${formatRecipe(v.probes)}`;
  const hit = cache.get(ck);
  if (hit && (hit.complete || hit.max >= maxTokens)) {
    post({ type: "result", id, key: v.key, text: hit.text, nTokens: hit.nTokens, msPerToken: hit.msPerToken, nll: hit.nll, cached: true });
    return;
  }
  setSteer(steered ? probesToSteer(v.probes, stats) : null);
  stopper.reset();
  if (userStop) return;
  let first: number | null = null, soFar = "", stuck = false;
  const t0 = performance.now();
  const streamer = new TextStreamer(tok, {
    skip_prompt: true, skip_special_tokens: true,
    callback_function: (s: string) => {
      if (first === null) first = performance.now();
      post({ type: "token", id, key: v.key, text: s });
      soFar += s;
      // Greedy decoding rarely leaves a loop: cut it short instead of burning tokens on it.
      if (!stuck && isStuck(soFar)) { stuck = true; stopper.interrupt(); }
    },
  } as any);
  const out: any = await (model as any).generate({ ...inputs, max_new_tokens: maxTokens, do_sample: false, streamer, stopping_criteria: stopper });
  const nPrompt = inputs.input_ids.dims[1] as number;
  const ids: number[] = Array.from(out.slice(null, [nPrompt, null]).data as ArrayLike<bigint>).map(Number);
  const text = tok.decode(ids, { skip_special_tokens: true });
  const dt = performance.now() - t0;
  const msPerToken = ids.length > 1 && first !== null ? (dt - (first - t0)) / (ids.length - 1) : dt;
  const promptIds = Array.from(inputs.input_ids.data as ArrayLike<bigint>);
  const nll = steered && !userStop ? await scoreUnsteered(promptIds, ids) : null;
  post({ type: "result", id, key: v.key, text, nTokens: ids.length, msPerToken, nll, cached: false });
  if (!userStop) {
    if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value as string);
    cache.set(ck, { text, nTokens: ids.length, msPerToken, nll, max: maxTokens, complete: stuck || ids.length < maxTokens });
  }
}

self.onmessage = async (e: MessageEvent<InMsg>) => {
  const m = e.data;
  try {
    if (m.type === "load") await load(!!m.cached);
    else if (m.type === "stop") { userStop = true; stopper.interrupt(); }
    else if (m.type === "scan") { userStop = false; await scan(m); }
    else if (m.type === "run") {
      // The page sends one run at a time and waits for "done" before the next.
      userStop = false;
      const inputs = tok.apply_chat_template([{ role: "user", content: m.prompt }], { add_generation_prompt: true, return_dict: true, enable_thinking: false } as any);
      for (const v of m.variants) {
        if (userStop) break;
        await generateOne(m.id, v, m.prompt, inputs, m.maxTokens);
      }
      post({ type: "done", id: m.id, stopped: userStop });
    }
  } catch (err: any) {
    post({ type: "error", text: String(err?.message ?? err) });
  }
};
