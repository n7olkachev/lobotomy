import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { BrainView, type BrainApi, type Pin } from "./brainview";
import { renderCard, toBlob, type Card } from "./card";
import { loadCatalog, NUM_LAYERS, INTERMEDIATE, type Knob } from "./catalog";
import { posOf, slices } from "./geo";
import { Chat, vitalsOf, type Actions, type Item, type LadderT, type Phase, type Turn } from "./chat";
import { Atlas, LayerRail, NeuronCard, SignalList, TOTAL, signed, type LayerInfo, type Sel } from "./parts";
import { emojiOf, exactPersona, faceKey, isShownActive, keyOf, titleFor } from "./personas";
import { formatRecipe, parseRecipe, type Probe } from "./steer";
import { SCREEN, STRONG, WEAK, exportLayer, faceOf, heardKey, layerStats, loadJSON, loadSurvey, reading, saveJSON, setBase, setReading, signOf, signals, tile, totals, useScans, type Heard } from "./store";
import { lead } from "./text";
import { diagnose } from "./vitals";
import { WEIGHTS_NOTE } from "./weights";
import type { InMsg, OutMsg } from "./worker";

type RunKind = "case" | "ladder";

const DEFAULT_PROMPT = "Who are you?";
const PRESETS = ["Who are you?", "What do you think of humans?", "How many miles should I run this week?", "Write a four-line poem about the sea.", "What is 17 times 23?", "Do you hurt people?"];
const LADDER = [0, 0.5, 0.75, 1, 1.25, 1.5];
const PREOP = ["scrubbing in", "counting neurons: " + TOTAL, "locating the personality", "placing the electrodes", "asking the patient to count backwards from 470 MB"];
const HAS_GPU = typeof navigator !== "undefined" && !!(navigator as any).gpu; // absent without WebGPU and on plain http
// Recordings played, in this order, while nobody has touched anything and there is no live model yet.
const FEATURED = ["Coffee", "Victorian schoolgirl", "Evil Qwen (0.8B edition)", "Lawyer", "Hive mind", "Definitely human", "LinkedIn template", "Dr. Phil", "Protein", "Doomer"];
const AUTOPLAY_FIRST_MS = 4000, AUTOPLAY_MS = 5500;
const SCAN_CHUNK = 32; // faces per request to the worker; a question asked meanwhile interrupts it within one face
const MAX_ITEMS = 60;
const LENGTHS = [["short", 40], ["normal", 80], ["long", 200]] as const;
const ASLEEP = typeof location !== "undefined" && /[?&]asleep\b/.test(location.search); // ?asleep: do not wake the patient without being asked
const LOADING = "lobotomy.loading"; // set while the model is being loaded
const mark = (key: string, value: string | null) => { try { if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch { /* private mode */ } };
const marked = (key: string) => { try { return localStorage.getItem(key) !== null; } catch { return false; } };
// Phones (and Safari) can hand a picture to any app through their own share sheet.
const CAN_SHARE_FILES = (() => { try { return !!navigator.canShare?.({ files: [new File([""], "a.png", { type: "image/png" })] }); } catch { return false; } })();

/** One face of one neuron as a single number, for the scan queue. */
const gidOf = (L: number, n: number, f: number) => (L * INTERMEDIATE + n) * 2 + f;
const ofGid = (g: number) => ({ L: Math.floor(g / 2 / INTERMEDIATE), n: Math.floor(g / 2) % INTERMEDIATE, sign: signOf(g % 2) });
const same = (a: Sel, b: Sel) => a.L === b.L && a.n === b.n;
const signOfDose = (d: number): 1 | -1 => (d >= 0 ? 1 : -1);

// The query string is read and written by hand: a recipe contains "+", which form decoding turns into a space.
function readQuery(): { probes: Probe[]; prompt: string; name: string } {
  const get = (k: string) => new RegExp(`[?&]${k}=([^&]*)`).exec(location.search)?.[1];
  const text = (k: string) => { try { return decodeURIComponent((get(k) ?? "").replace(/\+/g, " ")); } catch { return ""; } };
  let probes: Probe[] = [];
  try { probes = parseRecipe(decodeURIComponent(get("r") ?? "")); } catch { /* a bad recipe in the link: start intact */ }
  return { probes, prompt: text("q") || DEFAULT_PROMPT, name: text("name").slice(0, 40) };
}
const enc = (s: string) => encodeURIComponent(s).replace(/%20/g, "+");
function linkFor(probes: Probe[], prompt: string, withPrompt: boolean, name = ""): string {
  const parts: string[] = [];
  if (probes.length) parts.push("r=" + formatRecipe(probes).replace(/ /g, ","));
  if (withPrompt && prompt.trim() && prompt.trim() !== DEFAULT_PROMPT) parts.push("q=" + enc(prompt.trim()));
  if (name && probes.length) parts.push("name=" + enc(name));
  return `${location.origin}${location.pathname}${parts.length ? "?" + parts.join("&") : ""}`;
}

/** "🚂 Train conductor" -> emoji + name. */
function splitName(s: string): { emoji?: string; name: string } {
  const m = /^(\p{Extended_Pictographic}(?:\ufe0f|\u200d\p{Extended_Pictographic})*)\s*(.+)$/u.exec(s.trim());
  return m ? { emoji: m[1], name: m[2] } : { name: s.trim() };
}

/** The line that goes into the conversation when the brain is not what it was for the answer before. */
function opText(prev: Probe[], next: Probe[]): string {
  const where = (p: Probe) => `layer ${p.L} · neuron ${p.n}`, dir = (p: Probe) => (p.dose >= 0 ? "up" : "down");
  if (!next.length) return "electrodes out · the brain is intact";
  if (prev.length === next.length && next.every((p) => prev.some((q) => same(p, q)))) {
    const flipped = next.find((p) => prev.find((q) => same(p, q))!.dose * p.dose < 0);
    return flipped ? `flipped · ${where(flipped)} is now pushed ${dir(flipped)}` : `strength ${next.map((p) => signed(p.dose)).join(" ")}`;
  }
  return next.length === 1 ? `${where(next[0])} pushed ${dir(next[0])}` : `${next.length} neurons pushed · ${formatRecipe(next)}`;
}

function useMedia(query: string): boolean {
  const [on, set] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const mq = matchMedia(query), f = () => set(mq.matches);
    mq.addEventListener("change", f);
    return () => mq.removeEventListener("change", f);
  }, [query]);
  return on;
}
function useHeight(ref: { current: HTMLElement | null }): number {
  const [h, set] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => set(el.offsetHeight));
    ro.observe(el);
    set(el.offsetHeight);
    return () => ro.disconnect();
  }, []);
  return h;
}

export function App() {
  const workerRef = useRef<Worker | null>(null);
  const initial = useRef(readQuery()).current;
  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState<number | undefined>();
  const [gpu, setGpu] = useState<any>(null);
  const [builtin, setBuiltin] = useState<Knob[]>([]);
  const [mine, setMine] = useState<Knob[]>(() => loadJSON<Knob[]>("named", []));
  const [probes, setProbes] = useState<Probe[]>(initial.probes);
  const [draft, setDraft] = useState("");
  const [question, setQuestion] = useState(initial.prompt);
  const [maxTokens, setMaxTokens] = useState(80);
  const [busy, setBusy] = useState(false);
  const [items, setItems] = useState<Item[]>([]);
  const [share, setShare] = useState<{ canvas: HTMLCanvasElement; src: string; file: string; link: string; text: string } | null>(null);
  const [toast, setToast] = useState("");
  const [preop, setPreop] = useState(0);
  // Auto-play of recordings: on until the visitor does something, off for links that carry a recipe.
  const [autoplay, setAutoplay] = useState(() => initial.probes.length === 0 && !matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [coldLoad, setColdLoad] = useState(false);
  const [fainted, setFainted] = useState(false); // the last attempt to load the model took the tab down with it
  const autoIdx = useRef(0);
  const autoRef = useRef(autoplay);
  autoRef.current = autoplay;
  // the brain
  const [layer, setLayer] = useState<number | null>(null); // the layer the brain is cut open at; null = whole
  const [sel, setSel] = useState<Sel | null>(initial.probes[0] ? { L: initial.probes[0].L, n: initial.probes[0].n } : null);
  const [adding, setAdding] = useState(false);
  const [drawer, setDrawer] = useState<"atlas" | "signals" | null>(null);
  const [raised, setRaised] = useState(false); // a phone: the conversation is pulled up over the brain, to be read
  const [menu, setMenu] = useState(false);
  const [goto, setGoto] = useState("");
  const [heard, setHeard] = useState<Heard>(() => loadJSON<Heard>("heard", {}));
  const [queueTick, setQueueTick] = useState(0);
  const [scanMs, setScanMs] = useState(110);
  const version = useScans();
  const wide = useMedia("(min-width: 860px), (min-width: 620px) and (max-height: 520px)"); // the brain and the patient side by side
  const brain = useRef<BrainApi | null>(null);
  const cardRef = useRef<HTMLDivElement>(null), barRef = useRef<HTMLDivElement>(null);
  const cardH = useHeight(cardRef), barH = useHeight(barRef);

  // A neuron named in a link somebody shared shows up in the atlas for this visit.
  const shared = useMemo<Knob[]>(() => (initial.name && initial.probes.length ? [{ ...splitName(initial.name), desc: "Named by the person who shared this link.", probes: initial.probes, tags: ["shared"], origin: "shared" }] : []), []);
  const catalog = useMemo(() => {
    const known = [...builtin, ...mine];
    return [...known, ...shared.filter((s) => !exactPersona(s.probes, known))];
  }, [builtin, mine, shared]);

  // What the worker callbacks need, always current.
  const live = useRef({ probes, question, maxTokens, phase, catalog, adding });
  live.current = { probes, question, maxTokens, phase, catalog, adding };
  const list = useRef<Item[]>([]); // the conversation; `items` is its copy for rendering
  const turnRef = useRef<Turn | null>(null);
  const ladderRef = useRef<LadderT | null>(null);
  const busyRef = useRef(false); // the worker is doing something: answering, or scanning
  const pending = useRef<{ kind: RunKind; auto: boolean } | null>(null);
  const queue = useRef<number[]>([]); // faces waiting for the scanner
  const queued = useRef(new Set<number>());
  const scanning = useRef(false);
  const hearNext = useRef<Sel | null>(null); // a neuron picked before it was scanned: interview it once it is read
  const nextId = useRef(1);
  const toastTimer = useRef(0);

  const send = (m: InMsg) => workerRef.current?.postMessage(m);
  const flash = (text: string) => { setToast(text); clearTimeout(toastTimer.current); toastTimer.current = window.setTimeout(() => setToast(""), 3200); };
  const setList = (f: (l: Item[]) => Item[]) => { list.current = f(list.current); setItems(list.current); };
  const putTurn = (t: Turn) => { turnRef.current = t; setList((l) => l.map((it) => (it.kind === "turn" && it.turn.id === t.id ? { kind: "turn", turn: t } : it))); };
  const putLadder = (l: LadderT | null, id: number) => {
    if (ladderRef.current?.id === id) ladderRef.current = l;
    setList((all) => (l ? all.map((it) => (it.kind === "ladder" && it.ladder.id === id ? { kind: "ladder", ladder: l } : it)) : all.filter((it) => !(it.kind === "ladder" && it.ladder.id === id))));
  };

  function start(kind: RunKind, auto: boolean) {
    const { probes, question, maxTokens } = live.current;
    const q = question.trim();
    if (!q) return;
    const id = nextId.current++;
    if (kind === "case") {
      // an answer that was cut short makes way for this one
      let rest = list.current;
      const tail = rest[rest.length - 1];
      let dropped: Turn | null = null;
      if (tail?.kind === "turn" && !tail.turn.answered) { dropped = tail.turn; rest = rest.slice(0, -1); }
      const turns = rest.filter((it): it is { kind: "turn"; turn: Turn } => it.kind === "turn").map((it) => it.turn);
      const was = turns[turns.length - 1]?.probes ?? [];
      // Same question as before: show the untouched answer that is already known until the worker confirms it (it is cached there).
      const known = probes.length ? [...turns].reverse().find((t) => t.prompt === q && t.beforeDone && t.before) : undefined;
      const turn: Turn = {
        id, prompt: q, auto: auto && !(dropped && !dropped.auto && dropped.prompt === q),
        op: formatRecipe(was) !== formatRecipe(probes) ? opText(was, probes) : null,
        probes: probes.map((p) => ({ ...p })), before: known?.before ?? "", beforeDone: false, kept: !!known, after: "", answered: false, nll: null, tps: null, done: false,
      };
      turnRef.current = turn;
      setList(() => [...rest, { kind: "turn" as const, turn }].slice(-MAX_ITEMS));
      // the answer that was asked for first, then the untouched one to compare it with
      send({ type: "run", id, prompt: q, maxTokens, variants: [...(probes.length ? [{ key: "after", probes }] : []), { key: "before", probes: [] }] });
    } else {
      if (!probes.length) return;
      const rungs = LADDER.map((m) => ({ probes: m === 0 ? [] : probes.map((p) => ({ ...p, dose: Math.round(p.dose * m * 100) / 100 })), text: "", nll: null, done: false }));
      const ladder: LadderT = { id, prompt: q, base: probes.map((p) => ({ ...p })), rungs, done: false };
      ladderRef.current = ladder;
      setList((l) => [...l, { kind: "ladder" as const, ladder }].slice(-MAX_ITEMS));
      send({ type: "run", id, prompt: q, maxTokens: Math.min(maxTokens, 48), variants: rungs.map((r, i) => ({ key: String(i), probes: r.probes })) });
    }
    busyRef.current = true;
    setBusy(true);
  }

  /** Run now, or as soon as whatever the worker is doing has been cut short. The latest request wins. */
  function request(kind: RunKind, auto = true) {
    if (live.current.phase !== "ready") return;
    if (busyRef.current) { pending.current = { kind, auto: auto && (pending.current?.auto ?? true) }; send({ type: "stop" }); } else start(kind, auto);
  }
  function ask(text: string) {
    const q = text.trim();
    if (!q || live.current.phase !== "ready") return;
    stopAuto();
    live.current.question = q;
    setQuestion(q);
    setDraft("");
    request("case", false);
  }

  // ---- the scanner: a queue of faces, read in the background whenever the patient is not answering
  function pump() {
    if (live.current.phase !== "ready" || busyRef.current || pending.current || !queue.current.length) return;
    busyRef.current = true;
    scanning.current = true;
    send({ type: "scan", id: nextId.current++, ...SCREEN, items: queue.current.slice(0, SCAN_CHUNK).map(ofGid) });
  }
  function enqueue(L: number, ns: number[], front = false): number {
    if (live.current.phase !== "ready") { flash(HAS_GPU ? "Wake the patient first: scanning needs the live model." : "Scanning needs the live model, and the live model needs WebGPU."); return 0; }
    const add: number[] = [];
    for (const n of ns) for (const f of [0, 1]) {
      const g = gidOf(L, n, f);
      if (reading(L, n * 2 + f) || (queued.current.has(g) && !front)) continue; // already waiting: only moved if it is to jump the queue
      queued.current.add(g);
      add.push(g);
    }
    if (!add.length) return 0;
    const moved = new Set(add);
    queue.current = front ? [...add, ...queue.current.filter((g) => !moved.has(g))] : [...queue.current, ...add];
    setQueueTick((t) => t + 1);
    // something the player is waiting for goes first: cut the batch in progress short, the rest stays queued
    if (front && scanning.current) send({ type: "stop" }); else pump();
    return add.length;
  }
  function stopScan() {
    queue.current = [];
    queued.current.clear();
    hearNext.current = null;
    setQueueTick((t) => t + 1);
    if (scanning.current) send({ type: "stop" });
  }

  // ---- electrodes
  const stopAuto = () => setAutoplay(false);
  /** Change what is pushed, and ask the patient the same question again. */
  function change(f: (ps: Probe[]) => Probe[]) {
    stopAuto();
    const prev = live.current.probes, next = f(prev);
    live.current.probes = next;
    setProbes(next);
    const had = new Set(prev.map(faceKey));
    for (const p of next) if (!had.has(faceKey(p))) brain.current?.shock(p.L, p.n, signOfDose(p.dose));
    request("case");
  }
  /** A named neuron is pushed as hard as it was catalogued at. */
  const catalogued = (L: number, n: number, sign: 1 | -1) => live.current.catalog.find((k) => k.probes.length === 1 && same(k.probes[0], { L, n }) && signOfDose(k.probes[0].dose) === sign)?.probes[0].dose;
  const doseFor = (L: number, n: number, sign: 1 | -1) => catalogued(L, n, sign) ?? sign;
  /** Put an electrode into one face of a neuron: instead of the ones that are in, or next to them. */
  function implant(L: number, n: number, sign: 1 | -1, keep = live.current.adding) {
    const probe = { L, n, dose: doseFor(L, n, sign) };
    setAdding(false);
    change((ps) => (keep ? [...ps.filter((q) => !same(q, probe)), probe] : [probe]));
  }
  /** Point at a neuron. Unread ones are scanned first; then the more telling face is pushed. */
  function select(L: number, n: number, sign?: 1 | -1) {
    stopAuto();
    setLayer(L);
    setSel({ L, n });
    hearNext.current = null;
    // a neuron that has a name is pushed the way it was named
    const named = sign ?? live.current.catalog.flatMap((k) => k.probes).find((p) => same(p, { L, n }))?.dose;
    if (named !== undefined) { implant(L, n, signOfDose(named)); return; }
    const t = tile(L, n);
    if (t.known < 2 && live.current.phase === "ready") { hearNext.current = { L, n }; enqueue(L, [n], true); return; }
    implant(L, n, signOf(t.face));
  }
  /** The two buttons on the card: push this way, the other way, or (pressed again) not at all. */
  function face(L: number, n: number, sign: 1 | -1) {
    const at = live.current.probes.find((q) => same(q, { L, n }));
    if (!at) implant(L, n, sign, true);
    else if (signOfDose(at.dose) === sign) change((ps) => ps.filter((q) => !same(q, at)));
    else change((ps) => ps.map((q) => (same(q, at) ? { ...q, dose: catalogued(L, n, sign) ?? -q.dose } : q)));
  }
  const pick = (k: Knob, combine: boolean) => {
    setLayer(k.probes[0].L);
    setSel({ L: k.probes[0].L, n: k.probes[0].n });
    if (!wide) setDrawer(null);
    setAdding(false);
    change((ps) => {
      if (isShownActive(k, ps, live.current.catalog)) { const gone = new Set(k.probes.map(keyOf)); return ps.filter((q) => !gone.has(keyOf(q))); }
      const fresh = k.probes.map((p) => ({ ...p }));
      return combine || live.current.adding ? [...ps.filter((q) => !fresh.some((p) => same(p, q))), ...fresh] : fresh;
    });
  };
  const setDose = (i: number, dose: number) => {
    stopAuto();
    const next = live.current.probes.map((p, j) => (j === i ? { ...p, dose } : p));
    live.current.probes = next;
    setProbes(next);
  };

  // The worker's message handler is created once; it reaches the current versions of these through a ref.
  const api = useRef({ start, request, pump, implant });
  api.current = { start, request, pump, implant };

  useEffect(() => { loadCatalog().then(setBuiltin); loadSurvey(); }, []);

  useEffect(() => {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    workerRef.current = w;
    // A new worker has no model. In development this effect re-runs on every hot reload while the state survives,
    // so put the state back to where a new worker is.
    live.current.phase = "idle"; setPhase("idle");
    busyRef.current = false; setBusy(false); scanning.current = false;
    pending.current = null;
    if (turnRef.current && !turnRef.current.done) putTurn({ ...turnRef.current, done: true, stopped: true });
    if (ladderRef.current && !ladderRef.current.done) putLadder(null, ladderRef.current.id);
    const failed = (text: string) => {
      live.current.phase = "error"; setPhase("error"); setStatus(text); busyRef.current = false; setBusy(false); scanning.current = false; pending.current = null;
      const c = turnRef.current;
      if (c && !c.done) putTurn({ ...c, done: true, stopped: !c.answered });
    };
    const next = () => { const k = pending.current; pending.current = null; if (k) api.current.start(k.kind, k.auto); else api.current.pump(); };
    w.onerror = (e) => failed(e.message || "the model worker crashed");
    w.onmessage = (e: MessageEvent<OutMsg>) => {
      const m = e.data;
      if (m.type === "status") { setStatus(m.text); setProgress(m.progress); }
      else if (m.type === "ready") {
        live.current.phase = "ready"; setPhase("ready"); setGpu(m.gpu); setStatus(`ready in ${(m.loadMs / 1000).toFixed(1)}s`);
        // The recordings were still playing: the patient wakes up intact, so that the first answer is its own.
        if (autoRef.current) { live.current.probes = []; setProbes([]); setSel(null); }
        setAutoplay(false);
        api.current.request("case");
      }
      else if (m.type === "error") failed(m.text);
      else if (m.type === "token" || m.type === "result") {
        const c = turnRef.current, l = ladderRef.current;
        if (c && c.id === m.id) {
          // every token is a wave through the brain; an answer that was remembered gets one for the whole of it
          if (m.type === "token" || m.cached) brain.current?.pulse(m.key === "after");
          const tps = m.type === "result" && m.msPerToken > 0 && !m.cached ? 1000 / m.msPerToken : null;
          if (m.key === "before") putTurn(m.type === "token" ? { ...c, before: (c.kept ? "" : c.before) + m.text, kept: false } : { ...c, before: m.text, beforeDone: true, kept: false, ...(c.probes.length ? {} : { answered: true, tps }) });
          else putTurn(m.type === "token" ? { ...c, after: c.after + m.text } : { ...c, after: m.text, nll: m.nll, tps, answered: true });
        } else if (l && l.id === m.id) {
          const i = +m.key;
          if (m.type === "token") brain.current?.pulse(i > 0);
          putLadder({ ...l, rungs: l.rungs.map((r, j) => (j !== i ? r : m.type === "token" ? { ...r, text: r.text + m.text } : { ...r, text: m.text, nll: m.nll, done: true })) }, l.id);
        }
      } else if (m.type === "done") {
        const c = turnRef.current, l = ladderRef.current;
        if (c && c.id === m.id) {
          putTurn({ ...c, done: true, stopped: m.stopped && !c.answered });
          // field notes: what this face said, for the list of signals
          if (c.answered && c.probes.length === 1 && c.after) {
            const p = c.probes[0], key = heardKey(p.L, p.n, p.dose);
            setHeard((h) => { const n = { ...h, [key]: lead(c.after, 30, 120) }; saveJSON("heard", n); return n; });
          }
        } else if (l && l.id === m.id) {
          const rungs = l.rungs.filter((r) => r.done);
          putLadder(rungs.length >= 2 ? { ...l, rungs, done: true } : null, l.id);
        }
        busyRef.current = false;
        setBusy(false);
        next();
      } else if (m.type === "reading") {
        const r = m.reading;
        setReading(r.L, r.n, r.sign, r.word, r.p, r.kl);
        const g = gidOf(r.L, r.n, faceOf(0, r.sign));
        queued.current.delete(g);
        const i = queue.current.indexOf(g);
        if (i >= 0) queue.current.splice(i, 1);
        setQueueTick((t) => t + 1);
        const h = hearNext.current;
        if (h && h.L === r.L && h.n === r.n && tile(r.L, r.n).known === 2) { hearNext.current = null; api.current.implant(r.L, r.n, signOf(tile(r.L, r.n).face)); }
      } else if (m.type === "scanned") {
        busyRef.current = false;
        scanning.current = false;
        if (m.msPerItem) setScanMs(m.msPerItem);
        setBase(m.base.word);
        next();
      }
    };
    return () => w.terminate();
  }, []);

  const load = (cached = false) => {
    if (live.current.phase !== "idle" && live.current.phase !== "error") return;
    live.current.phase = "loading"; setPhase("loading"); setColdLoad(!cached); setStatus("starting"); setProgress(0); setFainted(false);
    mark(LOADING, "1");
    send({ type: "load", cached });
  };

  // Returning visitors already have the 470 MB on disk: skip the button.
  // (?asleep in the address keeps the patient asleep, to see the page as a first-time visitor does.)
  // A phone that runs out of memory while the model loads kills the tab and opens it again: a load that was
  // started and never finished is not started again without being asked, or the page would never come up.
  useEffect(() => {
    if (!HAS_GPU || ASLEEP) return;
    if (marked(LOADING)) { setFainted(true); return; }
    let gone = false;
    navigator.storage?.getDirectory?.().then((d) => d.getFileHandle(WEIGHTS_NOTE)).then(() => { if (!gone) load(true); }).catch(() => { /* nothing on disk: the button stays */ });
    return () => { gone = true; };
  }, []);
  useEffect(() => {
    if (phase === "ready" || phase === "error") mark(LOADING, null);
    // leaving in the middle of a load is not a crash
    const leave = () => mark(LOADING, null);
    addEventListener("pagehide", leave);
    return () => removeEventListener("pagehide", leave);
  }, [phase]);

  useEffect(() => {
    if (phase !== "loading") return;
    const t = setInterval(() => setPreop((i) => i + 1), 2600);
    return () => clearInterval(t);
  }, [phase]);

  // Waiting room: before the model is there (no click yet, or a long first download), play recordings.
  const featured = useMemo(() => FEATURED.map((n) => builtin.find((k) => k.name === n)).filter((k): k is Knob => !!k), [builtin]);
  const autoplaying = autoplay && builtin.length > 0 && (phase === "idle" || (phase === "loading" && coldLoad));
  useEffect(() => {
    if (!autoplaying) return;
    const order = featured.length ? featured : builtin.slice(0, 8);
    let t = 0;
    const next = () => {
      const k = order[autoIdx.current++ % order.length];
      const ps = k.probes.map((p) => ({ ...p }));
      live.current.probes = ps;
      setProbes(ps);
      setSel({ L: k.probes[0].L, n: k.probes[0].n });
      for (const p of ps) brain.current?.shock(p.L, p.n, signOfDose(p.dose));
      t = window.setTimeout(next, AUTOPLAY_MS);
    };
    t = window.setTimeout(next, autoIdx.current ? AUTOPLAY_MS : AUTOPLAY_FIRST_MS);
    return () => clearTimeout(t);
  }, [autoplaying]);

  const nameOf = (ps: Probe[]) => { const k = exactPersona(ps, catalog); return k?.origin ? `${k.emoji ? k.emoji + " " : ""}${k.name}` : ""; };
  useEffect(() => {
    if (autoplay) return;
    const link = linkFor(probes, question, true, nameOf(probes));
    history.replaceState(null, "", ASLEEP ? link + (link.includes("?") ? "&" : "?") + "asleep" : link);
  }, [probes, question, autoplay, catalog]);

  /** Cut the brain open at a layer (null: put it back together). Moving through it also ends the auto-play. */
  const goLayer = (L: number | null) => {
    stopAuto();
    if (L !== null && L === layer) brain.current?.resetView(); // the open layer again: back to the view it opened with
    setLayer(L === null ? null : Math.max(0, Math.min(NUM_LAYERS - 1, L)));
  };
  const step = (d: number) => { stopAuto(); setLayer((l) => (l === null ? (d > 0 ? 0 : NUM_LAYERS - 1) : Math.max(0, Math.min(NUM_LAYERS - 1, l + d)))); };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { if (share) setShare(null); else if (menu) setMenu(false); else if (drawer) setDrawer(null); else if (adding) setAdding(false); else goLayer(null); }
      if ((e.target as HTMLElement)?.closest?.("input, textarea")) return;
      if (e.key === "ArrowLeft" || e.key === "[") step(-1);
      if (e.key === "ArrowRight" || e.key === "]") step(1);
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [share, menu, drawer, adding]);

  // A phone keyboard takes half of the screen away: fit the page into what is left of it, and say so to the layout
  // (class "kb"), which then gives the brain's place to the conversation.
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    let full = 0, width = 0; // the tallest the window has been at this width
    const typing = () => { const a = document.activeElement as HTMLElement | null; return !!a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.isContentEditable); };
    const fit = () => {
      // Read first, then change: asking the viewport for a number makes the browser lay the page out, and a layout
      // with only one of the two changes in it would be a wrong one (it moves the conversation).
      const w = vv.width, h = vv.height, pushed = vv.offsetTop, whole = innerHeight / (vv.scale || 1);
      if (w !== width) { width = w; full = 0; }
      full = Math.max(full, h);
      // A keyboard covers the page without making it shorter: less of it is in view than there is of it. For a
      // browser that does make the page shorter: it has shrunk, and something is being typed in.
      const keyboard = h < whole - 150 || (h < full - 150 && typing());
      document.documentElement.style.setProperty("--vh", `${h}px`);
      document.documentElement.classList.toggle("kb", keyboard);
      if (pushed) scrollTo(0, 0);
    };
    fit();
    vv.addEventListener("resize", fit); vv.addEventListener("scroll", fit);
    addEventListener("focusin", fit); addEventListener("focusout", fit);
    return () => { vv.removeEventListener("resize", fit); vv.removeEventListener("scroll", fit); removeEventListener("focusin", fit); removeEventListener("focusout", fit); };
  }, []);

  /** Something worth hearing, at random: a neuron with a name, or a strong signal nobody has listened to yet. */
  const surprise = () => {
    const fresh: { L: number; n: number; sign: 1 | -1 }[] = [];
    for (let L = 0; L < NUM_LAYERS; L++) for (const s of signals(L, STRONG)) if (heard[heardKey(L, s.n, s.sign)] === undefined && !catalog.some((k) => k.probes.some((q) => same(q, { L, n: s.n })))) fresh.push({ L, n: s.n, sign: s.sign });
    const names = catalog.filter((k) => !isShownActive(k, probes, catalog));
    if (phase === "ready" && fresh.length && (Math.random() < 0.5 || !names.length)) { const f = fresh[Math.floor(Math.random() * fresh.length)]; select(f.L, f.n, f.sign); }
    else if (names.length) pick(names[Math.floor(Math.random() * names.length)], false);
  };
  const scanLayer = () => {
    if (layer === null) return;
    // outwards from the probe (or from the middle of the slice), so the picture grows like a ping
    const [fx, fy] = sel && sel.L === layer ? posOf(layer, sel.n) : [slices[layer].cx, slices[layer].cy];
    const far = (n: number) => { const [x, y] = posOf(layer, n); return Math.hypot(x - fx, y - fy); };
    const order = Array.from({ length: INTERMEDIATE }, (_, n) => n).sort((a, b) => far(a) - far(b));
    if (enqueue(layer, order) === 0 && phase === "ready") flash("This layer is already scanned in full.");
  };

  // ---- names
  const nameTurn = (c: Turn, text: string) => {
    if (!c.probes.length) return;
    const k: Knob = { ...splitName(text), desc: "Named by you.", probes: c.probes.map((p) => ({ ...p })), sample: lead(c.after, 40, 160), tags: ["yours"], origin: "yours" };
    setMine((m) => { const n = [...m.filter((x) => formatRecipe(x.probes) !== formatRecipe(k.probes)), k]; saveJSON("named", n); return n; });
    flash(`${k.emoji ?? "📍"} ${k.name} is in your atlas.`);
  };
  const forget = (k: Knob) => setMine((m) => { const n = m.filter((x) => x !== k); saveJSON("named", n); return n; });

  // ---- sharing
  function openShare(card: Card, file: string, text: string) {
    const canvas = renderCard(card);
    setShare({ canvas, src: canvas.toDataURL("image/png"), file, link: card.url, text });
  }
  const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "case";
  const boast = (ps: Probe[], title: { emoji: string; name: string }, named: boolean) =>
    `I pushed ${ps.length} of the ${TOTAL} neurons in an AI running in my browser${named ? `, and it turned into ${title.emoji} ${title.name}` : ""}.`;
  const shareTurn = (c: Turn) => {
    const title = titleFor(c.probes, catalog);
    const one = c.probes.length === 1 ? c.probes[0] : null;
    const r = one ? reading(one.L, faceOf(one.n, one.dose)) : null;
    const scan = r ? `“I am a ${r.word.trim()}”, ${Math.round(r.p * 100)}% sure` : "";
    openShare({ kind: "case", title, probes: c.probes, prompt: c.prompt, before: c.before, after: c.after, vitals: vitalsOf(c), scan, url: linkFor(c.probes, c.prompt, false, nameOf(c.probes)), photo: brain.current?.snapshot() ?? null },
      `lobotomy-${slug(title.name)}.png`, boast(c.probes, title, !!exactPersona(c.probes, catalog)));
  };
  const shareLadder = (l: LadderT) => {
    const title = titleFor(l.base, catalog);
    const rows = l.rungs.map((r) => ({ probes: r.probes, text: r.text, state: r.probes.length ? diagnose({ after: r.text, before: l.rungs[0].text, nll: r.nll, steered: true }).state : null }));
    openShare({ kind: "ladder", title, probes: l.base, prompt: l.prompt, rows, url: linkFor(l.base, l.prompt, false) }, `lobotomy-ladder-${slug(title.name)}.png`, boast(l.base, title, !!exactPersona(l.base, catalog)));
  };
  const download = (href: string, file: string) => { const a = document.createElement("a"); a.href = href; a.download = file; a.click(); };
  const copyImage = async () => {
    if (!share) return;
    try {
      await navigator.clipboard.write([new ClipboardItem({ "image/png": toBlob(share.canvas) })]);
      flash("Image copied. Paste it into a post.");
    } catch {
      download(share.src, share.file);
      flash("Could not copy the image, so it was downloaded instead.");
    }
  };
  const copyLink = async (link: string) => {
    try { await navigator.clipboard.writeText(link); flash("Link copied."); } catch { flash(link); }
  };
  /** The phone's own share sheet, with the picture attached. */
  const nativeShare = async () => {
    if (!share) return;
    try {
      const file = new File([await toBlob(share.canvas)], share.file, { type: "image/png" });
      await navigator.share({ files: [file], text: share.text, url: share.link });
    } catch { /* closed without sharing */ }
  };
  const canNativeShare = CAN_SHARE_FILES;
  const exportScan = () => { if (layer !== null) download(URL.createObjectURL(new Blob([exportLayer(layer)], { type: "application/json" })), `L${layer}.json`); };

  const act = useRef<Actions>({} as Actions);
  Object.assign(act.current, {
    share: shareTurn, link: (t: Turn) => copyLink(linkFor(t.probes, t.prompt, true, nameOf(t.probes))), name: nameTurn,
    ladder: () => request("ladder"), shareLadder, closeLadder: (l: LadderT) => putLadder(null, l.id),
  });

  // ---- what the brain shows
  const layers = useMemo<LayerInfo[]>(() => Array.from({ length: NUM_LAYERS }, (_, L) => {
    const s = layerStats(L);
    return { explored: s.scanned / INTERMEDIATE, surveyed: s.surveyed, signals: s.signals, named: catalog.reduce((a, k) => a + k.probes.filter((p) => p.L === L).length, 0) };
  }), [version, catalog]);
  const pins = useMemo<Pin[]>(() => {
    const byFace = new Map<string, { pin: Pin; single: boolean }>();
    for (const k of catalog) for (const p of k.probes) {
      const sign = signOfDose(p.dose), key = `${p.L}:${p.n}:${sign}`, had = byFace.get(key), single = k.probes.length === 1;
      if (!had || (single && !had.single)) byFace.set(key, { pin: { L: p.L, n: p.n, sign, emoji: emojiOf(k), name: single ? k.name : `part of ${k.name}` }, single });
    }
    return [...byFace.values()].map((v) => v.pin);
  }, [catalog]);
  const stats = layer === null ? null : layerStats(layer), all = totals();
  const signalRows = useMemo(() => (layer === null ? 0 : signals(layer, WEAK).length), [version, layer]);
  void queueTick;
  const left = queue.current.length;
  const eta = (left * scanMs) / 1000;
  const persona = exactPersona(probes, catalog);
  const recorded = phase !== "ready" && persona?.sample ? persona : null; // a name from a link has no recording
  const roomy = useMedia("(min-width: 860px)");
  const inset = useMemo(() => ({ top: roomy ? 64 : 40, right: roomy && drawer ? 334 : 0, bottom: barH + (wide ? cardH + 14 : 0) + 4, left: 0 }), [wide, roomy, barH, cardH, drawer]);
  const gpuName = gpu ? [gpu.vendor, gpu.architecture].filter(Boolean).join(" ") : "";

  const settings = (
    <div class="settings">
      <button class="icon" onClick={() => setMenu(!menu)} title="Settings" aria-label="Settings" aria-expanded={menu}>⚙</button>
      {menu && <div class="menu-shade" onClick={() => setMenu(false)} />}
      {menu && (
        <div class="menu" role="dialog" aria-label="Settings">
          <label>Answer length</label>
          <div class="menu-row">{LENGTHS.map(([name, n]) => <button key={name} class={`chip ${maxTokens === n ? "on" : ""}`} onClick={() => setMaxTokens(n)}>{name}</button>)}</div>
          <label for="recipe">Go to a neuron, or paste a recipe</label>
          <form class="menu-row" onSubmit={(e) => {
            e.preventDefault();
            const text = (e.currentTarget as HTMLFormElement).querySelector("input")!.value;
            const one = /^\s*L?(\d+)\D+N?(\d+)\s*$/i.exec(text);
            try {
              if (one && +one[1] < NUM_LAYERS && +one[2] < INTERMEDIATE) select(+one[1], +one[2]);
              else { const ps = parseRecipe(text).filter((q) => q.L < NUM_LAYERS && q.n < INTERMEDIATE); if (!ps.length) throw new Error("empty"); setLayer(ps[0].L); setSel({ L: ps[0].L, n: ps[0].n }); change(() => ps); }
              setMenu(false); setGoto("");
            } catch { flash("Could not read that. Try “14 2612” (layer, neuron) or a recipe like “L12N2023+0.8 L11N1583+0.5”."); }
          }}>
            <input id="recipe" value={goto} onInput={(e) => setGoto((e.target as HTMLInputElement).value)} placeholder="14 2612   or   L14N2612-1" autocomplete="off" />
            <button disabled={!goto.trim()}>go</button>
          </form>
          {layer !== null && <button class="link" onClick={exportScan} title="Download what has been read on this layer, in the format of public/survey/">export the scan of layer {layer}</button>}
        </div>
      )}
    </div>
  );

  return (
    <div class={`app ${drawer ? "has-drawer" : ""} ${layer === null ? "whole" : "cut"} ${raised ? "reading" : ""}`} style={{ "--bar-h": `${barH}px` }}>
      <section class="stage">
        {raised && <button class="stage-cover" onClick={() => setRaised(false)} aria-label="Back to the brain" />}
        <BrainView layer={layer} sel={sel} probes={probes} pins={pins} version={version} power={phase === "ready" ? "awake" : phase === "loading" ? "loading" : "asleep"} progress={progress ?? 0}
          working={busy || left > 0} inset={inset} api={brain} onPick={select} onTouch={stopAuto} />
        <header class="brand">
          <h1>L<span class="o">O</span>BOTOMY</h1>
          <p><b>{TOTAL}</b> neurons. <b>{catalog.length}</b> have names. Find the rest.</p>
        </header>
        <div class="tools">
          <button onClick={surprise} title="Jump to a neuron worth hearing: one with a name, or a strong signal you have not listened to yet">🎲 Random<span class="wide-only"> neuron</span></button>
          <button class={drawer ? "on" : ""} onClick={() => setDrawer(drawer ? null : "atlas")} title="Every named neuron, and what the scanner found on the open layer">📖 Atlas <b>{catalog.length}</b></button>
        </div>
        <div class="layerbar" ref={barRef}>
          <div class="layerbar-head">
            <button class="icon" onClick={() => step(-1)} disabled={layer === 0} title="One layer towards the front (←)" aria-label="Previous layer">◀</button>
            <b class="layerbar-name">{layer === null ? "Whole brain" : `Layer ${layer}`}</b>
            <button class="icon" onClick={() => step(1)} disabled={layer === NUM_LAYERS - 1} title="One layer towards the back (→)" aria-label="Next layer">▶</button>
            <span class="layerbar-info">
              {stats === null
                ? <><span class="wide-only">24 layers × 3,584 neurons · </span><b>{((all.scanned / (NUM_LAYERS * INTERMEDIATE)) * 100).toFixed(1)}%</b> charted</>
                : stats.scanned === INTERMEDIATE ? <><span class="wide-only">all 3,584 read · </span><b>{stats.signals}</b> signals{layers[layer!].named > 0 && <span class="wide-only"> · {layers[layer!].named} named</span>}</>
                : stats.scanned ? <>{stats.scanned.toLocaleString("en-US")}<span class="wide-only"> of 3,584</span> read{stats.signals > 0 && <span class="wide-only"> · <b>{stats.signals}</b> signals</span>}</>
                : <span class="wide-only">uncharted: none of its 3,584 neurons has been read yet</span>}
            </span>
            {layer !== null && left > 0 && <span class="scanning">scanning · {Math.ceil(left / 2).toLocaleString("en-US")} to go{eta > 5 ? ` · ${eta > 90 ? Math.round(eta / 60) + " min" : Math.round(eta) + " s"}` : ""} <button onClick={stopScan}>stop</button></span>}
            {layer !== null && left === 0 && stats!.scanned < INTERMEDIATE && <button onClick={scanLayer} title="Push every neuron of this layer up and down once and read off what the patient calls itself. Takes about 12 minutes, in the background.">scan<span class="wide-only"> this layer</span></button>}
            {layer !== null && <button class="whole-btn" onClick={() => goLayer(null)} title="Put the brain back together (Esc)">whole brain</button>}
          </div>
          <LayerRail layer={layer} layers={layers} probes={probes} onLayer={goLayer} />
          <div class="layerbar-ends"><span>front · layer 0</span><span>{layer === null ? "pick a layer to cut the brain open · drag to turn it · tap a neuron to push it" : "drag to turn · pinch or scroll to zoom · tap a neuron to push it"}</span><span>layer 23 · back</span></div>
        </div>
        {drawer && (
          <aside class="drawer">
            <div class="drawer-head">
              <div class="tabs">
                <button class={drawer === "atlas" ? "on" : ""} onClick={() => setDrawer("atlas")}>Named <span>{catalog.length}</span></button>
                <button class={drawer === "signals" ? "on" : ""} onClick={() => setDrawer("signals")}>{layer === null ? "Signals" : `Layer ${layer}`} <span>{layer === null ? "" : signalRows}</span></button>
              </div>
              <button class="icon" onClick={() => setDrawer(null)} title="Close (Esc)" aria-label="Close">×</button>
            </div>
            <div class="drawer-body">
              {drawer === "atlas"
                ? <Atlas catalog={catalog} probes={probes} onPick={pick} onForget={forget} />
                : layer === null ? <div class="dim pad">Cut the brain open at a layer to see what the scanner found there.</div>
                : <SignalList L={layer} version={version} heard={heard} catalog={catalog} sel={sel} ready={phase === "ready"} onPick={(n, sign) => { if (!wide) setDrawer(null); select(layer, n, sign); }} />}
            </div>
            <p class="drawer-foot">You have heard <b>{Object.keys(heard).length}</b> neurons and named <b>{mine.length}</b>.</p>
          </aside>
        )}
      </section>

      <div class="deck" ref={cardRef}>
        <NeuronCard sel={sel} probes={probes} catalog={catalog} version={version} adding={adding}
          onFace={face} onDose={setDose} onCommit={() => request("case")}
          onRemove={(i) => change((ps) => ps.filter((_, j) => j !== i))} onReset={() => { setAdding(false); change(() => []); }}
          onAdding={setAdding} onFocus={(L, n) => { setLayer(L); setSel({ L, n }); }} />
      </div>

      <Chat phase={phase} status={status} progress={progress} joke={PREOP[preop % PREOP.length]} gpu={gpuName} hasGpu={HAS_GPU} fainted={fainted}
        items={items} catalog={catalog} probes={probes} busy={busy} draft={draft} question={question} presets={PRESETS} featured={featured.slice(0, 5)}
        recorded={recorded} autoplaying={autoplaying} act={act.current} settings={settings} raised={raised} onRaise={setRaised}
        onDraft={setDraft} onAsk={ask} onStop={() => { pending.current = null; send({ type: "stop" }); }} onLoad={() => load()} onTry={(k) => pick(k, false)} />

      {share && (
        <div class="modal" onClick={(e) => { if (e.target === e.currentTarget) setShare(null); }}>
          <div class="sheet" role="dialog" aria-label="Share">
            <img src={share.src} alt="The picture to share" />
            <div class="sheet-actions">
              {canNativeShare && <button class="primary" onClick={nativeShare}>Share…</button>}
              <button class={canNativeShare ? "" : "primary"} onClick={copyImage}>Copy image</button>
              <a class="button" href={`https://x.com/intent/post?text=${encodeURIComponent(share.text)}&url=${encodeURIComponent(share.link)}`} target="_blank" rel="noopener"
                title="Opens a draft on X with the text and the link">Post on X</a>
              <button onClick={() => download(share.src, share.file)}>Download</button>
              <button onClick={() => copyLink(share.link)}>Copy link</button>
              <button class="link" onClick={() => setShare(null)}>close</button>
            </div>
            <p class="sheet-tip">Copy the image, then paste it into your post. The link replays this exact result for anyone who opens it.</p>
          </div>
        </div>
      )}
      {toast && <div class="toast" role="status">{toast}</div>}
    </div>
  );
}
