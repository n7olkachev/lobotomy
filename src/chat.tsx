// The right-hand side: the patient. A monitor, the conversation, and the box to ask questions in.
import { Component, type ComponentChildren, type FunctionComponent } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { Knob } from "./catalog";
import { Answer } from "./md";
import { TOTAL, signed } from "./parts";
import { emojiOf, exactPersona, titleFor } from "./personas";
import { formatRecipe, type Probe } from "./steer";
import { findLoop, punchline } from "./text";
import { diagnose, type Vitals } from "./vitals";

export type Phase = "idle" | "loading" | "ready" | "error";
const PHONE = typeof matchMedia !== "undefined" && matchMedia("(pointer: coarse)").matches;

/** One question and its answer, with the brain as it was at that moment. */
export interface Turn {
  id: number;
  prompt: string;
  /** asked again by the page after the brain was changed, not typed by the visitor */
  auto: boolean;
  /** what was done to the brain since the answer before this one */
  op: string | null;
  probes: Probe[];
  /** the untouched model's answer */
  before: string;
  beforeDone: boolean;
  /** `before` was carried over from an earlier turn and has not been confirmed by the worker yet */
  kept?: boolean;
  /** the answer with the electrodes in ("" when there are none) */
  after: string;
  /** the answer that was asked for is complete (the untouched one, for comparison, may still be coming) */
  answered: boolean;
  nll: number | null;
  tps: number | null;
  done: boolean;
  stopped?: boolean;
}
export interface Rung { probes: Probe[]; text: string; nll: number | null; done: boolean }
export interface LadderT { id: number; prompt: string; base: Probe[]; rungs: Rung[]; done: boolean }
export type Item = { kind: "turn"; turn: Turn } | { kind: "ladder"; ladder: LadderT };
export const vitalsOf = (c: Turn): Vitals => diagnose({ after: c.after, before: c.before, nll: c.nll, steered: c.probes.length > 0 });

/** What the buttons under an answer do. The object is the same on every render, so answers are not redrawn for it. */
export interface Actions {
  share: (t: Turn) => void;
  link: (t: Turn) => void;
  name: (t: Turn, name: string) => void;
  ladder: () => void;
  shareLadder: (l: LadderT) => void;
  closeLadder: (l: LadderT) => void;
}

/** Re-renders only when a prop is a different object: a long conversation is not laid out again for every token. */
function memo<P extends object>(C: FunctionComponent<P>): FunctionComponent<P> {
  return class extends Component<P> {
    shouldComponentUpdate(next: P) {
      for (const k in next) if (next[k] !== (this.props as P)[k]) return true;
      return false;
    }
    render() { return <C {...(this.props as P)} />; }
  } as unknown as FunctionComponent<P>;
}

// ---- monitor ---------------------------------------------------------------------------------------------------

const BASE = 22;
function ekgPath(kind: string): string {
  let seed = 1 + [...kind].reduce((a, c) => a + c.charCodeAt(0), 0);
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const pts: [number, number][] = [[0, BASE]];
  const beat = (x: number, amp: number, w = 1) => pts.push(
    [x, BASE], [x + 6 * w, BASE - amp * 0.16], [x + 11 * w, BASE], [x + 15 * w, BASE + amp * 0.2], [x + 19 * w, BASE - amp],
    [x + 23 * w, BASE + amp * 0.5], [x + 27 * w, BASE], [x + 37 * w, BASE - amp * 0.24], [x + 46 * w, BASE],
  );
  if (kind === "flatline") for (let x = 20; x < 600; x += 24) pts.push([x, BASE + (rnd() - 0.5) * 1.2]);
  else if (kind === "off") { /* a flat line */ }
  else if (kind === "asleep") for (let x = 30; x < 540; x += 190) beat(x, 7, 1.5);
  else if (kind === "looping") for (let x = 6; x < 590; x += 40) beat(x, 11, 0.7);
  else if (kind === "slurring") for (let x = 8; x < 540; x += 40 + rnd() * 95) beat(x, 5 + rnd() * 15, 0.7 + rnd() * 0.8);
  else if (kind === "altered") for (let x = 14; x < 540; x += 100) beat(x, 10 + rnd() * 9, 0.9 + rnd() * 0.3);
  else if (kind === "live") for (let x = 10; x < 560; x += 75) beat(x, 15);
  else for (let x = 20; x < 540; x += 100) beat(x, 16);
  pts.push([600, BASE]);
  return "M" + pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join("L");
}
const EKG_CACHE = new Map<string, string>();
function Ekg({ kind }: { kind: string }) {
  let d = EKG_CACHE.get(kind);
  if (!d) EKG_CACHE.set(kind, (d = ekgPath(kind)));
  return (
    <svg class="ekg" viewBox="0 0 600 44" preserveAspectRatio="none" aria-hidden="true">
      <g class="ekg-scroll"><path d={d} /><path d={d} transform="translate(600,0)" /></g>
    </svg>
  );
}

/** The strip on top of the conversation: who the patient is, how it is doing, how much of it was changed. */
function Monitor({ phase, busy, vitals, changed, gpu, children }: { phase: Phase; busy: boolean; vitals: Vitals | null; changed: number; gpu: string; children?: ComponentChildren }) {
  const kind = phase !== "ready" ? (phase === "loading" ? "asleep" : "off") : busy ? "live" : vitals ? vitals.state : "stable";
  const state = phase === "idle" ? "asleep" : phase === "loading" ? "waking up" : phase === "error" ? "no pulse" : busy ? "answering" : vitals && vitals.state !== "intact" ? vitals.label : "awake";
  return (
    <header class={`monitor v-${kind}`}>
      <div class="monitor-id">
        <b>The patient</b>
        <span title={gpu ? `Running on ${gpu}. Nothing is sent to a server.` : "A language model with 86,016 MLP neurons in 24 layers"}>Qwen3.5 · 0.8B{gpu ? " · runs on your GPU, no server" : " · a real language model"}</span>
      </div>
      <div class={`monitor-count ${changed ? "changed" : ""}`} title={`${changed} of the model's ${TOTAL} neurons are being pushed`}>
        <b>{changed}</b><span>of {TOTAL}<br />neurons changed</span>
      </div>
      {children}
      <div class="monitor-line"><Ekg kind={kind} /><b class="monitor-state">{state}</b></div>
    </header>
  );
}

// ---- the conversation ------------------------------------------------------------------------------------------

function sizeClass(text: string): string {
  const loop = findLoop(text);
  const n = loop ? loop.start + loop.unit.length * 3 : text.length;
  return n <= 90 ? "q-xl" : n <= 230 ? "q-lg" : "q-md";
}

const TurnView = memo(function TurnView({ t, again, last, hideBefore, catalog, canLadder, act }: {
  t: Turn; again: boolean; last: boolean; hideBefore: boolean; catalog: Knob[]; canLadder: boolean; act: Actions;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const steered = t.probes.length > 0;
  const title = titleFor(t.probes, catalog);
  const main = steered ? t.after : t.before;
  const live = !t.answered && !t.done;
  const settled = t.done && t.answered;
  const v = settled && steered ? vitalsOf(t) : null;
  const nameable = last && settled && !t.stopped && steered && !exactPersona(t.probes, catalog);
  return (
    <>
      {t.op && <div class="op"><span>⚡ {t.op}</span></div>}
      <div class={`msg user ${again ? "again" : ""}`}><p title={again ? "Asked again, because the brain was changed" : ""}>{again && <i>↻ </i>}{t.prompt}</p></div>
      <div class={`msg patient ${steered ? "steered" : "intact"} ${v ? `v-${v.state}` : ""}`}>
        <span class="avatar" aria-hidden="true">{title.emoji}</span>
        <div class="bubble">
          <div class="who">
            <b>{title.name}</b>
            <span>{steered ? <><em>{t.probes.length}</em> of {TOTAL} neurons changed</> : "nothing changed"}</span>
          </div>
          <div class={`quote ${sizeClass(main)}`}>
            {main ? <Answer text={main} live={live} /> : <span class="dim">{live ? <span class="caret" /> : t.stopped ? "stopped" : "(says nothing at all)"}</span>}
          </div>
          {steered && !hideBefore && (t.before || !t.done) && (
            <div class={`untouched ${open ? "open" : ""}`} onClick={() => setOpen(!open)} title={open ? "" : "Click to read the whole answer"}>
              <span class="lbl">untouched, it says</span>
              <div class="untouched-text">{t.before ? <Answer text={t.before} live={!t.beforeDone && !t.done} /> : <span class="dim">…</span>}</div>
            </div>
          )}
          {settled && (v || t.tps != null) && (
            <div class="vitals">
              {v && <b class="state" title="How the answer reads to the untouched model">{v.label}</b>}
              {v?.score != null && <span title="How plausible the answer still is to the untouched model: 100 reads like a normal answer, 0 is word salad">coherence {v.score}</span>}
              {v?.symptoms.map((s) => <span class="symptom" key={s.name} title={s.hint}>{s.name}</span>)}
              {t.tps != null && <span class="dim right" title="Generated in this tab, on your GPU">{t.tps.toFixed(0)} tok/s</span>}
            </div>
          )}
          {nameable && (
            <form class="naming" onSubmit={(e) => { e.preventDefault(); if (name.trim()) { act.name(t, name.trim()); setName(""); } }}>
              <span>Nobody has named this one. What is it?</span>
              <input value={name} onInput={(e) => setName((e.target as HTMLInputElement).value)} placeholder="e.g. 🚂 Train conductor" maxLength={40} aria-label="A name for this neuron" />
              <button disabled={!name.trim()}>name it</button>
            </form>
          )}
          {settled && steered && !t.stopped && (
            <div class="actions">
              <button class="primary" onClick={() => act.share(t)} title="A picture of this operation to post, and a link that repeats it">Share</button>
              {last && <button onClick={act.ladder} disabled={!canLadder} title="Ask the same question at six rising strengths and watch the patient go">Turn it up<span class="wide-only">, step by step</span></button>}
              <button onClick={() => act.link(t)} title="Copy a link that repeats this operation">Copy link</button>
              <span class="rx" title="The recipe: layer, neuron and strength of every electrode">℞ {formatRecipe(t.probes)}</span>
            </div>
          )}
        </div>
      </div>
    </>
  );
});

const LadderView = memo(function LadderView({ l, catalog, act }: { l: LadderT; catalog: Knob[]; act: Actions }) {
  const title = titleFor(l.base, catalog);
  return (
    <div class="msg ladder">
      <div class="ladder-head">
        <b>{title.emoji} {title.name}, step by step</b>
        <span>“{l.prompt}” at six strengths</span>
        <button onClick={() => act.shareLadder(l)} disabled={!l.done}>Share</button>
        <button class="icon" onClick={() => act.closeLadder(l)} title="Remove from the conversation" aria-label="Remove">×</button>
      </div>
      {l.rungs.map((r, i) => {
        const v = r.done ? diagnose({ after: r.text, before: l.rungs[0].text, nll: r.nll, steered: r.probes.length > 0 }) : null;
        const p = r.text ? punchline(r.text, 170) : null;
        return (
          <div class={`rung ${v ? `v-${v.state}` : ""}`} key={i} style={{ "--level": l.rungs.length > 1 ? i / (l.rungs.length - 1) : 0 }}>
            <span class="rung-dose">{r.probes.length ? r.probes.map((q) => signed(q.dose)).join(" ") : "untouched"}</span>
            <span class="rung-text">{p ? <>{p.text}{p.loop && <span class="loop-badge" title="Stuck in a loop">↻</span>}</> : <span class="dim">{i === 0 || l.rungs[i - 1].done ? <span class="caret" /> : "…"}</span>}</span>
            <span class="rung-state">{v && v.state !== "intact" ? v.label : ""}</span>
          </div>
        );
      })}
    </div>
  );
});

/** Reveals `text` over about a second, like the live model typing it. */
function useTyped(text: string): string {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { setN(text.length); return; }
    setN(0);
    const step = Math.max(1, Math.round(text.length / 55));
    const t = setInterval(() => setN((v) => { if (v + step >= text.length) clearInterval(t); return Math.min(text.length, v + step); }), 18);
    return () => clearInterval(t);
  }, [text]);
  return text.slice(0, n);
}

/** Before the model is loaded: what the patient said when this neuron was catalogued. */
function Recording({ k, auto }: { k: Knob; auto: boolean }) {
  const full = k.sample ?? k.desc;
  const typed = useTyped(full);
  return (
    <>
      <div class="op"><span>⚡ {k.probes.length === 1 ? `layer ${k.probes[0].L} · neuron ${k.probes[0].n} pushed ${k.probes[0].dose >= 0 ? "up" : "down"}` : `${k.probes.length} neurons pushed · ${formatRecipe(k.probes)}`}</span></div>
      <div class={`msg patient steered recording ${auto ? "auto" : ""}`}>
        <span class="avatar" aria-hidden="true">{emojiOf(k)}</span>
        <div class="bubble">
          <div class="who"><b>{k.name}</b><span><em>{k.probes.length}</em> of {TOTAL} neurons changed · <i class="rec">recording</i></span></div>
          <div class={`quote ${sizeClass(full)}`}><Answer text={typed} live={typed.length < full.length} /></div>
          <p class="note">{k.desc}</p>
          {auto && <div class="autoplay" title="Playing recordings. Tap a neuron to take over."><i /></div>}
        </div>
      </div>
    </>
  );
}

function Intro({ named, featured, onTry }: { named: number; featured: Knob[]; onTry: (k: Knob) => void }) {
  return (
    <article class="intro">
      <h2>The brain of a real AI. <em>Poke it.</em></h2>
      <p>A language model (Qwen3.5, 0.8 billion parameters) runs in this tab, on your GPU. That is its brain: <b>{TOTAL} neurons</b>, 24 layers of 3,584.</p>
      <ol>
        <li><b>Tap a neuron.</b> It is pushed up or down while the model thinks.</li>
        <li><b>The patient answers again</b>, as coffee, a lawyer, a Victorian schoolgirl…</li>
        <li><b>Find a new one and name it.</b> Only {named} have names so far.</li>
      </ol>
      {featured.length > 0 && (
        <div class="try">
          <span>Try a known one:</span>
          {featured.map((k) => <button key={k.name} class="chip" onClick={() => onTry(k)} title={k.desc}>{emojiOf(k)} {k.name}</button>)}
        </div>
      )}
    </article>
  );
}

export function Chat(p: {
  phase: Phase; status: string; progress?: number; joke: string; gpu: string; hasGpu: boolean;
  /** the last time the model was loaded, the tab died (a phone out of memory) */
  fainted: boolean;
  items: Item[]; catalog: Knob[]; probes: Probe[]; busy: boolean;
  draft: string; question: string; presets: string[]; featured: Knob[];
  /** the persona whose recording is shown while there is no live model */
  recorded: Knob | null; autoplaying: boolean;
  act: Actions; settings?: ComponentChildren;
  onDraft: (s: string) => void; onAsk: (text: string) => void; onStop: () => void; onLoad: () => void; onTry: (k: Knob) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  // Follow the answer as it grows, unless the reader has scrolled away. An exchange that is taller than the window
  // stays at its beginning (the question and the first lines), not at its buttons.
  const pinned = useRef(0), inner = useRef<HTMLDivElement>(null);
  const follow = () => {
    const el = scroller.current;
    if (!el || !stick.current) return;
    const marks = el.querySelectorAll<HTMLElement>(".msg.user, .op");
    let from: HTMLElement | undefined = marks[marks.length - 1];
    if (from?.previousElementSibling?.classList.contains("op")) from = from.previousElementSibling as HTMLElement; // with the line that says what was done
    pinned.current = Math.max(0, Math.min(el.scrollHeight - el.clientHeight, from ? from.offsetTop - 10 : Infinity));
    el.scrollTop = pinned.current;
  };
  useLayoutEffect(follow);
  // text that types itself (a recording) grows without this component being rendered again
  useEffect(() => {
    const ro = new ResizeObserver(follow);
    ro.observe(inner.current!);
    return () => ro.disconnect();
  }, []);
  const onScroll = () => { const el = scroller.current!; stick.current = Math.abs(el.scrollTop - pinned.current) < 60 || el.scrollHeight - el.scrollTop - el.clientHeight < 60; };

  const turns = p.items.filter((it): it is { kind: "turn"; turn: Turn } => it.kind === "turn");
  const lastTurn = turns[turns.length - 1]?.turn ?? null;
  const ready = p.phase === "ready";
  const vitals = lastTurn && lastTurn.done && lastTurn.answered && !lastTurn.stopped ? vitalsOf(lastTurn) : null;
  const unheard = !ready && p.probes.length > 0 && !p.recorded;
  /** On a phone: put the keyboard away, the answer is under it. */
  const fold = () => { if (matchMedia("(pointer: coarse)").matches) (document.activeElement as HTMLElement | null)?.blur?.(); };
  const submit = (e: Event) => {
    e.preventDefault();
    stick.current = true;
    p.onAsk((e.currentTarget as HTMLFormElement).querySelector("input")!.value); // what is in the box now, typed or pasted or dictated
    fold();
  };

  let prev: Turn | null = null;
  return (
    <section class="chat">
      <Monitor phase={p.phase} busy={p.busy} vitals={vitals} changed={p.probes.length} gpu={ready ? p.gpu : ""}>{p.settings}</Monitor>
      <div class="timeline" ref={scroller} onScroll={onScroll}>
       <div class="timeline-in" ref={inner}>
        {turns.length === 0 && <Intro named={p.catalog.length} featured={p.featured} onTry={p.onTry} />}
        {p.items.map((it) => {
          if (it.kind === "ladder") return <LadderView key={"l" + it.ladder.id} l={it.ladder} catalog={p.catalog} act={p.act} />;
          const t = it.turn, before = prev;
          prev = t;
          return (
            <TurnView key={t.id} t={t} catalog={p.catalog} act={p.act} last={t === lastTurn} canLadder={!p.busy}
              again={t.auto && before?.prompt === t.prompt}
              hideBefore={!!before && before.prompt === t.prompt && before.probes.length === 0 && before.answered} />
          );
        })}
        {!ready && p.recorded && <Recording key={p.recorded.name} k={p.recorded} auto={p.autoplaying} />}
        {unheard && <div class="op quiet"><span>Nobody has recorded this one. {p.hasGpu ? "Wake the patient to hear it." : "Hearing it needs a browser with WebGPU."}</span></div>}
       </div>
      </div>
      <footer class="composer">
        {ready && (
          <>
            <div class="presets" aria-label="Questions to ask">
              {p.presets.map((q) => <button key={q} class={`chip ${q === p.question ? "on" : ""}`} onClick={() => { stick.current = true; p.onAsk(q); fold(); }}>{q}</button>)}
            </div>
            <form class="ask" onSubmit={submit}>
              <input value={p.draft} onInput={(e) => p.onDraft((e.target as HTMLInputElement).value)} placeholder="Ask the patient anything…" aria-label="Question for the patient" enterkeyhint="send" autocomplete="off" />
              {p.busy ? <button type="button" onClick={p.onStop}>stop</button> : <button class="primary" disabled={!p.draft.trim()}>ask</button>}
            </form>
          </>
        )}
        {p.phase === "idle" && p.hasGpu && p.fainted && <p class="fainted">Last time the patient did not wake up: this device may be short of memory for a 470 MB model. Closing other tabs helps. The recordings play without it.</p>}
        {p.phase === "idle" && p.hasGpu && (
          <button class="wake primary" onClick={p.onLoad}>
            <b>{p.fainted ? "Try to wake the patient again" : "Wake the patient"}</b>
            <span>{PHONE ? "470 MB, once (use Wi-Fi) · runs on your phone, no server" : "470 MB, once · runs on your GPU, no server"}</span>
          </button>
        )}
        {p.phase === "idle" && !p.hasGpu && (
          <div class="nogpu">
            <b>This browser has no WebGPU, so the patient stays asleep.</b>
            <span>The recordings play: tap any emoji on the brain. For the live model use Chrome or Edge on a computer, a recent Android phone, or Safari 26.</span>
          </div>
        )}
        {p.phase === "loading" && (
          <div class="waking">
            <div class="waking-bar"><i style={{ width: `${Math.round((p.progress ?? 0) * 100)}%` }} /></div>
            <div class="waking-text"><span>{p.status}</span><i>{p.joke}…</i></div>
          </div>
        )}
        {p.phase === "error" && (
          <div class="nogpu err">
            <b>The patient did not wake up.</b>
            <span>{p.status}</span>
            <button onClick={p.onLoad}>try again</button>
          </div>
        )}
      </footer>
    </section>
  );
}
