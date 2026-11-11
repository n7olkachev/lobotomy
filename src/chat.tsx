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
function Monitor({ phase, busy, vitals, changed, gpu, raised, onRaise, children }: {
  phase: Phase; busy: boolean; vitals: Vitals | null; changed: number; gpu: string; raised: boolean; onRaise: (on: boolean) => void; children?: ComponentChildren;
}) {
  const kind = phase !== "ready" ? (phase === "loading" ? "asleep" : "off") : busy ? "live" : vitals ? vitals.state : "stable";
  const state = phase === "idle" ? "asleep" : phase === "loading" ? "waking up" : phase === "error" ? "no pulse" : busy ? "answering" : vitals && vitals.state !== "intact" ? vitals.label : "awake";
  // On a phone the strip is the handle between the brain and the conversation: a tap turns them around, and so does a
  // swipe, for whoever takes the handle for something to drag.
  const from = useRef<number | null>(null), swiped = useRef(false);
  return (
    <header class={`monitor v-${kind}`}>
      <button class="grip" aria-expanded={raised} aria-label={raised ? "Make the conversation smaller" : "Make the conversation bigger"}
        onPointerDown={(e) => { from.current = e.clientY; swiped.current = false; }}
        onPointerUp={(e) => { const dy = from.current === null ? 0 : e.clientY - from.current; from.current = null; if (Math.abs(dy) > 14) { swiped.current = true; onRaise(dy < 0); } }}
        onPointerCancel={() => { from.current = null; }}
        onClick={() => { if (!swiped.current) onRaise(!raised); swiped.current = false; }} />
      <div class="monitor-id">
        <b>The patient</b>
        <span title={gpu ? `Running locally on ${gpu}. Nothing leaves this device.` : "A language model with 86,016 MLP neurons in 24 layers"}>Qwen3.5 · 0.8B{gpu ? " · runs on your GPU, no server" : " · a real language model"}</span>
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

/** "2 of 86,016 neurons changed"; on a phone, where the monitor right above says out of how many, "2 neurons changed". */
const Changed = ({ n }: { n: number }) => <><em>{n}</em><span class="wide-only"> of {TOTAL} neurons</span><span class="narrow-only"> {n === 1 ? "neuron" : "neurons"}</span> changed</>;

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
      <div class={`msg user ${again ? "again" : ""}`}><p title={again ? "Asked again after the brain was changed" : ""}>{again && <i>↻ </i>}{t.prompt}</p></div>
      <div class={`msg patient ${steered ? "steered" : "intact"} ${v ? `v-${v.state}` : ""}`}>
        <span class="avatar" aria-hidden="true">{title.emoji}</span>
        <div class="bubble">
          <div class="who">
            <b>{title.name}</b>
            <span>{steered ? <Changed n={t.probes.length} /> : "nothing changed"}</span>
          </div>
          <div class={`quote ${sizeClass(main)}`}>
            {main ? <Answer text={main} live={live} /> : <span class="dim">{live ? <span class="caret" /> : t.stopped ? "stopped" : "(no answer)"}</span>}
          </div>
          {steered && !hideBefore && (t.before || !t.done) && (
            <div class={`untouched ${open ? "open" : ""}`} onClick={() => setOpen(!open)} title={open ? "" : "Show the whole answer"}>
              <span class="lbl">Untouched model</span>
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
              <span>This {t.probes.length === 1 ? "neuron" : "combination"} has no name yet. What would you call it?</span>
              <input value={name} onInput={(e) => setName((e.target as HTMLInputElement).value)} placeholder="🚂 Train conductor" maxLength={40} aria-label="Name" />
              <button disabled={!name.trim()}>name it</button>
            </form>
          )}
          {settled && steered && !t.stopped && (
            <div class="actions">
              <button class="primary" onClick={() => act.share(t)} title="A picture of this result to post, and a link that reproduces it">Share</button>
              {last && <button onClick={act.ladder} disabled={!canLadder} title="Ask the same question at six increasing strengths">Turn it up<span class="wide-only">, step by step</span></button>}
              <button onClick={() => act.link(t)} title="Copy a link that reproduces this result">Copy link</button>
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
          <div class="who"><b>{k.name}</b><span><span class="wide-only"><em>{k.probes.length}</em> of {TOTAL} neurons changed · </span><i class="rec">recording</i></span></div>
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
      <h2>One of these neurons is <em>coffee.</em></h2>
      <p>A real language model (Alibaba's Qwen3.5, the small 0.8B version) runs right in this tab, on your own GPU. Every dot of the brain is one of its <b>{TOTAL} neurons</b>.</p>
      <ol>
        <li><b>Tap a neuron</b> to push it up or down while the model writes.</li>
        <li><b>It answers the same question again</b>, now as coffee, a lawyer, a Victorian schoolgirl…</li>
        <li><b>Name the ones you find.</b> Only {named} have names so far.</li>
      </ol>
      {featured.length > 0 && (
        <div class="try">
          <span>Start with:</span>
          {featured.map((k) => <button key={k.name} class="chip" onClick={() => onTry(k)} title={k.desc}>{emojiOf(k)} {k.name}</button>)}
        </div>
      )}
    </article>
  );
}

// ---- where the conversation is scrolled to ---------------------------------------------------------------------

const ROOM = 280;   // px of an answer that have to fit under its question for the question to stay in the window
const NEAR = 12;    // this close to the end counts as the end
const SETTLE = 150; // ms without a scroll or wheel event after which the reader has stopped scrolling

/**
 * Brings every new exchange into view, and leaves a reader who scrolls alone.
 *
 * An exchange that fits into the window is shown whole, with the conversation ending at the bottom edge. A taller one
 * stays at its beginning while the rest is typed below the edge: at the question or, when that would leave no room to
 * read the answer (the few lines of window a phone has), at the answer. Once the reader scrolls, the page keeps its
 * hands off: it follows the text again only if they went to the end, and otherwise waits for the next exchange.
 *
 * `id` is the number of the turn or ladder that is last (the numbers only go up), null for a recording.
 */
function useFollow(key: string, id: number | null) {
  const scroller = useRef<HTMLDivElement>(null), inner = useRef<HTMLDivElement>(null);
  const st = useRef({
    mode: "pin" as "pin" | "end" | "free",
    /** a new exchange that has not been scrolled to yet */
    due: false,
    /** the reader's scrolling: a finger is down, or it was lifted (or the wheel turned) and the text may still be moving */
    user: "" as "" | "touch" | "coast",
    /** the scroll position as the page last set it or saw it, and how tall the window was then */
    seen: 0, height: 0,
    newest: -1, timer: 0,
  }).current;

  /** Where the last exchange belongs while nobody has scrolled. */
  const pin = (el: HTMLElement): number => {
    const max = el.scrollHeight - el.clientHeight;
    const marks = el.querySelectorAll<HTMLElement>(".msg.user, .op, .msg.ladder");
    let from: HTMLElement | undefined = marks[marks.length - 1];
    if (!from) return max;
    if (from.previousElementSibling?.classList.contains("op")) from = from.previousElementSibling as HTMLElement; // with the line that says what was done
    const answers = el.querySelectorAll<HTMLElement>(".msg.patient");
    const answer = answers[answers.length - 1];
    let top = from.offsetTop;
    if (answer && answer.offsetTop > top && el.clientHeight - 10 - (answer.offsetTop - top) < ROOM) top = answer.offsetTop;
    return Math.max(0, Math.min(max, top - 10));
  };
  /** Sees that the text is not where the page left it: the reader scrolled (or the browser did, to stay in range). */
  const sync = (el: HTMLElement): boolean => {
    const top = el.scrollTop, resized = el.clientHeight !== st.height;
    st.height = el.clientHeight;
    if (Math.abs(top - st.seen) < 1) return false;
    st.seen = top;
    // Moved while the window was changing size, with no finger on it: that was the browser keeping the text in range
    // (a keyboard going away), and says nothing about where the reader wants to be.
    if (!st.due && (st.user || !resized)) {
      const max = el.scrollHeight - el.clientHeight;
      st.mode = max - top > NEAR ? "free" : pin(el) < max - NEAR ? "end" : "pin";
    }
    return true;
  };
  const follow = () => {
    const el = scroller.current;
    if (!el) return;
    sync(el);
    // A box in the conversation that is being typed in (the name of a neuron) is kept in view by the browser.
    const typing = document.activeElement instanceof HTMLInputElement && el.contains(document.activeElement);
    // Under a finger the text only moves for a new exchange.
    if (st.mode === "free" || ((st.user || typing) && !st.due)) return;
    const to = st.mode === "end" ? el.scrollHeight - el.clientHeight : pin(el);
    if (Math.abs(el.scrollTop - to) >= 1) el.scrollTop = to;
    st.seen = el.scrollTop;
    st.due = false;
  };

  useLayoutEffect(() => {
    if (id !== null && id <= st.newest) { st.mode = "free"; st.due = false; } // the last one was taken away: there is nothing new to show
    else { st.mode = "pin"; st.due = true; if (id !== null) st.newest = id; }
    follow();
  }, [key]);

  useEffect(() => {
    const el = scroller.current!;
    const settle = () => { clearTimeout(st.timer); st.timer = window.setTimeout(() => { st.user = ""; follow(); }, SETTLE); };
    const lifted = (e: Event) => { if (!(e as TouchEvent).touches.length && st.user === "touch") { st.user = "coast"; settle(); } };
    const touched = (e: TouchEvent) => {
      st.user = "touch";
      clearTimeout(st.timer);
      // A finger is lifted from the element it came down on. An answer that is being typed may have redrawn that
      // element away by then, and the event no longer passes through the page: listen on the element as well.
      e.target?.addEventListener("touchend", lifted, { once: true, passive: true });
      e.target?.addEventListener("touchcancel", lifted, { once: true, passive: true });
    };
    const wheeled = () => { if (st.user !== "touch") { st.user = "coast"; settle(); } };
    // what a flick set in motion keeps moving after the finger is gone
    const scrolled = () => { if (sync(el) && st.user === "coast") settle(); };
    // Passive listeners: the browser does not wait for the page before it scrolls.
    el.addEventListener("touchstart", touched, { passive: true });
    el.addEventListener("wheel", wheeled, { passive: true });
    el.addEventListener("scroll", scrolled, { passive: true });
    document.addEventListener("touchend", lifted, { capture: true, passive: true });
    document.addEventListener("touchcancel", lifted, { capture: true, passive: true });
    // The text grows without this component being rendered again (a recording types itself), and the window changes
    // size under it (a phone's keyboard, the card of the neuron above).
    const ro = new ResizeObserver(follow);
    ro.observe(el);
    ro.observe(inner.current!);
    return () => {
      el.removeEventListener("touchstart", touched); el.removeEventListener("wheel", wheeled); el.removeEventListener("scroll", scrolled);
      document.removeEventListener("touchend", lifted, { capture: true }); document.removeEventListener("touchcancel", lifted, { capture: true });
      ro.disconnect();
      clearTimeout(st.timer);
    };
  }, []);
  return { scroller, inner };
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
  /** a phone: the conversation is pulled up over the brain */
  raised: boolean; onRaise: (on: boolean) => void;
  onDraft: (s: string) => void; onAsk: (text: string) => void; onStop: () => void; onLoad: () => void; onTry: (k: Knob) => void;
}) {
  const turns = p.items.filter((it): it is { kind: "turn"; turn: Turn } => it.kind === "turn");
  const lastTurn = turns[turns.length - 1]?.turn ?? null;
  const ready = p.phase === "ready";
  const vitals = lastTurn && lastTurn.done && lastTurn.answered && !lastTurn.stopped ? vitalsOf(lastTurn) : null;
  const unheard = !ready && p.probes.length > 0 && !p.recorded;
  // what is last in the conversation: a recording (or the line that says there is none), else the last turn or ladder
  const tail = p.items[p.items.length - 1];
  const tailId = tail ? (tail.kind === "turn" ? tail.turn.id : tail.ladder.id) : 0;
  const playing = !ready && p.recorded ? p.recorded.name : unheard ? "" : null;
  const { scroller, inner } = useFollow(playing !== null ? `recording ${playing}` : `item ${tailId}`, playing !== null ? null : tailId);
  /** On a phone: put the keyboard away, the answer is under it. */
  const fold = () => { if (matchMedia("(pointer: coarse)").matches) (document.activeElement as HTMLElement | null)?.blur?.(); };
  const submit = (e: Event) => {
    e.preventDefault();
    p.onAsk((e.currentTarget as HTMLFormElement).querySelector("input")!.value); // what is in the box now, typed or pasted or dictated
    fold();
  };

  let prev: Turn | null = null;
  return (
    <section class="chat">
      <Monitor phase={p.phase} busy={p.busy} vitals={vitals} changed={p.probes.length} gpu={ready ? p.gpu : ""} raised={p.raised} onRaise={p.onRaise}>{p.settings}</Monitor>
      <div class="timeline" ref={scroller}>
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
        {unheard && <div class="op quiet"><span>There is no recording of this one. {p.hasGpu ? "Wake the patient to hear it live." : "Hearing it live takes a browser with WebGPU."}</span></div>}
       </div>
      </div>
      <footer class="composer">
        {ready && (
          <>
            {/* Safari does not give a tapped button the focus: the box would lose it, the row (shown while the box has it) would go, and the tap with it. */}
            <div class="presets" aria-label="Questions to ask" onMouseDown={(e) => e.preventDefault()}>
              {p.presets.map((q) => <button key={q} class={`chip ${q === p.question ? "on" : ""}`} onClick={() => { p.onAsk(q); fold(); }}>{q}</button>)}
            </div>
            <form class="ask" onSubmit={submit}>
              <input value={p.draft} onInput={(e) => p.onDraft((e.target as HTMLInputElement).value)} placeholder="Ask the patient anything…" aria-label="Question for the patient" enterkeyhint="send" autocomplete="off" />
              {p.busy ? <button type="button" onClick={p.onStop}>stop</button> : <button class="primary" disabled={!p.draft.trim()}>ask</button>}
            </form>
          </>
        )}
        {p.phase === "idle" && p.hasGpu && p.fainted && <p class="fainted">The model did not finish loading last time. This device may be short of memory for it (470 MB): closing other tabs can help. The recordings play either way.</p>}
        {p.phase === "idle" && p.hasGpu && (
          <button class="wake primary" onClick={p.onLoad}>
            <b>{p.fainted ? "Try waking the patient again" : "Wake the patient"}</b>
            <span>{PHONE ? "470 MB, once · use Wi-Fi · runs on your phone" : "a 470 MB download, once · runs on your GPU, no server"}</span>
          </button>
        )}
        {p.phase === "idle" && !p.hasGpu && (
          <div class="nogpu">
            <b>The live model needs WebGPU, and this browser does not have it.</b>
            <span>The recordings still play: tap any emoji on the brain. For the live model, use Chrome or Edge on a computer, a recent Android phone, or Safari 26.</span>
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
            <b>The model failed to load.</b>
            <span>{p.status}</span>
            <button onClick={p.onLoad}>try again</button>
          </div>
        )}
      </footer>
    </section>
  );
}
