// The controls that belong to the brain: the neuron under the probe, the layer rail, the atlas of named neurons
// and the list of what the scanner found on a layer.
import type { ComponentChildren } from "preact";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { NUM_LAYERS, type Knob } from "./catalog";
import { UNNAMED, emojiOf, isShownActive, neuronCount, ownerOf, TOTAL_NEURONS } from "./personas";
import type { Probe } from "./steer";
import { STRONG, WEAK, faceOf, heardKey, reading, signals, type Heard } from "./store";
import { lead } from "./text";

export const TOTAL = TOTAL_NEURONS.toLocaleString("en-US");
export const signed = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)}`;
export interface Sel { L: number; n: number }
const same = (a: Sel, b: Sel) => a.L === b.L && a.n === b.n;

// ---- the neuron under the probe ------------------------------------------------------------------------------

const DOSE_MIN = 0.05, DOSE_MAX = 1.5;
const frac = (v: number) => (Math.min(DOSE_MAX, Math.max(DOSE_MIN, v)) - DOSE_MIN) / (DOSE_MAX - DOSE_MIN);

/**
 * The neuron that is selected, with its two faces (pushed up, pushed down: what the scanner read off each) and how
 * hard it is pushed; under it, the other electrodes that are in.
 */
export function NeuronCard(p: {
  sel: Sel | null; probes: Probe[]; catalog: Knob[]; version: number; adding: boolean;
  onFace: (L: number, n: number, sign: 1 | -1) => void;
  onDose: (i: number, dose: number) => void; onCommit: () => void;
  onRemove: (i: number) => void; onReset: () => void; onAdding: (on: boolean) => void; onFocus: (L: number, n: number) => void;
}) {
  const [open, setOpen] = useState(false); // on a phone the buttons and the slider are folded away until asked for
  const focus: Sel | null = p.sel ?? p.probes[p.probes.length - 1] ?? null;
  if (!focus && !p.probes.length) {
    return <div class="neuron-card empty"><span><b>Tap a neuron</b> to push it. The emoji mark the ones that have names.</span></div>;
  }
  const i = focus ? p.probes.findIndex((q) => same(q, focus)) : -1;
  const pr = i >= 0 ? p.probes[i] : null;
  const sign = pr ? (pr.dose >= 0 ? 1 : -1) : 0;
  const owner = pr ? ownerOf(pr, p.catalog, p.probes) : null;
  const ref = pr && owner?.probes.find((q) => same(q, pr) && q.dose * pr.dose > 0)?.dose;
  const mag = pr ? Math.min(DOSE_MAX, Math.max(DOSE_MIN, Math.abs(pr.dose))) : 1;
  const others = p.probes.map((q, j) => ({ q, j })).filter(({ j }) => j !== i);
  return (
    <div class={`neuron-card ${sign > 0 ? "pos" : sign < 0 ? "neg" : "off"}`}>
      {focus && (
        <div class={`neuron ${pr ? "" : "idle"} ${open || !pr ? "unfolded" : ""}`}>
          <div class="neuron-head" onClick={() => setOpen(!open)}>
            <span class="neuron-emoji">{pr && owner ? emojiOf(owner) : pr ? UNNAMED : "🔬"}</span>
            <b class="neuron-name">{pr ? owner?.name ?? "Unnamed neuron" : "Not pushed"}{pr && <span class="neuron-dose">{sign > 0 ? "▲" : "▼"} {signed(pr.dose)}</span>}</b>
            <span class="neuron-where">layer {focus.L} · neuron {focus.n}</span>
          </div>
          {pr && (
            <label class="strength">
              <span>strength</span>
              <span class="slider" style={{ "--fill": `${frac(mag) * 100}%` }}>
                <input type="range" min={DOSE_MIN} max={DOSE_MAX} step={0.05} value={mag} aria-label="How hard the neuron is pushed"
                  onInput={(e) => p.onDose(i, sign * +(e.target as HTMLInputElement).value)} onChange={p.onCommit} />
                {ref != null && <i class="slider-mark" style={{ left: `calc(8px + (100% - 16px) * ${frac(Math.abs(ref))})` }} title={`The strength it was catalogued at: ${signed(ref)}`} />}
              </span>
              <output title="In units of the layer's typical activity">{signed(pr.dose)}</output>
            </label>
          )}
          {pr && (
            <span class="neuron-out">
              <button class={`neuron-fold ${open ? "on" : ""}`} onClick={() => setOpen(!open)} aria-expanded={open}>{open ? "done" : "adjust"}</button>
              <button class="icon" onClick={() => p.onRemove(i)} title="Take this electrode out" aria-label="Take this electrode out">×</button>
            </span>
          )}
          <div class="faces">
            <span class="faces-says" title="The scanner pushes the neuron, asks “Who are you?” and reads the next word of “I am a …”">I am a…</span>
            {([1, -1] as const).map((s) => {
              const r = reading(focus.L, faceOf(focus.n, s));
              const k = p.catalog.find((c) => c.probes.length === 1 && same(c.probes[0], focus) && c.probes[0].dose * s > 0);
              return (
                <button key={s} class={`face ${s > 0 ? "pos" : "neg"} ${sign === s ? "on" : ""}`} onClick={() => p.onFace(focus.L, focus.n, s)} aria-pressed={sign === s}
                  title={sign === s ? "Pushed this way now. Click to take the electrode out." : s > 0 ? "Push this neuron up and ask the patient again" : "Push this neuron down and ask the patient again"}>
                  <b>{s > 0 ? "▲ " : "▼ "}<span class="long">push </span>{s > 0 ? "up" : "down"}</b>
                  {r ? <span><span class={`word k${r.kind}`}>{r.word.trim() || "⏎"}</span> <i>{Math.round(r.p * 100)}%</i></span> : <span><i>not read yet</i></span>}
                  {k && <em>{emojiOf(k)}</em>}
                </button>
              );
            })}
          </div>
        </div>
      )}
      {others.length > 0 && (
        <div class="neuron-others">
          {others.map(({ q, j }) => {
            const k = ownerOf(q, p.catalog, p.probes);
            return (
              <div class={`neuron-chip ${q.dose >= 0 ? "pos" : "neg"}`} key={`${q.L}:${q.n}`}>
                <button class="neuron-chip-main" onClick={() => p.onFocus(q.L, q.n)} title="Show this electrode">
                  <span>{k ? emojiOf(k) : UNNAMED}</span><b>{k?.name ?? "Unnamed neuron"}</b><i><span class="where">L{q.L} · N{q.n} </span>{q.dose >= 0 ? "▲" : "▼"} {Math.abs(q.dose).toFixed(2)}</i>
                </button>
                <button class="icon" onClick={() => p.onRemove(j)} title="Take this electrode out" aria-label="Take this electrode out">×</button>
              </div>
            );
          })}
        </div>
      )}
      {p.probes.length > 0 && (
        <div class={`neuron-foot ${open || p.adding ? "needed" : ""} ${p.probes.length > 1 ? "many" : ""}`}>
          {p.adding
            ? <span class="adding">Tap another neuron to add it. <button class="link" onClick={() => p.onAdding(false)}>cancel</button></span>
            : <button class="link" onClick={() => p.onAdding(true)} title="Keep what is pushed now and push one more neuron with it">+ add another neuron</button>}
          <button class="link right" onClick={p.onReset} title="Take every electrode out: the patient is intact again">reset the brain</button>
        </div>
      )}
    </div>
  );
}

// ---- layers ----------------------------------------------------------------------------------------------------

export interface LayerInfo { explored: number; surveyed: boolean; signals: number; named: number }

/** The 24 layers in a row, front of the brain on the left. Tap or drag to cut the brain open at a layer. */
export function LayerRail({ layer, layers, probes, onLayer }: { layer: number | null; layers: LayerInfo[]; probes: Probe[]; onLayer: (L: number) => void }) {
  const root = useRef<HTMLDivElement>(null);
  const dragging = useRef(false), last = useRef(-1);
  const pick = (e: PointerEvent, moved: boolean) => {
    const b = root.current!.getBoundingClientRect();
    const L = Math.max(0, Math.min(NUM_LAYERS - 1, Math.floor(((e.clientX - b.left) / b.width) * NUM_LAYERS)));
    if (moved && L === last.current) return; // a drag reports a layer once; a tap on the open layer is passed on (it resets the view)
    last.current = L;
    onLayer(L);
  };
  return (
    <div class="rail" ref={root} role="slider" aria-label="Layer" aria-valuemin={0} aria-valuemax={NUM_LAYERS - 1} aria-valuenow={layer ?? undefined} tabIndex={0}
      onPointerDown={(e) => { dragging.current = true; try { (e.currentTarget as Element).setPointerCapture(e.pointerId); } catch { /* a pointer that is already gone */ } pick(e, false); }}
      onPointerMove={(e) => { if (dragging.current) pick(e, true); }}
      onKeyDown={(e) => { if (e.key === "Home") onLayer(0); if (e.key === "End") onLayer(NUM_LAYERS - 1); }}
      onPointerUp={() => { dragging.current = false; }} onPointerCancel={() => { dragging.current = false; }}>
      {layers.map((info, L) => {
        const e = probes.find((q) => q.L === L), f = L / (NUM_LAYERS - 1);
        // the colour the layer has in the brain: violet at the front, rose at the back
        const hue = `rgb(${Math.round(92 + 94 * f)},${Math.round(84 - 12 * f)},${Math.round(196 - 62 * f)})`;
        return (
          <i key={L} class={`tick ${L === layer ? "on" : ""} ${e ? (e.dose >= 0 ? "pos" : "neg") : ""}`} style={{ "--explored": info.explored, "--h": hue }}
            title={`Layer ${L}: ${info.explored >= 1 ? "charted" : info.explored > 0 ? `${Math.round(info.explored * 100)}% charted` : "uncharted"}${info.named ? `, ${info.named} named` : ""}`}>
            {info.named > 0 && <u />}
          </i>
        );
      })}
    </div>
  );
}

// ---- atlas: the neurons that have names ------------------------------------------------------------------------

/** Scrolls a card into view inside the drawer, without moving the page. */
function reveal(el: HTMLElement) {
  const box = el.closest<HTMLElement>(".drawer-body");
  if (!box || box.scrollHeight <= box.clientHeight + 1) return;
  const r = el.getBoundingClientRect(), c = box.getBoundingClientRect();
  if (r.top < c.top || r.bottom > c.bottom) box.scrollTo({ top: box.scrollTop + r.top - c.top - (c.height - r.height) / 2, behavior: "smooth" });
}

/** Every neuron (or combination) that has a name: the ones the site ships with and the player's own. */
export function Atlas({ catalog, probes, onPick, onForget }: { catalog: Knob[]; probes: Probe[]; onPick: (k: Knob, mix: boolean) => void; onForget: (k: Knob) => void }) {
  const [tag, setTag] = useState("all");
  const root = useRef<HTMLDivElement>(null);
  const tags = ["all", ...Array.from(new Set(catalog.flatMap((k) => k.tags ?? [])))];
  const shown = catalog.filter((k) => tag === "all" || k.tags?.includes(tag));
  const firstOn = shown.find((k) => isShownActive(k, probes, catalog))?.name;
  useEffect(() => {
    const el = firstOn ? root.current?.querySelector<HTMLElement>(".card.on") : null;
    if (el) reveal(el);
  }, [firstOn]);
  return (
    <div class="shelf" ref={root}>
      <div class="tags">
        {tags.map((t) => <button key={t} class={`chip ${tag === t ? "on" : ""}`} onClick={() => setTag(t)}>{t}</button>)}
      </div>
      <div class="cards">
        {catalog.length === 0 && <div class="dim pad">loading the atlas…</div>}
        {shown.map((k) => {
          const on = isShownActive(k, probes, catalog);
          return (
            <div class={`card ${on ? "on" : ""}`} key={k.name}>
              <button class="card-main" onClick={() => onPick(k, false)} title={k.desc}>
                <span class="card-emoji">{emojiOf(k)}</span>
                <span class="card-name">{k.name}{k.origin && <span class="badge">{k.origin}</span>}</span>
                <span class="card-meta">{k.probes.length === 1 ? `layer ${k.probes[0].L}` : neuronCount(k.probes.length)}</span>
                <span class="card-quote">“{lead(k.sample ?? k.desc, 30, 96)}”</span>
              </button>
              {k.origin === "yours" && <button class="card-mix card-forget" onClick={() => onForget(k)} title="Remove this name from your atlas">×</button>}
              {!on && !k.origin && probes.length > 0 && <button class="card-mix" onClick={() => onPick(k, true)} title="Push this one together with what is pushed now">+</button>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** What the scanner found on a layer, strongest first: the list of things to go and listen to. */
export function SignalList({ L, version, heard, catalog, sel, ready, onPick }: {
  L: number; version: number; heard: Heard; catalog: Knob[]; sel: Sel | null; ready: boolean; onPick: (n: number, sign: 1 | -1) => void;
}) {
  const list = useMemo(() => signals(L, WEAK).slice(0, 400), [L, version]);
  const named = useMemo(() => {
    const m = new Map<string, Knob>();
    for (const k of catalog) if (k.probes.length === 1 && k.probes[0].L === L) m.set(`${k.probes[0].n}:${k.probes[0].dose >= 0 ? 1 : -1}`, k);
    return m;
  }, [catalog, L]);
  if (!list.length) return <div class="dim pad">Layer {L} has not been scanned yet. {ready ? "Scan it, or tap any of its neurons." : "Wake the patient, then scan it."}</div>;
  const rows: ComponentChildren[] = [];
  list.forEach((s, i) => {
    const k = named.get(`${s.n}:${s.sign}`), quote = heard[heardKey(L, s.n, s.sign)];
    if (s.kind < STRONG && (i === 0 || list[i - 1].kind >= STRONG)) rows.push(<div class="sig-rule" key="rule">weaker signals: less certain, or only a change of mood</div>);
    rows.push(
      <button class={`sig k${s.kind} ${sel && sel.L === L && sel.n === s.n ? "on" : ""} ${quote !== undefined ? "heard" : ""}`} key={`${s.n}:${s.sign}`} onClick={() => onPick(s.n, s.sign)}
        title={`Layer ${L}, neuron ${s.n}, pushed ${s.sign > 0 ? "up" : "down"}: the model would call itself “a${s.word}” (${Math.round(s.p * 100)}%)`}>
        <span class="sig-word">{s.word.trim() || "⏎"}</span>
        <span class="sig-p">{Math.round(s.p * 100)}%</span>
        <span class={`sig-id ${s.sign > 0 ? "pos" : "neg"}`}>{s.n}{s.sign > 0 ? "▲" : "▼"}</span>
        <span class="sig-note">{k ? `${emojiOf(k)} ${k.name}` : quote !== undefined ? `“${quote}”` : ""}</span>
      </button>,
    );
  });
  return <div class="signals">{rows}</div>;
}
