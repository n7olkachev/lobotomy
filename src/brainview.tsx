// The brain on screen: the WebGL canvas, the labels that float on it, and the gestures.
// Drag turns it, pinch or the wheel zooms, a tap picks the neuron under the finger.
import { useEffect, useRef, useState } from "preact/hooks";
import { INTERMEDIATE as I, NUM_LAYERS as NL } from "./catalog";
import { rim } from "./geo";
import { BrainGL, type Hit, type Photo, type Power } from "./gl";
import type { Probe } from "./steer";
import { CHATTER, NONE, ODD, QUIET, STRONG, WEAK, faceOf, layerVersion, onReveal, reading, tile } from "./store";

/** A neuron that has a name, pushed one way. */
export interface Pin { L: number; n: number; sign: 1 | -1; emoji: string; name: string }
export interface BrainApi {
  /** a token was generated: a wave runs through the layers */
  pulse(steered: boolean): void;
  /** an electrode went in */
  shock(L: number, n: number, sign: 1 | -1): void;
  resetView(): void;
  /** a still of the brain for the share image */
  snapshot(): Photo | null;
}
interface Pop { id: number; x: number; y: number; word: string }

// ---- colours: what the scanner read off a neuron -------------------------------------------------------------
// Tissue nobody has read is dim and takes the colour of its place in the brain: violet at the front, rose at the back.
const FRONT = [66, 60, 132], BACK = [124, 48, 88], SEEN = [42, 54, 70], BUSY = [88, 120, 152], DIM = [128, 78, 54], WARM = [222, 122, 58];
const HOT = [255, 150, 68], HOTTER = [255, 226, 150], WHITE = [255, 250, 232], STRANGE = [181, 140, 255];
function hash(n: number, L: number) {
  const h = Math.sin(n * 12.9898 + L * 78.233) * 43758.5453;
  return h - Math.floor(h);
}
function paint(L: number, out: Uint8Array) {
  const f = L / (NL - 1), tissue = FRONT.map((v, i) => v + (BACK[i] - v) * f);
  const put = (n: number, a: number[], b: number[], t: number, hot: number, gain = 1) => {
    const k = Math.max(0, Math.min(1, t)), o = n * 4;
    out[o] = Math.min(255, (a[0] + (b[0] - a[0]) * k) * gain); out[o + 1] = Math.min(255, (a[1] + (b[1] - a[1]) * k) * gain); out[o + 2] = Math.min(255, (a[2] + (b[2] - a[2]) * k) * gain); out[o + 3] = hot * 255;
  };
  for (let n = 0; n < I; n++) {
    const t = tile(L, n), edge = rim[L * I + n] ? 1.7 : 1;
    switch (t.kind) {
      case NONE: put(n, tissue, tissue, 0, 0, (0.7 + hash(n, L) * 0.5) * edge); break;
      case QUIET: put(n, SEEN, BUSY, t.kl / 30, 0, edge); break;
      case CHATTER: put(n, SEEN, BUSY, 0.15 + t.kl / 14, 0, edge); break;
      case WEAK: put(n, DIM, WARM, (t.p - 0.12) / 0.4, 0.2); break;
      case ODD: put(n, STRANGE, STRANGE, 0, 0.4); break;
      default: if (t.p < 0.6) put(n, HOT, HOTTER, (t.p - 0.3) / 0.3, 0.55 + t.p * 0.4); else put(n, HOTTER, WHITE, (t.p - 0.6) / 0.35, 0.55 + t.p * 0.45);
    }
  }
}
/** How much a neuron is worth a tap, for fingers that cover a dozen of them. */
const interest = (L: number, n: number) => { const k = tile(L, n).kind; return k === STRONG ? 3 : k === ODD ? 2 : k === WEAK ? 1 : 0; };

const keyOf = (p: { L: number; n: number; sign: number }) => `${p.L}:${p.n}:${p.sign > 0 ? "+" : "-"}`;

export function BrainView(p: {
  layer: number | null; sel: Hit | null; probes: Probe[]; pins: Pin[]; version: number;
  power: Power; progress: number;
  /** the model is answering or scanning */
  working: boolean;
  /** parts of the canvas that are under the controls, in CSS pixels */
  inset: { top: number; right: number; bottom: number; left: number };
  api: { current: BrainApi | null };
  onPick: (L: number, n: number, sign?: 1 | -1) => void;
  /** the visitor took hold of the brain */
  onTouch: () => void;
}) {
  const wrap = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null), tip = useRef<HTMLDivElement>(null);
  const glRef = useRef<BrainGL | null>(null);
  const nodes = useRef(new Map<string, HTMLElement>());
  const painted = useRef(new Int32Array(NL).fill(-1));
  const [failed, setFailed] = useState(false);
  const [hover, setHover] = useState<Hit | null>(null);
  const [pops, setPops] = useState<Pop[]>([]);
  const [grabbing, setGrabbing] = useState(false);
  const ptrs = useRef(new Map<number, { x: number; y: number }>()); // fingers (or the mouse button) that are down on the canvas
  const live = useRef({ ...p, hover });
  live.current = { ...p, hover };

  // electrodes in neurons nobody has named get a plain tag
  const named = new Set(p.pins.map(keyOf));
  const loose = p.probes.filter((q) => !named.has(keyOf({ ...q, sign: q.dose >= 0 ? 1 : -1 })));
  const pushed = new Set(p.probes.map((q) => keyOf({ ...q, sign: q.dose >= 0 ? 1 : -1 })));

  /** Move every label to where its neuron is now. Runs after each frame, outside of rendering. */
  const place = () => {
    const gl = glRef.current;
    if (!gl) return;
    const { pins, layer, probes, hover } = live.current;
    const put = (key: string, L: number, n: number) => {
      const el = nodes.current.get(key);
      if (!el) return;
      const s = gl.project(L, n), o = layer === null || L === layer ? s.shown : 0;
      if (o < 0.06) { if (el.style.display !== "none") el.style.display = "none"; return; }
      el.style.display = "";
      el.style.opacity = Math.min(1, o * 1.2).toFixed(2);
      el.style.transform = `translate(${s.x.toFixed(1)}px,${s.y.toFixed(1)}px)`;
      el.style.zIndex = String(Math.max(1, 1000 - Math.round(s.depth * 100)));
    };
    for (const pin of pins) put("p" + keyOf(pin), pin.L, pin.n);
    for (const q of probes) put("e" + keyOf({ ...q, sign: q.dose >= 0 ? 1 : -1 }), q.L, q.n);
    // names on the open slice: one that would run into a name already placed stays off (the pushed ones go first)
    if (layer !== null) {
      const taken: { x: number; y: number; w: number }[] = [];
      const on = new Set(probes.map((q) => keyOf({ ...q, sign: q.dose >= 0 ? 1 : -1 })));
      const here = pins.filter((pin) => pin.L === layer).sort((a, b) => +on.has(keyOf(b)) - +on.has(keyOf(a)));
      for (const pin of here) {
        const el = nodes.current.get("p" + keyOf(pin));
        if (!el) continue;
        const s = gl.project(pin.L, pin.n), w = pin.name.length * 6.6 + 16;
        const clash = taken.some((t) => Math.abs(t.y - s.y) < 30 && Math.abs(t.x - s.x) < (t.w + w) / 2 + 4);
        if (!clash) taken.push({ x: s.x, y: s.y, w });
        el.classList.toggle("crowded", clash);
      }
    }
    const t = tip.current, box = wrap.current;
    if (t && hover && box) {
      // over the neuron; under it near the top of the canvas, and pulled in from the sides
      const s = gl.project(hover.L, hover.n), half = Math.min(120, box.clientWidth / 2);
      t.style.transform = `translate(${s.x.toFixed(1)}px,${s.y.toFixed(1)}px)`;
      t.style.setProperty("--dx", `${(Math.max(half, Math.min(box.clientWidth - half, s.x)) - s.x).toFixed(1)}px`);
      t.classList.toggle("below", s.y < 110);
    }
  };

  useEffect(() => {
    let gl: BrainGL;
    try { gl = new BrainGL(canvas.current!); setFailed(false); } catch (err) { console.warn(err); setFailed(true); return; }
    glRef.current = gl;
    const el = wrap.current!;
    const fit = () => {
      gl.resize(el.clientWidth, el.clientHeight, devicePixelRatio);
      // the emoji on the whole brain are as big as the brain allows
      el.style.setProperty("--pin", `${Math.max(9, Math.min(15, Math.min(el.clientWidth, el.clientHeight) / 34)).toFixed(1)}px`);
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    gl.onFrame = () => place();
    p.api.current = {
      pulse: (steered) => gl.pulse(steered),
      shock: (L, n, sign) => gl.shock(L, n, sign),
      resetView: () => gl.resetView(),
      snapshot: () => {
        const photo = gl.snapshot(), { pins, probes } = live.current;
        photo.marks.forEach((m, i) => {
          const q = probes[i], pin = q && pins.find((x) => x.L === q.L && x.n === q.n && x.sign === (q.dose >= 0 ? 1 : -1));
          m.label = pin ? `${pin.emoji} ${pin.name.replace(/^part of /, "")}` : q ? `L${q.L} · N${q.n}` : "";
        });
        return photo;
      },
    };
    // the wheel zooms (and must not scroll the page): needs a listener that is not passive
    const cv = canvas.current!;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const b = cv.getBoundingClientRect();
      gl.zoomAt(Math.exp(e.deltaY * (e.ctrlKey ? 0.012 : 0.0016)), e.clientX - b.left, e.clientY - b.top);
      live.current.onTouch();
    };
    cv.addEventListener("wheel", wheel, { passive: false });
    // Safari reports a pinch on a trackpad as a gesture, not as a wheel (fingers on a screen are pointers: handled below)
    let scale = 1;
    const pinch = (e: Event) => {
      e.preventDefault();
      const g = e as Event & { scale: number; clientX: number; clientY: number };
      if (e.type === "gesturechange" && !ptrs.current.size && g.scale > 0) {
        const b = cv.getBoundingClientRect();
        gl.zoomAt(scale / g.scale, g.clientX - b.left, g.clientY - b.top);
        live.current.onTouch();
      }
      scale = g.scale || 1;
    };
    cv.addEventListener("gesturestart", pinch);
    cv.addEventListener("gesturechange", pinch);
    return () => {
      cv.removeEventListener("wheel", wheel); cv.removeEventListener("gesturestart", pinch); cv.removeEventListener("gesturechange", pinch);
      ro.disconnect(); gl.dispose(); glRef.current = null; p.api.current = null; painted.current.fill(-1);
    };
  }, []);

  // what the map knows, as colours
  useEffect(() => {
    const gl = glRef.current;
    if (!gl) return;
    const buf = new Uint8Array(I * 4);
    for (let L = 0; L < NL; L++) {
      const v = layerVersion(L);
      if (painted.current[L] === v) continue;
      painted.current[L] = v;
      paint(L, buf);
      gl.setColors(L, buf);
    }
  }, [p.version, failed]);
  useEffect(() => { glRef.current?.setLayer(p.layer); }, [p.layer]);
  useEffect(() => { glRef.current?.setSelected(p.sel); }, [p.sel?.L, p.sel?.n]);
  useEffect(() => { glRef.current?.setElectrodes(p.probes.map((q) => ({ L: q.L, n: q.n, sign: q.dose >= 0 ? 1 : -1, strength: Math.abs(q.dose) }))); }, [p.probes]);
  useEffect(() => { glRef.current?.setPower(p.power, p.progress); }, [p.power, p.progress]);
  useEffect(() => { glRef.current?.setWorking(p.working); }, [p.working]);
  useEffect(() => { glRef.current?.setInsets(p.inset.top, p.inset.right, p.inset.bottom, p.inset.left); }, [p.inset.top, p.inset.right, p.inset.bottom, p.inset.left]);
  useEffect(() => { place(); });

  // faces the scanner reads right now: the neuron flashes, and a strong word floats up from it
  useEffect(() => onReveal((r) => {
    const gl = glRef.current;
    if (!gl) return;
    gl.flash(r.L, r.n);
    const layer = live.current.layer;
    if (r.kind < STRONG || (layer !== null && layer !== r.L)) return;
    const s = gl.project(r.L, r.n);
    if (s.shown > 0.5) setPops((ps) => [...ps.slice(-7), { id: Math.random(), x: s.x, y: s.y, word: r.word.trim() }]);
  }), []);

  // ---- gestures ----------------------------------------------------------------------------------------------
  const gest = useRef({ moved: false, pinch: false, pan: false, x0: 0, y0: 0, dist: 0, cx: 0, cy: 0 });
  const at = (e: PointerEvent) => { const b = canvas.current!.getBoundingClientRect(); return { x: e.clientX - b.left, y: e.clientY - b.top }; };
  const hits = () => live.current.pins as Hit[];
  const pinAt = (h: Hit) => live.current.pins.find((q) => q.L === h.L && q.n === h.n);

  const down = (e: PointerEvent) => {
    const gl = glRef.current;
    if (!gl) return;
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); } catch { /* a pointer that is already gone */ }
    const q = at(e);
    ptrs.current.set(e.pointerId, q);
    if (ptrs.current.size === 1) {
      gest.current = { moved: false, pinch: false, pan: e.button !== 0 || e.shiftKey, x0: q.x, y0: q.y, dist: 0, cx: 0, cy: 0 };
      gl.grab(true);
    } else if (ptrs.current.size === 2) {
      const [a, b] = [...ptrs.current.values()];
      Object.assign(gest.current, { pinch: true, moved: true, dist: Math.hypot(a.x - b.x, a.y - b.y), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 });
    }
  };
  const move = (e: PointerEvent) => {
    const gl = glRef.current;
    if (!gl) return;
    const q = at(e), was = ptrs.current.get(e.pointerId), g = gest.current;
    if (!was) {
      if (e.pointerType !== "mouse") return;
      const h = gl.pick(q.x, q.y, hits(), 5);
      gl.setHover(h);
      setHover((old) => (old?.L === h?.L && old?.n === h?.n ? old : h));
      return;
    }
    const px = was.x, py = was.y;
    was.x = q.x; was.y = q.y;
    if (ptrs.current.size === 1 && !g.pinch) {
      if (!g.moved && Math.hypot(q.x - g.x0, q.y - g.y0) > (e.pointerType === "mouse" ? 4 : 10)) {
        g.moved = true;
        setGrabbing(true); setHover(null); gl.setHover(null);
        live.current.onTouch();
      }
      if (g.moved) { if (g.pan) gl.pan(px, py, q.x, q.y); else gl.orbit(q.x - px, q.y - py); }
    } else if (ptrs.current.size === 2) {
      const [a, b] = [...ptrs.current.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y), cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
      if (g.dist > 0 && dist > 0) gl.zoomAt(g.dist / dist, cx, cy);
      gl.pan(g.cx, g.cy, cx, cy);
      Object.assign(g, { dist, cx, cy });
      live.current.onTouch();
    }
  };
  const up = (e: PointerEvent) => {
    const gl = glRef.current;
    if (!gl || !ptrs.current.delete(e.pointerId) || ptrs.current.size) return;
    gl.grab(false);
    setGrabbing(false);
    const g = gest.current;
    if (g.moved || g.pinch || e.type !== "pointerup") return;
    const q = at(e), touch = e.pointerType !== "mouse";
    const h = gl.pick(q.x, q.y, hits(), touch ? 18 : 6, touch ? interest : undefined);
    if (h) live.current.onPick(h.L, h.n, pinAt(h)?.sign);
  };
  const leave = () => { glRef.current?.setHover(null); setHover(null); };

  const tipPin = (sign: 1 | -1) => (hover ? p.pins.find((q) => q.L === hover.L && q.n === hover.n && q.sign === sign) : undefined);
  return (
    <div class={`brain ${grabbing ? "grabbing" : ""} ${hover ? "aiming" : ""}`} ref={wrap}>
      <canvas ref={canvas} role="img" aria-label="The brain of the model in 3D: 24 layers of 3,584 neurons. The atlas lists the named ones." onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onPointerLeave={leave} onContextMenu={(e) => e.preventDefault()} />
      {failed && <div class="brain-fail">This browser cannot draw the brain (it needs WebGL2).<br />The atlas and the patient still work.</div>}
      <div class="labels">
        {p.pins.map((pin) => {
          const k = keyOf(pin), on = pushed.has(k);
          return (
            <span key={"p" + k} ref={(el) => { if (el) nodes.current.set("p" + k, el); else nodes.current.delete("p" + k); }}
              class={`pin ${pin.L === p.layer ? "open" : ""} ${on ? `on ${pin.sign > 0 ? "pos" : "neg"}` : ""}`} style={{ display: "none" }}>
              <i>{pin.emoji}</i><b>{pin.name}</b>
            </span>
          );
        })}
        {loose.map((q) => {
          const k = keyOf({ ...q, sign: q.dose >= 0 ? 1 : -1 });
          return (
            <span key={"e" + k} ref={(el) => { if (el) nodes.current.set("e" + k, el); else nodes.current.delete("e" + k); }} class={`pin open on tag ${q.dose >= 0 ? "pos" : "neg"}`} style={{ display: "none" }}>
              <b>L{q.L} · N{q.n} {q.dose >= 0 ? "▲" : "▼"}</b>
            </span>
          );
        })}
        {pops.map((q) => <span class="pop" key={q.id} style={{ left: `${q.x}px`, top: `${q.y}px` }} onAnimationEnd={() => setPops((ps) => ps.filter((x) => x.id !== q.id))}>{q.word}</span>)}
        {hover && (
          <div class="tip" ref={tip}>
            <div class="tip-box">
              <div class="tip-id">layer {hover.L} · neuron {hover.n}</div>
              {([1, -1] as const).map((sign) => {
                const f = reading(hover.L, faceOf(hover.n, sign)), pin = tipPin(sign);
                return (
                  <div class="tip-face" key={sign}>
                    <b class={sign > 0 ? "pos" : "neg"}>{sign > 0 ? "▲" : "▼"}</b> {f ? <>I am a <span class={`word k${f.kind}`}>{f.word.trim() || "⏎"}</span> <i>{Math.round(f.p * 100)}%</i></> : <i>not read yet</i>}
                    {pin && <span class="tip-name"> · {pin.emoji} {pin.name}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
