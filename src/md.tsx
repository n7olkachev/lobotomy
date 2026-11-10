// A small, safe markdown renderer for model output (no innerHTML), plus the "stuck in a loop" echo.
import type { ComponentChildren } from "preact";
import { findLoop } from "./text";

const INLINE = /`([^`\n]+)`|\*\*(.+?)\*\*|(^|[\s(])\*([^*\s](?:[^*\n]*?[^*\s])?)\*(?=[\s).,!?:;'"]|$)/g;
const stray = (s: string) => s.replace(/\*\*|__/g, "");

function inline(s: string): ComponentChildren[] {
  const out: ComponentChildren[] = [];
  let last = 0;
  INLINE.lastIndex = 0;
  for (let m = INLINE.exec(s); m; m = INLINE.exec(s)) {
    if (m.index > last) out.push(stray(s.slice(last, m.index)));
    if (m[1] !== undefined) out.push(<code>{m[1]}</code>);
    else if (m[2] !== undefined) out.push(<b>{stray(m[2])}</b>);
    else { if (m[3]) out.push(m[3]); out.push(<i>{m[4]}</i>); }
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push(stray(s.slice(last)));
  return out;
}

/** `tail` is appended inside the last text block, so it flows on the same line as the end of the text. */
export function Md({ text, tail }: { text: string; tail?: ComponentChildren }) {
  const blocks: { cls: string; body: ComponentChildren[]; code?: boolean }[] = [];
  let code: string[] | null = null;
  let gap = false;
  for (const ln of text.split("\n")) {
    if (/^\s*```/.test(ln)) {
      if (code) { blocks.push({ cls: "md-code", body: [code.join("\n")], code: true }); code = null; } else code = [];
      continue;
    }
    if (code) { code.push(ln); continue; }
    if (!ln.trim() || /^[\s*_#>-]+$/.test(ln)) { gap = blocks.length > 0; continue; } // blank, or markup with nothing in it
    const g = gap ? " md-gap" : "";
    gap = false;
    let m: RegExpExecArray | null;
    if ((m = /^\s*(#{1,6})\s+(.*)$/.exec(ln))) blocks.push({ cls: `md-h md-h${Math.min(3, m[1].length)}${g}`, body: inline(m[2]) });
    else if ((m = /^\s*([*\-•]|\d+[.)])\s+(.*)$/.exec(ln))) blocks.push({ cls: `md-li${g}`, body: [<span class="md-b">{/\d/.test(m[1]) ? m[1] : "•"}</span>, <span>{inline(m[2])}</span>] });
    else blocks.push({ cls: `md-p${g}`, body: inline(ln) });
  }
  if (code) blocks.push({ cls: "md-code", body: [code.join("\n")], code: true });
  const lastText = blocks.length && !blocks[blocks.length - 1].code ? blocks.length - 1 : -1;
  return (
    <>
      {blocks.map((b, i) => (b.code ? <pre class={b.cls} key={i}>{b.body}</pre> : <div class={b.cls} key={i}>{b.body}{i === lastText && tail}</div>))}
      {lastText === -1 && tail}
    </>
  );
}

/** An answer with its loop collapsed: the repeating unit once, two fading echoes, and a count. */
export function Answer({ text, live }: { text: string; live?: boolean }) {
  const caret = live ? <span class="caret" /> : null;
  const loop = findLoop(text);
  if (!loop) return <div class="md"><Md text={text} tail={caret} /></div>;
  const head = text.slice(0, loop.start);
  const unit = loop.unit.replace(/\s+$/, "");
  const badge = <span class="loop-badge" title={`Stuck in a loop: the same phrase ${loop.repeats} times in a row. Generation was cut short.`}>↻ ×{loop.repeats}</span>;
  const block = unit.includes("\n") || head === "" || /\n\s*$/.test(head);
  if (!block) {
    const sep = /\s$/.test(loop.unit) ? " " : "";
    const echoes = <>{[1, 2].map((i) => <span class={`echo echo-${i}`} key={i}>{sep}{inline(unit)}</span>)} {badge}{caret}</>;
    return <div class="md"><Md text={head + unit} tail={echoes} /></div>;
  }
  const echoes = unit.length <= 90 ? [1, 2] : [1];
  return (
    <div class="md">
      <Md text={head + unit} />
      {echoes.map((i) => <div class={`echo echo-${i} md-gap`} key={i}><Md text={unit} /></div>)}
      <div class="md-p">{badge}{caret}</div>
    </div>
  );
}
