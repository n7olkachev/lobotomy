// Recipes, dose normalisation, steer tensor. Shared by UI and worker.

export interface Probe { L: number; n: number; dose: number }
export interface Stats { num_layers: number; intermediate: number; resid_median: number[]; down: Float32Array }

/** "L14N8949+1 L13N7956-1" or comma separated -> probes */
export function parseRecipe(s: string): Probe[] {
  return s.trim().split(/[\s,]+/).filter(Boolean).map((t) => {
    const m = /^L(\d+)N(\d+)([+-]\d*\.?\d+)$/i.exec(t);
    if (!m) throw new Error(`bad recipe token "${t}"`);
    return { L: +m[1], n: +m[2], dose: +m[3] };
  });
}

export function formatProbe(p: Probe): string {
  const d = Math.round(p.dose * 100) / 100;
  return `L${p.L}N${p.n}${d >= 0 ? "+" : ""}${d}`;
}

export function formatRecipe(ps: Probe[]): string {
  return ps.map(formatProbe).join(" ");
}

/** steer[L, n] = dose * median_resid_norm_after_L / ||W_down[:, n]|| */
export function probesToSteer(ps: Probe[], st: Stats): Float32Array {
  const { num_layers: NL, intermediate: I } = st;
  const d = new Float32Array(NL * I);
  for (const { L, n, dose } of ps) {
    if (L < 0 || L >= NL || n < 0 || n >= I) continue;
    d[L * I + n] += (dose * st.resid_median[L]) / st.down[L * I + n];
  }
  return d;
}

export async function loadStats(base: string): Promise<Stats> {
  const [st, bin] = await Promise.all([
    fetch(`${base}stats.json`).then((r) => r.json()),
    fetch(`${base}down_norms.bin`).then((r) => r.arrayBuffer()),
  ]);
  return { num_layers: st.num_layers, intermediate: st.intermediate, resid_median: st.resid_median, down: new Float32Array(bin) };
}
