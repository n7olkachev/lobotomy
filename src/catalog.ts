// Named knobs for Qwen3.5-0.8B, loaded from /catalog.json (built from the sweep in ~/ai/neuron-onnx/stats).
import type { Probe } from "./steer";

export interface Knob {
  name: string; desc: string; probes: Probe[]; tags?: string[]; sample?: string; emoji?: string;
  /** named by the player (kept in their browser), or carried by a link somebody shared */
  origin?: "yours" | "shared";
}

export const NUM_LAYERS = 24;
export const INTERMEDIATE = 3584;

export async function loadCatalog(): Promise<Knob[]> {
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}catalog.json`, { cache: "no-cache" });
    if (!r.ok) return [];
    return (await r.json()) as Knob[];
  } catch {
    return [];
  }
}
