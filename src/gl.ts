// The brain, drawn with WebGL2: 86,016 neurons as points of light on 24 plates of glass.
// One class owns the canvas: camera, the cut through the brain, the animations, and finding the neuron under a finger.
import { INTERMEDIATE as I, NUM_LAYERS as NL } from "./catalog";
import { neuronNear, neuronsAround, plateRange, plates, positions, slices } from "./geo";

export interface Hit { L: number; n: number }
export interface Electrode { L: number; n: number; sign: 1 | -1; strength: number }
export type Power = "asleep" | "loading" | "awake";
export interface Projected { x: number; y: number; depth: number; shown: number }
/** A still of the canvas: the square (in its pixels) that holds the brain, and where the electrodes are in it. */
export interface Photo { canvas: HTMLCanvasElement; x: number; y: number; size: number; marks: { x: number; y: number; sign: 1 | -1; label: string }[] }

// ---- 4x4 matrices, column-major ------------------------------------------------------------------------------
type M4 = Float32Array;
function perspective(o: M4, fovy: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  o.fill(0);
  o[0] = f / aspect; o[5] = f; o[10] = (far + near) * nf; o[11] = -1; o[14] = 2 * far * near * nf;
}
function lookAt(o: M4, ex: number, ey: number, ez: number, cx: number, cy: number, cz: number) {
  let zx = ex - cx, zy = ey - cy, zz = ez - cz, l = 1 / Math.hypot(zx, zy, zz);
  zx *= l; zy *= l; zz *= l;
  let xx = zz, xz = -zx; // up × z, with up = +y
  l = 1 / Math.hypot(xx, xz); xx *= l; xz *= l;
  const yx = zy * xz, yy = zz * xx - zx * xz, yz = -zy * xx;
  o[0] = xx; o[1] = yx; o[2] = zx; o[3] = 0;
  o[4] = 0; o[5] = yy; o[6] = zy; o[7] = 0;
  o[8] = xz; o[9] = yz; o[10] = zz; o[11] = 0;
  o[12] = -(xx * ex + xz * ez); o[13] = -(yx * ex + yy * ey + yz * ez); o[14] = -(zx * ex + zy * ey + zz * ez); o[15] = 1;
}
function mul(o: M4, a: M4, b: M4) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
}
function invert(o: M4, m: M4): boolean {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3], a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11], a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return false;
  det = 1 / det;
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det; o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det; o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det; o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det; o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det; o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det; o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det; o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det; o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return true;
}

// ---- shaders ---------------------------------------------------------------------------------------------------
const POINT_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;   // rgb, and how hot the neuron is (a signal glows)
layout(location=2) in float aFlash; // when the scanner read it
uniform mat4 uVP;
uniform float uPx, uPitch, uZ, uBright, uTime, uTwinkle, uMin;
uniform vec3 uTint; uniform float uTintAmt;
uniform vec4 uShock[2]; uniform vec3 uShockCol[2];
out vec3 vCol; out float vHot; out float vCore;
float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
void main() {
  vec4 clip = uVP * vec4(aPos.xy, aPos.z + uZ, 1.0);
  gl_Position = clip;
  float hot = aCol.a, id = float(gl_VertexID);
  float tw = sin(uTime * (0.6 + 2.4 * hash(id)) + 6.2831 * hash(id + 0.37));
  vec3 c = aCol.rgb * uBright * (1.0 + uTwinkle * 0.3 * tw);
  float lum = max(aCol.r, max(aCol.g, aCol.b));
  c = mix(c, uTint * (0.2 + 0.9 * lum), uTintAmt);
  float age = (uTime - aFlash) / 0.55;
  if (age >= 0.0 && age < 1.0) c += vec3(1.0, 0.96, 0.88) * (1.0 - age) * 1.5;
  for (int i = 0; i < 2; i++) {
    if (uShock[i].w > 0.0) { float d = distance(aPos.xy, uShock[i].xy) - uShock[i].z; c += uShockCol[i] * uShock[i].w * exp(-d * d * 240.0); }
  }
  float px = uPitch * uPx / clip.w;
  float dot = px * mix(0.66, 0.96, hot);
  float size = max(dot, uMin);
  c *= min(1.0, (dot * dot) / (size * size)); // a dot smaller than a pixel keeps its energy, not its size
  float halo = 1.0 + 1.5 * hot;
  gl_PointSize = size * halo + 2.0;
  vCore = size / (size * halo + 2.0);
  vCol = c; vHot = hot;
}`;
const POINT_FS = `#version 300 es
precision highp float;
in vec3 vCol; in float vHot; in float vCore;
out vec4 o;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0);
  float e = fwidth(r);
  float core = 1.0 - smoothstep(vCore - e, vCore + e, r);
  float glow = vHot * 0.34 * exp(-r * r * 5.0) * (1.0 - smoothstep(0.8, 1.0, r));
  o = vec4(vCol * (core + glow), 0.0);
}`;
const PLATE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uVP; uniform float uZ;
void main() { gl_Position = uVP * vec4(aPos.xy, aPos.z + uZ, 1.0); }`;
const PLATE_FS = `#version 300 es
precision highp float;
uniform vec4 uColor; out vec4 o;
void main() { o = uColor; }`;
const SPRITE_VS = `#version 300 es
precision highp float;
uniform mat4 uVP; uniform vec3 uCenter; uniform float uSize;
void main() { gl_Position = uVP * vec4(uCenter, 1.0); gl_PointSize = uSize; }`;
const SPRITE_FS = `#version 300 es
precision highp float;
uniform vec4 uColor; uniform int uKind; uniform float uSize;
out vec4 o;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0), a;
  if (uKind == 0) a = exp(-r * r * 4.5) * (1.0 - smoothstep(0.7, 1.0, r));            // a glow
  else { float w = 3.0 / uSize; a = 1.0 - smoothstep(w, w * 2.2, abs(r - (1.0 - w * 2.4))); } // a ring
  o = vec4(uColor.rgb * uColor.a * a, 0.0);
}`;
const LINE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec4 aPos; // clip-space xy, then where along the needle (0 = tip) and across it (-1..1)
out vec2 vUV;
void main() { gl_Position = vec4(aPos.xy, 0.0, 1.0); vUV = aPos.zw; }`;
const LINE_FS = `#version 300 es
precision highp float;
in vec2 vUV; uniform vec4 uColor; out vec4 o;
void main() {
  float v = abs(vUV.y);
  float core = 1.0 - smoothstep(0.1, 0.24, v), glow = exp(-v * v * 6.0) * 0.4;
  o = vec4(uColor.rgb * uColor.a * (core + glow) * mix(1.0, 0.1, vUV.x), 0.0);
}`;
const BG_VS = `#version 300 es
void main() { gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0); }`;
const BG_FS = `#version 300 es
precision highp float;
uniform vec2 uRes, uMid; out vec4 o;
void main() {
  float d = distance(gl_FragCoord.xy, uMid) / max(uRes.x, uRes.y);
  vec3 c = mix(vec3(0.062, 0.078, 0.108), vec3(0.018, 0.024, 0.034), smoothstep(0.0, 0.62, d));
  c += vec3(0.085, 0.03, 0.11) * exp(-d * d * 14.0); // a haze behind the brain
  c += (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 160.0; // no banding
  o = vec4(c, 1.0);
}`;

// The glow: the picture is drawn into a texture, its bright parts are blurred through a chain of ever smaller
// textures and added back on top.
const POST_VS = `#version 300 es
out vec2 vUV;
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  vUV = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;
const DOWN_FS = `#version 300 es
precision highp float;
in vec2 vUV; uniform sampler2D uTex; uniform vec2 uTexel; uniform float uCut;
out vec4 o;
void main() {
  vec3 c = texture(uTex, vUV).rgb * 4.0;
  c += texture(uTex, vUV + uTexel).rgb + texture(uTex, vUV - uTexel).rgb;
  c += texture(uTex, vUV + vec2(uTexel.x, -uTexel.y)).rgb + texture(uTex, vUV - vec2(uTexel.x, -uTexel.y)).rgb;
  c /= 8.0;
  o = vec4(max(c - uCut, 0.0), 1.0);
}`;
const UP_FS = `#version 300 es
precision highp float;
in vec2 vUV; uniform sampler2D uTex; uniform vec2 uTexel;
out vec4 o;
void main() {
  vec3 c = texture(uTex, vUV + vec2(uTexel.x, 0.0)).rgb + texture(uTex, vUV - vec2(uTexel.x, 0.0)).rgb;
  c += texture(uTex, vUV + vec2(0.0, uTexel.y)).rgb + texture(uTex, vUV - vec2(0.0, uTexel.y)).rgb;
  c += (texture(uTex, vUV + uTexel * 0.7).rgb + texture(uTex, vUV - uTexel * 0.7).rgb + texture(uTex, vUV + vec2(uTexel.x, -uTexel.y) * 0.7).rgb + texture(uTex, vUV - vec2(uTexel.x, -uTexel.y) * 0.7).rgb);
  o = vec4(c / 8.0, 1.0);
}`;
const MIX_FS = `#version 300 es
precision highp float;
in vec2 vUV; uniform sampler2D uTex, uGlow; uniform float uAmount;
out vec4 o;
void main() { o = vec4(texture(uTex, vUV).rgb + texture(uGlow, vUV).rgb * uAmount, 1.0); }`;

// ---- the renderer ----------------------------------------------------------------------------------------------
const FOV = (30 * Math.PI) / 180;
const WHOLE = { x: 0, y: -0.13, z: 0, r: 1.0 };
const POS: [number, number, number] = [1, 0.62, 0.28], NEG: [number, number, number] = [0.3, 0.75, 1], ACC: [number, number, number] = [0.5, 0.94, 0.84];
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const wrap = (a: number) => a - Math.round(a / (2 * Math.PI)) * 2 * Math.PI;

interface Cam { yaw: number; pitch: number; zoom: number; x: number; y: number; z: number; rad: number }
interface Prog { p: WebGLProgram; u: Record<string, WebGLUniformLocation | null> }

export class BrainGL {
  private gl!: WebGL2RenderingContext;
  private point!: Prog; private plate!: Prog; private sprite!: Prog; private line!: Prog; private bg!: Prog;
  private pointVAO!: WebGLVertexArrayObject; private plateVAO!: WebGLVertexArrayObject; private lineVAO!: WebGLVertexArrayObject; private emptyVAO!: WebGLVertexArrayObject;
  private colBuf!: WebGLBuffer; private flashBuf!: WebGLBuffer; private lineBuf!: WebGLBuffer;
  private colors = new Uint8Array(NL * I * 4);
  private flashes = new Float32Array(NL * I).fill(-1e6);
  private lineData = new Float32Array(24);
  private w = 1; private h = 1; private dpr = 1;
  private inset = { top: 0, right: 0, bottom: 0, left: 0 };
  private cam: Cam = { yaw: 0.95, pitch: 0.2, zoom: 1, ...WHOLE, rad: WHOLE.r };
  private goal: Cam = { ...this.cam };
  private vp = new Float32Array(16); private ivp = new Float32Array(16);
  private proj = new Float32Array(16); private view = new Float32Array(16); private tmp = new Float32Array(16);
  private eye: [number, number, number] = [0, 0, 1];
  private layer: number | null = null;
  // per layer, eased towards a target every frame: how visible, how bright, how opaque its plate, how far it is pulled out
  private vis = new Float32Array(NL).fill(1); private bright = new Float32Array(NL).fill(0.52);
  private glass = new Float32Array(NL).fill(0.5); private pull = new Float32Array(NL);
  private electrodes: Electrode[] = [];
  private selected: Hit | null = null; private hover: Hit | null = null;
  private power: Power = "asleep"; private progress = 0;
  private waves: { t: number; steered: boolean }[] = [];
  private shocks: { L: number; x: number; y: number; t: number; col: [number, number, number] }[] = [];
  private speaking = 0; // eased 0..1: the patient is answering
  private cut = 0; // eased 0..1: a slice is open (less glow, so that single neurons stay sharp)
  private working = false; // the model is busy on the GPU
  private calm = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches; // no swaying, no shimmer
  private owed = 0; // time since the last frame that was drawn
  private swaying = false; // nobody is steering: the whole brain turns by itself
  private lastInput = -1e9; private dragging = false;
  private time = 0; private last = 0; private raf = 0; private lost = false; private frameNo = 0;
  private settle = 0; // seconds of full frame rate left after something changed
  private owned: { programs: WebGLProgram[]; buffers: WebGLBuffer[]; arrays: WebGLVertexArrayObject[] } = { programs: [], buffers: [], arrays: [] };
  private down!: Prog; private up!: Prog; private mix!: Prog;
  /** the picture before the glow, and the chain of smaller textures the glow is blurred through */
  private targets: { fb: WebGLFramebuffer; tex: WebGLTexture; w: number; h: number }[] = [];
  /** Called after every frame that was drawn, to move the labels that sit on top of the canvas. */
  onFrame: (() => void) | null = null;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    this.init();
    canvas.addEventListener("webglcontextlost", this.onLost);
    canvas.addEventListener("webglcontextrestored", this.onRestored);
    this.raf = requestAnimationFrame(this.loop);
  }
  private onLost = (e: Event) => { e.preventDefault(); this.lost = true; };
  private onRestored = () => { this.lost = false; this.init(); this.wake(); };

  /** Stop drawing and give the GPU its memory back. The context itself stays: the canvas may be handed to a new renderer. */
  dispose() {
    cancelAnimationFrame(this.raf);
    this.canvas.removeEventListener("webglcontextlost", this.onLost);
    this.canvas.removeEventListener("webglcontextrestored", this.onRestored);
    const gl = this.gl;
    for (const p of this.owned.programs) gl.deleteProgram(p);
    for (const b of this.owned.buffers) gl.deleteBuffer(b);
    for (const v of this.owned.arrays) gl.deleteVertexArray(v);
    this.owned = { programs: [], buffers: [], arrays: [] };
    this.dropTargets();
  }
  private dropTargets() {
    for (const t of this.targets) { this.gl.deleteFramebuffer(t.fb); this.gl.deleteTexture(t.tex); }
    this.targets = [];
  }
  /** Textures to draw into: [0] is the size of the canvas, each of the others half the one before. */
  private ensureTargets(W: number, H: number) {
    if (this.targets.length && this.targets[0].w === W && this.targets[0].h === H) return;
    const gl = this.gl;
    this.dropTargets();
    for (let i = 0, w = W, h = H; i < 6 && w >= 4 && h >= 4; i++, w = Math.ceil(w / 2), h = Math.ceil(h / 2)) {
      const tex = gl.createTexture()!, fb = gl.createFramebuffer()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      this.targets.push({ fb, tex, w, h });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private program(vs: string, fs: string, uniforms: string[]): Prog {
    const gl = this.gl, p = gl.createProgram()!;
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("program: " + gl.getProgramInfoLog(p));
    const u: Prog["u"] = {};
    for (const name of uniforms) u[name] = gl.getUniformLocation(p, name);
    this.owned.programs.push(p);
    return { p, u };
  }

  private init() {
    const gl = this.gl;
    this.owned = { programs: [], buffers: [], arrays: [] };
    const buffer = (data: ArrayBufferView, usage: number) => { const b = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, usage); this.owned.buffers.push(b); return b; };
    const vao = () => { const v = gl.createVertexArray()!; this.owned.arrays.push(v); return v; };
    this.point = this.program(POINT_VS, POINT_FS, ["uVP", "uPx", "uPitch", "uZ", "uBright", "uTime", "uTwinkle", "uMin", "uTint", "uTintAmt", "uShock", "uShockCol"]);
    this.plate = this.program(PLATE_VS, PLATE_FS, ["uVP", "uZ", "uColor"]);
    this.sprite = this.program(SPRITE_VS, SPRITE_FS, ["uVP", "uCenter", "uSize", "uColor", "uKind"]);
    this.line = this.program(LINE_VS, LINE_FS, ["uColor"]);
    this.bg = this.program(BG_VS, BG_FS, ["uRes", "uMid"]);
    this.down = this.program(POST_VS, DOWN_FS, ["uTex", "uTexel", "uCut"]);
    this.up = this.program(POST_VS, UP_FS, ["uTex", "uTexel"]);
    this.mix = this.program(POST_VS, MIX_FS, ["uTex", "uGlow", "uAmount"]);
    this.targets = [];

    this.pointVAO = vao();
    gl.bindVertexArray(this.pointVAO);
    buffer(positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    this.colBuf = buffer(this.colors, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 0, 0);
    this.flashBuf = buffer(this.flashes, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);

    this.plateVAO = vao();
    gl.bindVertexArray(this.plateVAO);
    buffer(plates, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    this.lineVAO = vao();
    gl.bindVertexArray(this.lineVAO);
    this.lineBuf = buffer(this.lineData, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);

    this.emptyVAO = vao();
    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); // premultiplied: a plate (0,0,0,a) dims what is behind it, light (rgb,0) adds
  }

  // ---- what to show ------------------------------------------------------------------------------------------
  /** Keep drawing at full rate for a while: something changed. */
  wake(seconds = 1.2) { this.settle = Math.max(this.settle, seconds); }

  resize(w: number, h: number, dpr: number) {
    this.w = Math.max(1, w); this.h = Math.max(1, h); this.dpr = Math.min(2, dpr || 1);
    const cw = Math.round(this.w * this.dpr), ch = Math.round(this.h * this.dpr);
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw; this.canvas.height = ch;
      // A canvas that changes size is emptied: draw it again now, before the browser shows it black for a frame.
      if (!this.lost) { this.draw(); this.onFrame?.(); }
    }
    this.wake();
  }
  /** Parts of the canvas that are covered by controls: the brain is centred in what is left. */
  setInsets(top: number, right: number, bottom: number, left: number) { this.inset = { top, right, bottom, left }; this.wake(); }

  /** RGBA for the 3,584 neurons of a layer; alpha is how much the neuron glows. */
  setColors(L: number, rgba: Uint8Array) {
    this.colors.set(rgba, L * I * 4);
    if (this.lost) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, L * I * 4, rgba);
    this.wake(0.3);
  }
  /** The scanner has just read this neuron. */
  flash(L: number, n: number) {
    const i = L * I + n;
    this.flashes[i] = this.time;
    if (this.lost) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.flashBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, i * 4, this.flashes, i, 1);
    this.wake(0.8);
  }
  setElectrodes(list: Electrode[]) { this.electrodes = list; this.wake(); }
  setSelected(hit: Hit | null) { this.selected = hit; this.wake(); }
  setHover(hit: Hit | null) { if (hit?.L !== this.hover?.L || hit?.n !== this.hover?.n) { this.hover = hit; this.wake(0.2); } }
  setPower(power: Power, progress = 0) { this.power = power; this.progress = progress; this.wake(); }
  /** The model is answering or scanning: leave it most of the GPU. */
  setWorking(on: boolean) { this.working = on; }
  /** One token: a wave runs through the layers, front to back. Past the first electrode it takes its colour. */
  pulse(steered: boolean) {
    const last = this.waves[this.waves.length - 1];
    if (last && this.time - last.t < 0.2) return; // a few waves with dark between them read better than a blur of light
    if (this.waves.length > 7) this.waves.shift();
    this.waves.push({ t: this.time, steered });
    this.wake(1);
  }
  /** An electrode went in: a ring spreads through its slice and on through the layers behind it. */
  shock(L: number, n: number, sign: 1 | -1) {
    const o = (L * I + n) * 3;
    if (this.shocks.length > 1) this.shocks.shift();
    this.shocks.push({ L, x: positions[o], y: positions[o + 1], t: this.time, col: sign > 0 ? POS : NEG });
    this.wake(2.5);
  }

  // ---- camera --------------------------------------------------------------------------------------------------
  get activeLayer() { return this.layer; }
  private fit(rad: number) {
    const fh = Math.max(0.25, 1 - (this.inset.top + this.inset.bottom) / this.h), fw = Math.max(0.25, 1 - (this.inset.left + this.inset.right) / this.w);
    const t = Math.tan(FOV / 2);
    return rad / Math.sin(Math.atan(Math.min(t * fh, ((t * this.w) / this.h) * fw)));
  }
  /** Cut the brain open at a layer, or put it back together (null). */
  setLayer(L: number | null, frame = true) {
    const was = this.layer;
    this.layer = L;
    const g = this.goal;
    if (L === null) {
      g.x = WHOLE.x; g.y = WHOLE.y; g.z = WHOLE.z; g.rad = WHOLE.r; g.zoom = 1;
      if (was !== null && frame) { g.yaw = this.cam.yaw + wrap((Math.cos(this.cam.yaw) >= 0 ? 0.95 : Math.PI - 0.95) * (Math.sin(this.cam.yaw) >= 0 ? 1 : -1) - this.cam.yaw); g.pitch = 0.2; }
    } else {
      const s = slices[L];
      g.z = s.z; g.rad = s.r;
      if (was === null || g.zoom > 0.97) { g.x = s.cx; g.y = s.cy; }
      this.keepInside(g);
      if (was === null && frame) {
        // face the cut: a little off the axis, from whichever end the camera is nearer to
        const front = Math.cos(this.cam.yaw) >= 0, side = Math.sin(this.cam.yaw) >= 0 ? 1 : -1;
        g.yaw = this.cam.yaw + wrap((front ? 0.42 * side : Math.PI - 0.42 * side) - this.cam.yaw);
        g.pitch = 0.14; g.zoom = 1;
      }
    }
    this.lastInput = this.time;
    this.wake(1.5);
  }
  private keepInside(c: Cam) {
    if (this.layer === null) return;
    const s = slices[this.layer], room = s.r * Math.max(0, 1.05 - c.zoom * 0.8);
    c.x = clamp(c.x, s.cx - room, s.cx + room); c.y = clamp(c.y, s.cy - room, s.cy + room);
  }
  private touch() { this.lastInput = this.time; this.wake(); }
  grab(on: boolean) { this.dragging = on; this.touch(); }
  orbit(dx: number, dy: number) {
    const c = this.cam;
    c.yaw -= dx * 0.0085; c.pitch = clamp(c.pitch + dy * 0.0085, -1.3, 1.3);
    this.goal.yaw = c.yaw; this.goal.pitch = c.pitch;
    this.touch();
  }
  /** Zoom by a factor, keeping the point of the open slice that is under (x, y) where it is. */
  zoomAt(factor: number, x: number, y: number) {
    const c = this.cam, min = this.layer === null ? 0.5 : 0.16, z = clamp(c.zoom * factor, min, 1.5);
    if (this.layer === null) { c.zoom = this.goal.zoom = z; this.touch(); return; }
    this.matrices();
    const a = this.onPlane(x, y, slices[this.layer].z);
    c.zoom = z;
    this.matrices();
    const b = this.onPlane(x, y, slices[this.layer].z);
    if (a && b) { c.x += a[0] - b[0]; c.y += a[1] - b[1]; }
    this.keepInside(c);
    Object.assign(this.goal, { zoom: c.zoom, x: c.x, y: c.y });
    this.touch();
  }
  /** Slide the open slice so that what was under (x0, y0) is under (x1, y1). */
  pan(x0: number, y0: number, x1: number, y1: number) {
    if (this.layer === null) return;
    this.matrices();
    const z = slices[this.layer].z, a = this.onPlane(x0, y0, z), b = this.onPlane(x1, y1, z), c = this.cam;
    if (!a || !b) return;
    c.x += a[0] - b[0]; c.y += a[1] - b[1];
    this.keepInside(c);
    this.goal.x = c.x; this.goal.y = c.y;
    this.touch();
  }
  /** Back to the view this layer (or the whole brain) opens with. */
  resetView() { const L = this.layer; this.layer = L === null ? 0 : null; this.setLayer(L); }

  private matrices() {
    const c = this.cam, d = c.zoom * this.fit(c.rad), cp = Math.cos(c.pitch);
    const ex = c.x + d * Math.sin(c.yaw) * cp, ey = c.y + d * Math.sin(c.pitch), ez = c.z + d * Math.cos(c.yaw) * cp;
    this.eye = [ex, ey, ez];
    perspective(this.proj, FOV, this.w / this.h, Math.max(0.05, d - 3), d + 4);
    lookAt(this.view, ex, ey, ez, c.x, c.y, c.z);
    mul(this.tmp, this.proj, this.view);
    // shift the picture so that the target sits in the middle of the part of the canvas that is not covered
    const ox = (this.inset.left - this.inset.right) / this.w, oy = (this.inset.bottom - this.inset.top) / this.h;
    const t = this.tmp, o = this.vp;
    for (let i = 0; i < 4; i++) { const w = t[i * 4 + 3]; o[i * 4] = t[i * 4] + ox * w; o[i * 4 + 1] = t[i * 4 + 1] + oy * w; o[i * 4 + 2] = t[i * 4 + 2]; o[i * 4 + 3] = w; }
    invert(this.ivp, this.vp);
  }

  /** World point -> CSS pixels on the canvas. */
  private toScreen(x: number, y: number, z: number): [number, number, number] {
    const m = this.vp, w = m[3] * x + m[7] * y + m[11] * z + m[15];
    return [((m[0] * x + m[4] * y + m[8] * z + m[12]) / w * 0.5 + 0.5) * this.w, (0.5 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * 0.5) * this.h, w];
  }
  /** Where the ray through a pixel meets the plane of a slice. */
  private onPlane(px: number, py: number, z: number): [number, number] | null {
    const m = this.ivp, nx = (px / this.w) * 2 - 1, ny = 1 - (py / this.h) * 2;
    const un = (nz: number) => { const w = m[3] * nx + m[7] * ny + m[11] * nz + m[15]; return [(m[0] * nx + m[4] * ny + m[8] * nz + m[12]) / w, (m[1] * nx + m[5] * ny + m[9] * nz + m[13]) / w, (m[2] * nx + m[6] * ny + m[10] * nz + m[14]) / w]; };
    const a = un(-1), b = un(1), dz = b[2] - a[2];
    if (Math.abs(dz) < 1e-9) return null;
    const t = (z - a[2]) / dz;
    return t < 0 ? null : [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  /** How many CSS pixels lie between two neighbours of a slice, where it is on screen now. */
  private pitchPx(L: number, x: number, y: number) {
    const z = slices[L].z + this.pull[L], a = this.toScreen(x, y, z), b = this.toScreen(x + slices[L].pitch, y, z), c = this.toScreen(x, y + slices[L].pitch, z);
    return Math.max(Math.hypot(b[0] - a[0], b[1] - a[1]), Math.hypot(c[0] - a[0], c[1] - a[1]));
  }

  /** Where a neuron is on screen; `shown` is 0 when its layer is cut away. */
  project(L: number, n: number): Projected {
    const o = (L * I + n) * 3, s = this.toScreen(positions[o], positions[o + 1], positions[o + 2] + this.pull[L]);
    return { x: s[0], y: s[1], depth: s[2], shown: s[2] > 0 ? this.vis[L] * (this.layer === null || this.layer === L ? 1 : 0.35) : 0 };
  }

  /**
   * The neuron under a point of the canvas. `pins` are neurons with a marker on them: they are easier to hit.
   * With a slice open only its neurons can be picked; on the whole brain the nearest slice under the point wins.
   */
  pick(px: number, py: number, pins: Hit[], reachPx: number, prefer?: (L: number, n: number) => number): Hit | null {
    this.matrices();
    let best: Hit | null = null, bestD = Math.max(14, reachPx * 1.6);
    for (const p of pins) {
      if (this.layer !== null ? p.L !== this.layer : this.vis[p.L] < 0.5) continue;
      const s = this.project(p.L, p.n), d = Math.hypot(s.x - px, s.y - py);
      if (s.depth > 0 && d < bestD) { bestD = d; best = p; }
    }
    if (best) return best;
    const order = Array.from({ length: NL }, (_, L) => L).sort((a, b) => Math.abs(slices[a].z - this.eye[2]) - Math.abs(slices[b].z - this.eye[2]));
    for (const L of this.layer !== null ? [this.layer] : order) {
      const at = this.onPlane(px, py, slices[L].z + this.pull[L]);
      if (!at) continue;
      const reach = Math.min(6, this.layer !== null ? Math.max(0.75, reachPx / Math.max(1, this.pitchPx(L, at[0], at[1]))) : 0.75);
      if (prefer && reach > 1) {
        // a fat finger on small cells: of the neurons under it, take the most interesting one, the nearest among equals
        let n = -1, top = -Infinity;
        for (const c of neuronsAround(L, at[0], at[1], reach)) { const v = prefer(L, c.n) - c.d / reach; if (v > top) { top = v; n = c.n; } }
        if (n >= 0) return { L, n };
        continue;
      }
      const n = neuronNear(L, at[0], at[1], reach);
      if (n >= 0) return { L, n };
    }
    return null;
  }

  // ---- the frame -----------------------------------------------------------------------------------------------
  private loop = (now: number) => {
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(0.05, Math.max(0, (now - this.last) / 1000));
    this.last = now;
    if (this.lost || document.hidden) return;
    this.frameNo++;
    this.settle -= dt;
    this.time += dt;
    this.owed += dt;
    // The model computes on the same GPU: while it is answering the brain is drawn at half the frame rate. So is
    // the slow sway of the whole brain; when nothing moves but the shimmer, a third of the rate will do.
    // A finger on the brain always gets every frame.
    const every = this.dragging ? 1 : this.working ? 2 : this.settle > 0 ? 1 : this.swaying ? 2 : 3;
    if (this.frameNo % every) return;
    this.update(Math.min(0.1, this.owed));
    this.owed = 0;
    this.draw();
    this.onFrame?.();
  };

  private update(dt: number) {
    const c = this.cam, g = this.goal, k = 1 - Math.exp(-dt / 0.16);
    const idle = this.time - this.lastInput;
    if (this.layer === null && !this.dragging && idle > 3.5 && !this.calm) {
      // nobody is steering: sway slowly around the three-quarter view, the angle a brain is best known from
      const side = Math.sin(g.yaw) >= 0 ? 1 : -1, want = side * (0.86 + 0.24 * Math.sin(this.time * 0.21)), step = 0.22 * dt;
      g.yaw += clamp(wrap(want - g.yaw), -step, step);
      g.pitch += clamp(0.2 + 0.05 * Math.sin(this.time * 0.33) - g.pitch, -step, step);
      c.yaw = g.yaw; c.pitch = g.pitch; // slow enough to need no easing (and no more than 30 frames a second)
      this.swaying = true;
    } else this.swaying = false;
    c.yaw += (g.yaw - c.yaw) * k; c.pitch += (g.pitch - c.pitch) * k; c.zoom += (g.zoom - c.zoom) * k;
    c.x += (g.x - c.x) * k; c.y += (g.y - c.y) * k; c.z += (g.z - c.z) * k; c.rad += (g.rad - c.rad) * k;
    if (Math.abs(g.yaw - c.yaw) + Math.abs(g.pitch - c.pitch) + Math.abs(g.zoom - c.zoom) + Math.abs(g.z - c.z) + Math.abs(g.x - c.x) + Math.abs(g.y - c.y) > 2e-4) this.settle = Math.max(this.settle, 0.3);

    // which way the cut faces: +1 the camera looks at the front of the brain, -1 at the back, 0 edge-on
    const facing = clamp(Math.cos(c.yaw) * Math.cos(c.pitch) * 5, -1, 1), A = this.layer, kl = 1 - Math.exp(-dt / 0.11);
    let moving = 0;
    for (let L = 0; L < NL; L++) {
      let vis = 1, bright = 0.52, glass = 0.5, pull = 0;
      if (A !== null) {
        const behind = (slices[A].z - slices[L].z) * facing; // > 0: further from the camera than the cut
        if (L === A) { bright = 0.74; glass = 0.93; }
        else if (behind < 0) { vis = (1 - Math.abs(facing)) * 0.45; bright = 0.25; glass = 0; pull = Math.sign(facing) * 0.22 * Math.abs(facing); }
        else { bright = 0.3 * Math.pow(0.93, Math.abs(L - A) - 1) + 0.05; glass = 0.34; }
      }
      moving += Math.abs(vis - this.vis[L]) + Math.abs(pull - this.pull[L]) + Math.abs(bright - this.bright[L]);
      this.vis[L] += (vis - this.vis[L]) * kl; this.bright[L] += (bright - this.bright[L]) * kl;
      this.glass[L] += (glass - this.glass[L]) * kl; this.pull[L] += (pull - this.pull[L]) * kl;
    }
    if (moving > 1e-3) this.settle = Math.max(this.settle, 0.3);
    this.cut += ((A === null ? 0 : 1) - this.cut) * kl;
    this.waves = this.waves.filter((w) => this.time - w.t < 0.7);
    this.shocks = this.shocks.filter((s) => this.time - s.t < 2.9);
    const speaking = this.waves.length ? 1 : 0;
    this.speaking += (speaking - this.speaking) * (1 - Math.exp(-dt / 0.25));
    if (this.waves.length || this.shocks.length || this.power === "loading") this.settle = Math.max(this.settle, 0.4);
  }

  private draw() {
    const gl = this.gl, W = this.canvas.width, H = this.canvas.height, t = this.time;
    this.matrices();
    this.ensureTargets(W, H);
    const T = this.targets;
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.length > 2 ? T[0].fb : null);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.bg.p);
    gl.bindVertexArray(this.emptyVAO);
    gl.uniform2f(this.bg.u.uRes, W, H);
    gl.uniform2f(this.bg.u.uMid, W / 2 + ((this.inset.left - this.inset.right) / 2) * this.dpr, H / 2 + ((this.inset.bottom - this.inset.top) / 2) * this.dpr);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.BLEND);

    const P = this.point, G = this.plate;
    gl.useProgram(P.p);
    gl.uniformMatrix4fv(P.u.uVP, false, this.vp);
    gl.uniform1f(P.u.uPx, (H / 2) / Math.tan(FOV / 2));
    gl.uniform1f(P.u.uTime, t);
    gl.uniform1f(P.u.uMin, 1.15 * this.dpr);
    gl.uniform1f(P.u.uTwinkle, this.calm ? 0 : (this.power === "awake" ? 0.8 : 0.45) + this.speaking * 0.7);
    gl.useProgram(G.p);
    gl.uniformMatrix4fv(G.u.uVP, false, this.vp);

    // the first layer with an electrode in it: a wave changes colour there
    let first = NL, up = 0, down = 0;
    for (const e of this.electrodes) { first = Math.min(first, e.L); if (e.sign > 0) up++; else down++; }
    const tint = up && down ? [0.95, 0.6, 0.75] : down ? NEG : POS;
    const shock = new Float32Array(8), shockCol = new Float32Array(6);
    const breath = this.power === "asleep" ? 0.82 : 1;

    const front = this.eye[2] > this.cam.z; // camera in front of the cut: draw from the back of the brain forwards
    for (let i = 0; i < NL; i++) {
      const L = front ? NL - 1 - i : i, vis = this.vis[L];
      if (vis < 0.01) continue;
      const s = slices[L];
      // a slice seen edge-on packs all of its light into a line: dim it by how much of its face is showing
      const vx = s.cx - this.eye[0], vy = s.cy - this.eye[1], vz = s.z + this.pull[L] - this.eye[2];
      const face = clamp(Math.abs(vz) / Math.hypot(vx, vy, vz), 0.07, 1) / 0.62;
      let bright = this.bright[L] * face * breath * (1 + 0.07 * Math.sin(t * 0.9 - L * 0.35));
      let amt = 0, steered = false;
      for (const w of this.waves) {
        const at = (t - w.t) * 44 - 2, d = L - at, b = Math.exp(-d * d / 1.1);
        bright += b * 0.4 * face;
        if (w.steered && L >= first) { amt = Math.max(amt, b); steered = true; }
      }
      if (this.power === "loading") {
        const on = (L + 0.5) / NL <= this.progress, at = (t * 16) % (NL + 10) - 3, d = L - at;
        bright = bright * (on ? 1 : 0.4) + Math.exp(-d * d / 3) * 0.6 * face;
      }
      shock.fill(0);
      this.shocks.forEach((sh, j) => {
        if (L < sh.L) return;
        const age = (t - sh.t - (L - sh.L) * 0.06) / 1.3;
        if (age < 0 || age > 1) return;
        shock.set([sh.x, sh.y, age * 1.75, (1 - age) * (1 - age) * Math.pow(0.9, L - sh.L) * 1.5 * (this.layer === null || this.layer === L ? 1 : 0.5)], j * 4);
        shockCol.set(sh.col, j * 3);
      });
      if (this.glass[L] > 0.01) {
        const a = this.glass[L] * vis, lit = L === this.layer ? 0.05 : 0.022;
        gl.useProgram(G.p);
        gl.bindVertexArray(this.plateVAO);
        gl.uniform1f(G.u.uZ, this.pull[L]);
        gl.uniform4f(G.u.uColor, lit * 0.75 * a, lit * a, lit * 1.35 * a, a);
        gl.drawArrays(gl.TRIANGLES, plateRange[L].start, plateRange[L].count);
      }
      gl.useProgram(P.p);
      gl.bindVertexArray(this.pointVAO);
      gl.uniform1f(P.u.uPitch, s.pitch);
      gl.uniform1f(P.u.uZ, this.pull[L]);
      gl.uniform1f(P.u.uBright, bright * vis);
      gl.uniform3f(P.u.uTint, steered ? tint[0] : ACC[0], steered ? tint[1] : ACC[1], steered ? tint[2] : ACC[2]);
      gl.uniform1f(P.u.uTintAmt, Math.min(0.62, amt * 0.75) * vis);
      gl.uniform4fv(P.u.uShock, shock);
      gl.uniform3fv(P.u.uShockCol, shockCol);
      gl.drawArrays(gl.POINTS, L * I, I);
    }

    // electrodes: a needle from above, a glow at its tip
    const S = this.sprite;
    const spriteAt = (L: number, n: number, sizePx: number, col: number[], alpha: number, kind: number) => {
      const o = (L * I + n) * 3;
      gl.useProgram(S.p);
      gl.bindVertexArray(this.emptyVAO);
      gl.uniformMatrix4fv(S.u.uVP, false, this.vp);
      gl.uniform3f(S.u.uCenter, positions[o], positions[o + 1], positions[o + 2] + this.pull[L]);
      gl.uniform1f(S.u.uSize, Math.min(480, sizePx * this.dpr));
      gl.uniform4f(S.u.uColor, col[0], col[1], col[2], alpha);
      gl.uniform1i(S.u.uKind, kind);
      gl.drawArrays(gl.POINTS, 0, 1);
    };
    for (const e of this.electrodes) {
      const o = (e.L * I + e.n) * 3, x = positions[o], y = positions[o + 1], z = positions[o + 2] + this.pull[e.L];
      const col = e.sign > 0 ? POS : NEG, open = this.layer === null || this.layer === e.L;
      const seen = open ? 1 : 0.4 * this.vis[e.L]; // in a layer behind the cut: only its glow shows through
      const tip = this.toScreen(x, y, z), far = this.toScreen(x * 1.18, 1.22, z);
      if (tip[2] <= 0 || far[2] <= 0 || seen < 0.02) continue;
      const px = this.pitchPx(e.L, x, y), beat = 1 + 0.12 * Math.sin(t * 5.2);
      spriteAt(e.L, e.n, Math.max(46, px * 9) * (0.75 + 0.35 * Math.min(1.5, e.strength)) * beat, col, 0.55 * seen, 0);
      if (!open) continue;
      // a quad along the needle, 14 px wide
      let dx = far[0] - tip[0], dy = far[1] - tip[1];
      const len = Math.hypot(dx, dy) || 1;
      dx /= len; dy /= len;
      const nx = -dy * 7, ny = dx * 7, d = this.lineData;
      const put = (i: number, sx: number, sy: number, u: number, v: number) => { d[i * 4] = (sx / this.w) * 2 - 1; d[i * 4 + 1] = 1 - (sy / this.h) * 2; d[i * 4 + 2] = u; d[i * 4 + 3] = v; };
      put(0, tip[0] + nx, tip[1] + ny, 0, 1); put(1, tip[0] - nx, tip[1] - ny, 0, -1); put(2, far[0] + nx, far[1] + ny, 1, 1);
      put(3, far[0] + nx, far[1] + ny, 1, 1); put(4, tip[0] - nx, tip[1] - ny, 0, -1); put(5, far[0] - nx, far[1] - ny, 1, -1);
      gl.useProgram(this.line.p);
      gl.bindVertexArray(this.lineVAO);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, d);
      gl.uniform4f(this.line.u.uColor, col[0], col[1], col[2], 0.95 * seen);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      spriteAt(e.L, e.n, Math.max(13, px * 1.5), [1, 1, 1], 0.95 * seen, 0);
    }
    const ring = (hit: Hit | null, alpha: number, grow: number) => {
      if (!hit || this.vis[hit.L] < 0.3) return;
      const o = (hit.L * I + hit.n) * 3;
      spriteAt(hit.L, hit.n, Math.max(16, this.pitchPx(hit.L, positions[o], positions[o + 1]) * 2.3) * grow, [1, 1, 1], alpha, 1);
    };
    ring(this.selected, 0.95, 1 + 0.1 * Math.sin(t * 4));
    if (this.hover && (this.hover.L !== this.selected?.L || this.hover.n !== this.selected?.n)) ring(this.hover, 0.6, 1);

    // the glow: bright parts, blurred on the way down the chain and added up on the way back
    if (T.length > 2) {
      gl.bindVertexArray(this.emptyVAO);
      gl.activeTexture(gl.TEXTURE0);
      gl.disable(gl.BLEND);
      gl.useProgram(this.down.p);
      gl.uniform1i(this.down.u.uTex, 0);
      for (let i = 1; i < T.length; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, T[i].fb);
        gl.viewport(0, 0, T[i].w, T[i].h);
        gl.bindTexture(gl.TEXTURE_2D, T[i - 1].tex);
        gl.uniform2f(this.down.u.uTexel, 1 / T[i - 1].w, 1 / T[i - 1].h);
        gl.uniform1f(this.down.u.uCut, i === 1 ? 0.18 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.up.p);
      gl.uniform1i(this.up.u.uTex, 0);
      for (let i = T.length - 2; i >= 1; i--) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, T[i].fb);
        gl.viewport(0, 0, T[i].w, T[i].h);
        gl.bindTexture(gl.TEXTURE_2D, T[i + 1].tex);
        gl.uniform2f(this.up.u.uTexel, 1 / T[i + 1].w, 1 / T[i + 1].h);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      gl.disable(gl.BLEND);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W, H);
      gl.useProgram(this.mix.p);
      gl.uniform1i(this.mix.u.uTex, 0);
      gl.uniform1i(this.mix.u.uGlow, 1);
      gl.uniform1f(this.mix.u.uAmount, 0.44 - 0.16 * this.cut);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, T[1].tex);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T[0].tex);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }
    gl.bindVertexArray(null);
  }

  /** The picture as it is now, for the share image, and the square of it that holds the brain. */
  snapshot(): Photo {
    const out = document.createElement("canvas");
    out.width = this.canvas.width; out.height = this.canvas.height;
    if (!this.lost) { this.draw(); out.getContext("2d")!.drawImage(this.canvas, 0, 0); }
    const i = this.inset, fw = this.w - i.left - i.right, fh = this.h - i.top - i.bottom, size = Math.min(fw, fh) * 1.06 * this.dpr;
    const marks = this.electrodes.map((e) => { const s = this.project(e.L, e.n); return { x: s.x * this.dpr, y: s.y * this.dpr, sign: e.sign, label: "" }; });
    return { canvas: out, x: (i.left + fw / 2) * this.dpr - size / 2, y: (i.top + fh / 2) * this.dpr - size / 2, size, marks };
  }
}
