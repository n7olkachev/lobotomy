import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

// Dev: model files are served by ~/ai/neuron-onnx/js/serve.mjs on :8765 and proxied under /models (same origin, no CORS).
export default defineConfig({
  plugins: [preact()],
  server: {
    headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" },
    proxy: { "/models": { target: "http://127.0.0.1:8765", changeOrigin: false } },
  },
  preview: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
  worker: { format: "es" },
  build: { target: "es2022" },
});
