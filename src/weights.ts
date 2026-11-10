// The model's weights: 470 MB that have to get from a server into the runtime. They are kept on disk (the origin's
// private file system), written as they arrive and read back one tensor at a time, so they are never all in memory
// at once. The library's own way holds the whole file in memory while the runtime builds the model from it, and that
// is when a phone's browser kills the tab.

/** The note next to the file that says the download is complete, and of what. The page looks for it: no button then. */
export const WEIGHTS_NOTE = "weights.json";
const FILE = "weights";

export interface Weights {
  /** what the runtime is given as the file */
  bytes: Uint8Array;
  /** to be called when the runtime has read what it needs */
  close(): void;
}

/**
 * A Uint8Array as far as the runtime can tell. It asks a weights file for two things only, its length and ranges of
 * bytes (`subarray`), a tensor at a time: each range is read from disk when it is asked for.
 */
function onDisk(file: FileSystemSyncAccessHandle, size: number): Uint8Array {
  const bytes = new Uint8Array(0);
  Object.defineProperties(bytes, {
    byteLength: { value: size },
    subarray: {
      value: (begin: number, end: number) => {
        const part = new Uint8Array(end - begin);
        for (let got = 0; got < part.length; ) {
          const n = file.read(part.subarray(got), { at: begin + got });
          if (!n) throw new Error("the weights file on disk is shorter than it should be");
          got += n;
        }
        return part;
      },
    },
  });
  return bytes;
}

/** Opens the file, which one tab at a time can do. A tab that was closed a moment ago may still be letting go of it. */
async function openFile(): Promise<{ file: FileSystemSyncAccessHandle; dir: FileSystemDirectoryHandle } | null> {
  try {
    const dir = await navigator.storage.getDirectory();
    const handle = await dir.getFileHandle(FILE, { create: true });
    for (let tries = 0; ; tries++) {
      try {
        return { file: await handle.createSyncAccessHandle(), dir };
      } catch (err) {
        if (tries === 5) throw err;
        await new Promise((r) => setTimeout(r, 500));
      }
    }
  } catch {
    return null; // no disk for us here: a private window, or another tab that is loading the model right now
  }
}

/**
 * The weights at `url`, for the runtime. Downloaded once. Where there is no disk to keep them on (a private window,
 * a full disk, another tab with the file open) they are downloaded into memory instead, every time.
 */
export async function openWeights(url: string, get: (url: string) => Promise<Response>, progress: (loaded: number, total: number) => void, disk = true): Promise<Weights> {
  // Until the weights were kept on disk they were in the library's cache: 470 MB that nothing reads any more.
  caches.open("transformers-cache").then(async (c) => { for (const k of await c.keys()) if (k.url.endsWith("_data")) await c.delete(k); }).catch(() => {});
  const on = disk ? await openFile() : null;
  if (on) {
    const note = await on.dir.getFileHandle(WEIGHTS_NOTE).then((h) => h.getFile()).then((f) => f.text()).then(JSON.parse).catch(() => null);
    if (note?.url === url && note.size === on.file.getSize()) return { bytes: onDisk(on.file, note.size), close: () => on.file.close() };
    await on.dir.removeEntry(WEIGHTS_NOTE).catch(() => {});
  }
  try {
    const res = await get(url);
    if (!res.ok || !res.body) throw new Error(`The model's weights could not be downloaded (${res.status}): ${url}`);
    const total = Number(res.headers.get("content-length")) || 0;
    if (!on && !total) return { bytes: new Uint8Array(await res.arrayBuffer()), close: () => {} };
    const memory = on ? null : new Uint8Array(total);
    on?.file.truncate(0);
    let loaded = 0, shown = -1;
    for (const reader = res.body.getReader(); ; ) {
      const { done, value } = await reader.read();
      if (done) break;
      if (on) on.file.write(value, { at: loaded });
      else memory!.set(value, loaded);
      loaded += value.length;
      const mb = Math.floor(loaded / 1e6); // a message per megabyte, not per packet
      if (mb !== shown) { shown = mb; progress(loaded, total || loaded); }
    }
    if (!on) return { bytes: memory!, close: () => {} };
    on.file.flush();
    const note = await (await on.dir.getFileHandle(WEIGHTS_NOTE, { create: true })).createSyncAccessHandle();
    note.truncate(0);
    note.write(new TextEncoder().encode(JSON.stringify({ url, size: loaded })), { at: 0 });
    note.close();
    return { bytes: onDisk(on.file, loaded), close: () => on.file.close() };
  } catch (err: any) {
    on?.file.close();
    if (on && err?.name === "QuotaExceededError") return openWeights(url, get, progress, false); // no room on disk
    throw err;
  }
}
