#!/usr/bin/env python3
r"""Makes public/ort/<version>/ort-wasm-simd-threaded.asyncify.wasm from the stock file of onnxruntime-web.

Safari's optimizing wasm compiler (OMG) needs gigabytes for some of the runtime's mid-size functions once they have
run for a while, and an iPhone kills the tab for it (WebKit bug 304810, onnxruntime issue 26827). JavaScriptCore only
sends a function to OMG if its body is at most Options::maximumOMGCandidateCost = 100000 bytes. So every body of
MIN..100000 bytes gets dead code put in after its locals, up to TARGET bytes:
    block  br 0  nop*N  end        (the nops are never reached; one taken branch per call)
Nothing else in the file changes, and the .mjs next to it is the stock one. Those functions stay on Safari's baseline
compiler.

    V=$(node -p "require('./node_modules/onnxruntime-web/package.json').version")
    mkdir -p public/ort/$V && cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs public/ort/$V/
    python3 scripts/pad-ort-wasm.py node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm \
        public/ort/$V/ort-wasm-simd-threaded.asyncify.wasm [--min 20000] [--target 101000]

and ORT_BUILD in src/worker.ts is set to $V.
"""
import sys

OMG_CAP = 100000

def leb(b, p):
    r = 0; s = 0
    while True:
        x = b[p]; p += 1
        r |= (x & 0x7f) << s; s += 7
        if not x & 0x80: return r, p

def enc(n):
    out = bytearray()
    while True:
        x = n & 0x7f; n >>= 7
        if n: out.append(x | 0x80)
        else: out.append(x); return bytes(out)

VALTYPES = {0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x70, 0x6f}

def pad_body(body, target):
    n, p = leb(body, 0)                      # locals: vec(count, valtype)
    for _ in range(n):
        _, p = leb(body, p)
        assert body[p] in VALTYPES, hex(body[p]); p += 1
    assert body[-1] == 0x0b
    fill = target - len(body) - 5            # 5 = block(2) + br 0(2) + end(1)
    assert fill >= 0
    return body[:p] + b'\x02\x40\x0c\x00' + b'\x01' * fill + b'\x0b' + body[p:]

def main():
    a = sys.argv[1:]
    src, dst = a[0], a[1]
    lo = int(a[a.index('--min') + 1]) if '--min' in a else 20000
    target = int(a[a.index('--target') + 1]) if '--target' in a else 101000
    assert target > OMG_CAP
    b = open(src, 'rb').read()
    assert b[:8] == b'\0asm\x01\0\0\0'
    out = bytearray(b[:8]); p = 8; padded = []; nimp = 0
    while p < len(b):
        sid = b[p]; size, q = leb(b, p + 1); end = q + size
        if sid == 2:                         # count imported functions (for reporting indices only)
            n, r = leb(b, q)
            for _ in range(n):
                l, r = leb(b, r); r += l; l, r = leb(b, r); r += l
                k = b[r]; r += 1
                if k == 0: _, r = leb(b, r); nimp += 1
                elif k == 1:
                    r += 1; fl, r = leb(b, r); _, r = leb(b, r)
                    if fl & 1: _, r = leb(b, r)
                elif k == 2:
                    fl, r = leb(b, r); _, r = leb(b, r)
                    if fl & 1: _, r = leb(b, r)
                elif k == 3: r += 2
                elif k == 4: r += 1; _, r = leb(b, r)
        if sid != 10:
            out += b[p:end]; p = end; continue
        n, r = leb(b, q)
        sec = bytearray(enc(n))
        for i in range(n):
            sz, r = leb(b, r); body = b[r:r + sz]; r += sz
            if lo <= sz <= OMG_CAP:
                body = pad_body(body, target); padded.append((i + nimp, sz))
            sec += enc(len(body)) + body
        assert r == end
        out += bytes([10]) + enc(len(sec)) + sec; p = end
    open(dst, 'wb').write(out)
    print(f'{src} -> {dst}: padded {len(padded)} bodies in [{lo}, {OMG_CAP}] to {target} bytes; file {len(b)} -> {len(out)} bytes')
    return padded

if __name__ == '__main__':
    main()
