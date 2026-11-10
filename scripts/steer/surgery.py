"""ONNX graph surgery: add a runtime neuron-steering input to an exported Qwen3.5 decoder.

Adds one graph input  steer: float32[num_layers, intermediate]  and, for every layer L, rewires

    down_proj(A)   ->   down_proj(A + mask * cast(steer[L]))

where A is the post-SwiGLU MLP activation ([B, T, intermediate]) and mask is 1 for every token
except absolute position 0 (the attention sink), matching the Lyra / Neuron Persona Library recipe:
"add a signed constant to one neuron at every position except the first".

The value to put at steer[L, n] for a recipe (L, n, dose) is
    c = dose * median_resid_norm_after_L / ||W_down[:, n]||
computed outside (see calib.py); the graph just adds whatever it is given.

Usage: python surgery.py in.onnx out.onnx [--optional]
  --optional  also add a zeros initializer named `steer`, making the input optional
              (ORT treats it as an overridable initializer; feeds without it run unsteered).
External data references are kept (same directory, load_external_data=False).
"""
import argparse, json, os, re, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOWN_RE = re.compile(r"^/model/layers\.(\d+)/mlp/down_proj/")


def find_down_nodes(g):
    """layer -> node whose input[0] is the MLP activation feeding down_proj."""
    out = {}
    for n in g.node:
        m = DOWN_RE.match(n.name or "")
        if m and n.op_type in ("MatMul", "MatMulNBits", "Gemm"):
            L = int(m.group(1))
            assert L not in out, f"two down_proj matmuls for layer {L}"
            out[L] = n
    return out


def act_dtype(g, node):
    """dtype of the activation tensor = dtype of the weight/scales input of the matmul."""
    inits = {i.name: i for i in g.initializer}
    if node.op_type == "MatMulNBits":
        return inits[node.input[2]].data_type  # scales
    w = inits.get(node.input[1])
    if w is not None:
        return w.data_type
    # weight might be produced by a node (e.g. Cast); fall back to value_info
    for vi in list(g.value_info) + list(g.input):
        if vi.name == node.input[0]:
            return vi.type.tensor_type.elem_type
    raise RuntimeError(f"cannot infer activation dtype for {node.name}")


def surgery(src, dst, optional=False, data_name=None):
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    if data_name:  # rewrite external-data locations so the file can be renamed (transformers.js derives the data name from the model name)
        n_rewritten = 0
        for t in g.initializer:
            for kv in t.external_data:
                if kv.key == "location":
                    kv.value = data_name; n_rewritten += 1
        print(f"rewrote {n_rewritten} external data locations -> {data_name}")
    assert not any(i.name == "steer" for i in g.input), "already has steer input"
    down = find_down_nodes(g)
    n_layers = len(down)
    assert n_layers > 0 and sorted(down) == list(range(n_layers)), sorted(down)
    dt = act_dtype(g, down[0])
    assert all(act_dtype(g, down[L]) == dt for L in down)
    inter = None
    for L, node in down.items():
        if node.op_type == "MatMulNBits":
            K = [a.i for a in node.attribute if a.name == "K"][0]
        else:
            w = {i.name: i for i in g.initializer}[node.input[1]]
            K = w.dims[0]
        inter = inter or K
        assert K == inter, (L, K, inter)
    print(f"{n_layers} layers, intermediate={inter}, activation dtype={TensorProto.DataType.Name(dt)}")

    # --- new input ------------------------------------------------------------------------
    g.input.append(helper.make_tensor_value_info("steer", TensorProto.FLOAT, [n_layers, inter]))
    if optional:
        g.initializer.append(numpy_helper.from_array(np.zeros((n_layers, inter), np.float32), "steer"))

    consts = [
        numpy_helper.from_array(np.array(1, np.int64), "steer/one"),
        numpy_helper.from_array(np.array(0, np.int64), "steer/zero"),
        numpy_helper.from_array(np.array([0, 2], np.int64), "steer/axes_0_2"),
    ]
    g.initializer.extend(consts)

    # --- mask over positions: 1 except absolute position 0 -> [1, T, 1] -----------------
    # T = Shape(input_ids)[1]; total = Shape(attention_mask)[1]; pos = Range(total-T, total)
    pre = [
        helper.make_node("Shape", ["input_ids"], ["steer/ids_shape"], name="steer/Shape_ids"),
        helper.make_node("Gather", ["steer/ids_shape", "steer/one"], ["steer/T"], axis=0, name="steer/T"),
        helper.make_node("Shape", ["attention_mask"], ["steer/am_shape"], name="steer/Shape_am"),
        helper.make_node("Gather", ["steer/am_shape", "steer/one"], ["steer/total"], axis=0, name="steer/total"),
        helper.make_node("Sub", ["steer/total", "steer/T"], ["steer/start"], name="steer/start"),
        helper.make_node("Range", ["steer/start", "steer/total", "steer/one"], ["steer/pos"], name="steer/Range"),
        helper.make_node("Equal", ["steer/pos", "steer/zero"], ["steer/is0"], name="steer/Equal"),
        helper.make_node("Not", ["steer/is0"], ["steer/not0"], name="steer/Not"),
        helper.make_node("Cast", ["steer/not0"], ["steer/mask_flat"], to=dt, name="steer/Cast_mask"),
        helper.make_node("Unsqueeze", ["steer/mask_flat", "steer/axes_0_2"], ["steer/mask"], name="steer/Unsqueeze"),
        helper.make_node("Cast", ["steer"], ["steer/cast"], to=dt, name="steer/Cast_steer"),
    ]

    # --- per layer ------------------------------------------------------------------------
    per_layer = []
    for L in range(n_layers):
        node = down[L]
        A = node.input[0]
        idx = numpy_helper.from_array(np.array(L, np.int64), f"steer/L{L}")
        g.initializer.append(idx)
        per_layer += [
            helper.make_node("Gather", ["steer/cast", f"steer/L{L}"], [f"steer/vec{L}"], axis=0, name=f"steer/Gather{L}"),
            helper.make_node("Mul", ["steer/mask", f"steer/vec{L}"], [f"steer/add{L}"], name=f"steer/Mul{L}"),
            helper.make_node("Add", [A, f"steer/add{L}"], [f"{A}_steered"], name=f"steer/Add{L}"),
        ]
        node.input[0] = f"{A}_steered"

    # insert: pre nodes right after the last node producing input_ids/attention_mask consumers... simplest is
    # to put them at the front (they depend only on graph inputs), per-layer nodes right before their matmul.
    new_nodes = list(pre)
    for n in g.node:
        for L in range(n_layers):
            if n is down[L]:
                new_nodes.extend(per_layer[3 * L: 3 * L + 3])
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)

    try:
        onnx.checker.check_model(m)  # needs the external data file to exist; ORT re-validates at load anyway
    except onnx.checker.ValidationError as e:
        print("checker skipped:", str(e).splitlines()[0][:120])
    onnx.save(m, dst)
    meta = {"src": os.path.basename(src), "num_layers": n_layers, "intermediate": inter,
            "act_dtype": TensorProto.DataType.Name(dt), "optional_input": optional}
    with open(dst + ".steer.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", dst, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--optional", action="store_true")
    ap.add_argument("--data-name", default=None, help="rename external data location inside the graph (link/copy the file yourself)")
    a = ap.parse_args()
    surgery(a.src, a.dst, a.optional, a.data_name)
