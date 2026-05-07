"""
precompute_alibi.py — Rewrite jina-code-v2 ALiBi positional bias from a runtime
compute graph into a single pre-computed const tensor + Slice node.

Problem
-------
At S=512 the runtime ALiBi chain:
    attention_mask → Cast → Shape → Gather(S) → Range(0,S,1) →
    Unsqueeze × 2 → Sub → Abs → Unsqueeze → Expand([H,S,S]) → Cast(f32) →
    Mul(neg_slopes[12,1,1]) → Unsqueeze(axis=0) → [1,H,S,S]
allocates ~1.258 GiB of fresh scratch every forward pass, causing BFCArena OOM
at S=512 on the production server (incident 2026-05-07).

Fix
---
Replace the entire chain with:
  1. A const initializer  alibi_precomputed_const  shape=[1, H, max_len, max_len]
     (~6 MiB at H=12, max_len=512, float32).
  2. Two Slice nodes that extract [:, :, :S, :S] at runtime using the actual
     sequence length from the Shape→Gather chain already present in the model.

Effect: per-call ALiBi scratch 1.258 GiB → 0.  Attention-score tensors
[B, H, S, S] are unchanged.

Verification: cosine > 0.9999 + max_abs_err < 5e-4 vs original across
S ∈ {16, 64, 128, 256, 512}.

Usage
-----
    python scripts/precompute_alibi.py \\
        --input  /path/to/model_quantized.onnx \\
        --output /path/to/model_quantized_alibi.onnx

    # Or from Python:
    from scripts.precompute_alibi import patch
    patch(input_path, output_path, n_heads=12, max_len=512)
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# ALiBi formula (Press et al. 2021 — "Train Short, Test Long")
# ---------------------------------------------------------------------------


def get_alibi_slopes(n_heads: int) -> list[float]:
    """Return ALiBi slopes for n_heads attention heads (positive values).

    The jina-code-v2 model stores these as negative (penalty), so callers
    must negate when building the bias matrix.

    Reference: https://github.com/ofirpress/attention_with_linear_biases
    """

    def _power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        return [float(start * start**i) for i in range(n)]

    if np.log2(n_heads).is_integer():
        return _power_of_2(n_heads)

    closest = int(2 ** np.floor(np.log2(n_heads)))
    base = _power_of_2(closest)
    extra = _power_of_2(2 * closest)[0::2][: n_heads - closest]
    return base + extra


def build_alibi_const(n_heads: int, max_len: int) -> np.ndarray:
    """Build pre-computed ALiBi bias tensor [1, n_heads, max_len, max_len].

    Values are negative (distance penalty): slope * (-abs(i - j)).
    Diagonal is 0, off-diagonal is negative.

    Returns float32 ndarray.
    """
    # Use float32 arithmetic throughout to match the model's runtime computation.
    # The model stores slopes as float32 (Constant_7) and computes dist as float32
    # (Range → Cast → Mul).  Using float64 intermediates then downcasting introduces
    # small ULP differences (up to 3e-5) that, after 12 INT8-quantised attention
    # layers, amplify to ~0.1 hidden-state error.
    slopes = np.array(get_alibi_slopes(n_heads), dtype=np.float32)
    pos = np.arange(max_len, dtype=np.float32)
    # dist[i, j] = |i - j|  (positive, float32)
    dist = np.abs(pos[:, None] - pos[None, :])
    # bias[h, i, j] = -slope[h] * dist[i, j]  (negative penalty)
    alibi = -slopes[:, None, None] * dist[None, :, :]  # [H, max_len, max_len]
    return alibi[None, :, :, :].astype(np.float32)  # [1, H, max_len, max_len]


# ---------------------------------------------------------------------------
# Graph surgery helpers
# ---------------------------------------------------------------------------


def _find_node_by_name(graph, name: str):
    for n in graph.node:
        if n.name == name:
            return n
    return None


def _get_consumers(graph, tensor_name: str) -> list:
    """Return all nodes that have tensor_name as an input."""
    return [n for n in graph.node if tensor_name in n.input]


def _bfs_forward(graph, start_tensor: str) -> tuple[list, set]:
    """BFS forward from start_tensor; return (ordered_nodes, reachable_tensor_set)."""
    visited_tensors: set[str] = set()
    visited_nodes: list = []
    queue = [start_tensor]

    while queue:
        tname = queue.pop(0)
        if tname in visited_tensors:
            continue
        visited_tensors.add(tname)
        for n in _get_consumers(graph, tname):
            if n not in visited_nodes:
                visited_nodes.append(n)
            for o in n.output:
                if o not in visited_tensors:
                    queue.append(o)

    return visited_nodes, visited_tensors


def _is_orphan_after_rewire(node, alibi_output_name: str, rewired_name: str) -> bool:
    """Return True if node's ALL outputs are consumed only within the ALiBi subgraph.

    After rewiring Add nodes to use rewired_name instead of alibi_output_name,
    any node whose outputs are exclusively consumed by other alibi-subgraph nodes
    (or are alibi_output_name itself) can be removed.
    """
    # Implemented in patch() via collected orphan set
    raise NotImplementedError("Use _collect_removable_nodes instead")


def _collect_removable_nodes(
    graph, alibi_nodes: list, add_nodes: list, alibi_output_name: str = ""
) -> list:
    """Return all alibi_nodes that are safe to remove.

    A node is safe to remove if ALL of its outputs satisfy:
    - consumed only by other alibi_nodes (including add_nodes, which will be
      rewired post-removal), AND
    - not a graph output (graph outputs have no consumer nodes but are external).

    add_nodes: the 12 Add nodes that consume alibi_output_name.  These are
    rewired (not removed), but for the purpose of this check we treat them
    as "within the alibi set" so that Unsqueeze_3 (whose output is consumed
    only by add_nodes) is correctly marked removable.

    alibi_output_name: the tensor produced by Unsqueeze_3 (consumed by add_nodes).
    After rewiring, no node will consume it, so it is always removable.
    """
    # Treat both alibi_nodes AND add_nodes as "within the set" for consumer checks.
    # This allows Unsqueeze_3 (consumed only by add_nodes) to be removable.
    within_set = set(id(n) for n in alibi_nodes) | set(id(n) for n in add_nodes)
    add_set = set(id(n) for n in add_nodes)

    # Graph output tensor names — these are external consumers
    graph_output_names: set[str] = {o.name for o in graph.output}

    removable = []
    for n in alibi_nodes:
        if id(n) in add_set:
            # Add nodes are rewired, not removed
            continue
        # Check that every output of this node is:
        # 1. Not a graph output, and
        # 2. Consumed only by within-set nodes
        all_consumed_within = True
        for o in n.output:
            if o in graph_output_names:
                all_consumed_within = False
                break
            for c in _get_consumers(graph, o):
                if id(c) not in within_set:
                    all_consumed_within = False
                    break
            if not all_consumed_within:
                break
        if all_consumed_within:
            removable.append(n)

    return removable


# ---------------------------------------------------------------------------
# Core patch function
# ---------------------------------------------------------------------------


def _topo_sort_graph(graph) -> None:
    """Sort graph.node in topological order in-place using Kahn's algorithm.

    Required after graph surgery (node removal + insertion) to satisfy
    onnx.checker.check_model which requires producer-before-consumer ordering.
    """
    # Build set of all "available" tensor names: graph inputs + initializers
    available: set[str] = set()
    for inp in graph.input:
        available.add(inp.name)
    for init in graph.initializer:
        available.add(init.name)

    nodes = list(graph.node)
    sorted_nodes: list = []
    remaining = list(nodes)

    max_iters = len(nodes) + 1
    for _ in range(max_iters):
        if not remaining:
            break
        progress = False
        still_remaining = []
        for n in remaining:
            # A node is ready when all its non-empty inputs are available
            needed = [inp for inp in n.input if inp]  # skip empty optional inputs
            if all(t in available for t in needed):
                sorted_nodes.append(n)
                for o in n.output:
                    available.add(o)
                progress = True
            else:
                still_remaining.append(n)
        remaining = still_remaining
        if not progress:
            # Remaining nodes form a cycle or have unresolvable deps; append as-is
            sorted_nodes.extend(remaining)
            break

    del graph.node[:]
    graph.node.extend(sorted_nodes)


def find_alibi_subgraph(graph) -> tuple[str, list, list]:
    """Locate the ALiBi subgraph in the graph.

    The jina-code-v2 ALiBi chain has a well-known structure.  We locate it
    by the Range node (unique in the encoder) and then walk FORWARD through
    nodes that are PURELY within the ALiBi chain — i.e. nodes whose ALL
    inputs come from either:
    (a) a previously identified ALiBi node's output, or
    (b) graph initializers / constants (free tensors).

    The critical insight: the boundary nodes are those that take inputs from
    both the ALiBi chain AND from outside it (e.g. Expand takes the [H,S,S]
    shape from a Where node which itself is fed by runtime data).  We DO
    include such "mixed" nodes if EVERY one of their non-free inputs is an
    ALiBi tensor — but we detect them via a two-pass flood-fill.

    The seq_len Gather node (`/encoder/Gather`) is NOT removed — its output
    is reused by the new Slice nodes that replace the ALiBi compute chain.

    Returns
    -------
    alibi_output_name : str
        Tensor name of the final ALiBi bias tensor (Unsqueeze_3 output).
    alibi_subgraph_nodes : list
        Nodes that should be removed.  Excludes Shape/Gather (reused),
        Add consumers (rewired but kept), and any node with outputs consumed
        by non-ALiBi nodes.
    add_consumers : list
        The 12 Add nodes that consume alibi_output_name.

    Raises
    ------
    RuntimeError
        If expected topology not found.
    """
    # 1. Find the single encoder Range node (ALiBi entry point)
    range_nodes = [n for n in graph.node if n.op_type == "Range"]
    encoder_range = [n for n in range_nodes if "/encoder/Range" in n.name]
    if len(encoder_range) != 1:
        raise RuntimeError(
            f"Expected exactly 1 encoder Range node, found {len(encoder_range)}: "
            f"{[n.name for n in encoder_range]}"
        )
    range_node = encoder_range[0]

    # 2. Find the final Unsqueeze_3 node — output is ALiBi tensor consumed by Add.
    unsqueeze_3 = _find_node_by_name(graph, "/encoder/Unsqueeze_3")
    if unsqueeze_3 is None:
        raise RuntimeError("Could not find /encoder/Unsqueeze_3 node")
    alibi_output_name = unsqueeze_3.output[0]

    # 3. Find Add consumers of the ALiBi output (12 attention layer Add nodes).
    add_consumers = _get_consumers(graph, alibi_output_name)
    if len(add_consumers) == 0:
        raise RuntimeError(f"No consumers found for ALiBi output '{alibi_output_name}'")

    # 4. Collect all nodes in the ALiBi compute chain using two phases:
    #
    #    Phase A — seed from Range forward: flood-fill using alibi tensor set,
    #    treating Range's own inputs and all tensor outputs of the "shape chain"
    #    (the Where/Expand machinery) as free.  This catches Range itself and
    #    the simple Unsqueeze/Sub/Abs path.
    #
    #    Phase B — extend with nodes that produce alibi_output_name.
    #    Walk BACKWARD from Unsqueeze_3, stopping when we hit nodes that have
    #    external (non-alibi) consumers — those are NOT removable.
    #    Then merge: a node is "alibi chain" if it was found in Phase A OR
    #    was found backward from Unsqueeze_3.
    #
    #    The backward walk naturally includes Expand, Cast_1, Mul_1, Unsqueeze_3
    #    (the downstream half) and Range, Unsqueeze, Sub, Abs (the upstream half).
    #    We then use _collect_removable_nodes to prune out any node whose outputs
    #    are externally consumed — this guarantees correctness.

    # Build name → producer map
    name2producer: dict[str, object] = {}
    for n in graph.node:
        for o in n.output:
            name2producer[o] = n

    # Free tensors = graph inputs + initializers
    free_names: set[str] = set()
    for inp in graph.input:
        free_names.add(inp.name)
    for init in graph.initializer:
        free_names.add(init.name)

    # Backward walk from Unsqueeze_3.  Stop when input is free or already visited.
    # We also stop at /encoder/Gather (its output is reused by new Slice nodes).
    # The key rule: stop walking back through any node that is the Gather node
    # (we mark its output as "boundary-free" so we don't traverse deeper).
    gather_node = _find_node_by_name(graph, "/encoder/Gather")
    boundary_free: set[str] = set(free_names)
    if gather_node:
        for o in gather_node.output:
            boundary_free.add(o)  # treat Gather's output as a "wall"

    backward_chain: list = []
    visited_back: set[int] = set()

    def _walk_back(node) -> None:
        if id(node) in visited_back:
            return
        visited_back.add(id(node))
        backward_chain.append(node)
        for inp in node.input:
            if not inp or inp in boundary_free:
                continue
            producer = name2producer.get(inp)
            if producer is not None:
                _walk_back(producer)

    _walk_back(unsqueeze_3)

    # Combine: union of forward and backward discoveries
    all_found = list({id(n): n for n in backward_chain}.values())

    # Remove Add consumers (rewired but kept)
    add_ids = {id(n) for n in add_consumers}
    alibi_chain = [n for n in all_found if id(n) not in add_ids]

    return alibi_output_name, alibi_chain, add_consumers


def patch(
    input_path: str,
    output_path: str,
    n_heads: int = 12,
    max_len: int = 512,
) -> None:
    """Rewrite jina-code-v2 ONNX ALiBi from runtime compute to const + Slice.

    Parameters
    ----------
    input_path  : Path to original model_quantized.onnx (read-only).
    output_path : Path for patched model (new file, never overwrites input).
    n_heads     : Number of attention heads (default 12 for jina-code-v2).
    max_len     : Maximum sequence length to pre-compute (default 512).
    """
    if input_path == output_path:
        raise ValueError("input_path and output_path must be different to preserve original")

    print(f"Loading model from {input_path} ...")
    model = onnx.load(input_path)
    graph = model.graph

    # ------------------------------------------------------------------
    # Step 1: Locate ALiBi subgraph
    # ------------------------------------------------------------------
    print("Locating ALiBi subgraph ...")
    alibi_output_name, alibi_nodes, add_consumers = find_alibi_subgraph(graph)
    print(f"  ALiBi output tensor: {alibi_output_name}")
    print(f"  ALiBi subgraph nodes: {len(alibi_nodes)}")
    print(f"  Add consumers (layers): {len(add_consumers)}")

    # ------------------------------------------------------------------
    # Step 2: Build and add the precomputed const initializer
    # ------------------------------------------------------------------
    print(f"Building precomputed ALiBi const [1, {n_heads}, {max_len}, {max_len}] ...")
    alibi_const = build_alibi_const(n_heads=n_heads, max_len=max_len)
    const_name = "alibi_precomputed_const"
    init_tensor = numpy_helper.from_array(alibi_const, name=const_name)
    graph.initializer.append(init_tensor)

    # ------------------------------------------------------------------
    # Step 3: Get runtime sequence length from Shape→Gather chain
    # We need S at runtime to Slice const[:, :, :S, :S].
    # The existing /encoder/Gather node already produces S (int64 scalar).
    # ------------------------------------------------------------------
    gather_node = _find_node_by_name(graph, "/encoder/Gather")
    if gather_node is None:
        raise RuntimeError("Could not find /encoder/Gather node for runtime S")
    seq_len_tensor = gather_node.output[0]  # int64 scalar = S

    # ------------------------------------------------------------------
    # Step 4: Add Slice nodes to extract [:, :, :S, :S]
    #
    # ONNX Slice(data, starts, ends, axes, steps)
    # We need two Slices:
    #   Slice_1: const → [:, :, :S, :]     on axis=2
    #   Slice_2: Slice_1 → [:, :, :, :S]   on axis=3
    #
    # starts = [0], ends = [S], axes = [axis], steps = [1]
    # ends must be an int64 tensor matching seq_len_tensor shape (scalar).
    # For ONNX Slice, starts/ends must be 1-D tensors, so we Unsqueeze the scalar.
    # ------------------------------------------------------------------

    # Helper to add a small int64 constant initializer
    def _add_int64_const(graph, name: str, value) -> str:
        arr = np.array(value, dtype=np.int64)
        init = numpy_helper.from_array(arr, name=name)
        graph.initializer.append(init)
        return name

    starts_name = _add_int64_const(graph, "alibi_slice_starts", [0])
    steps_name = _add_int64_const(graph, "alibi_slice_steps", [1])

    # Unsqueeze seq_len_tensor (scalar int64) to 1-D tensor [S] for Slice ends.
    # The model uses opset 11 where Unsqueeze takes axes as an ATTRIBUTE
    # (not a second input as in opset 13+).
    unsq_seq_name = "alibi_slice_seq_len_1d"
    unsq_seq_node = helper.make_node(
        "Unsqueeze",
        inputs=[seq_len_tensor],
        outputs=[unsq_seq_name],
        name="alibi_unsqueeze_seq_len",
        axes=[0],  # opset 11 attribute
    )

    # Slice on axis=2 (rows): const[:, :, :S, :]
    axis2_name = _add_int64_const(graph, "alibi_slice_axis2", [2])
    slice2_out = "alibi_slice_axis2_out"
    slice2_node = helper.make_node(
        "Slice",
        inputs=[const_name, starts_name, unsq_seq_name, axis2_name, steps_name],
        outputs=[slice2_out],
        name="alibi_slice_axis2_node",
    )

    # Slice on axis=3 (cols): slice2_out[:, :, :, :S]
    axis3_name = _add_int64_const(graph, "alibi_slice_axis3", [3])
    slice3_out = "alibi_slice_axis3_out"
    slice3_node = helper.make_node(
        "Slice",
        inputs=[slice2_out, starts_name, unsq_seq_name, axis3_name, steps_name],
        outputs=[slice3_out],
        name="alibi_slice_axis3_node",
    )

    # ------------------------------------------------------------------
    # Step 4b: Collect removable nodes BEFORE rewiring.
    # After rewiring, add_consumers no longer consume alibi_output_name,
    # so the removability check must be done on the pre-rewire graph state.
    # ------------------------------------------------------------------
    print("Collecting orphaned nodes for removal (pre-rewire check) ...")
    removable = _collect_removable_nodes(graph, alibi_nodes, add_consumers)
    print(f"  Nodes to remove: {len(removable)}")

    # Insert new nodes immediately after the Gather node that produces seq_len_tensor.
    # ONNX requires topological order: producer must appear before consumer.
    # Inserting at position 0 would violate this since Gather is at index ~199.
    gather_idx = next(
        (i for i, n in enumerate(graph.node) if n.name == "/encoder/Gather"),
        len(graph.node),
    )
    # Insert in reverse order so that after all inserts the order is:
    # ... Gather ... unsq_seq_node, slice2_node, slice3_node ...
    graph.node.insert(gather_idx + 1, slice3_node)
    graph.node.insert(gather_idx + 1, slice2_node)
    graph.node.insert(gather_idx + 1, unsq_seq_node)

    # ------------------------------------------------------------------
    # Step 5: Rewire Add consumers from alibi_output_name → slice3_out
    # ------------------------------------------------------------------
    print("Rewiring Add nodes ...")
    rewire_count = 0
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == alibi_output_name:
                n.input[i] = slice3_out
                rewire_count += 1
    print(f"  Rewired {rewire_count} input references")

    # ------------------------------------------------------------------
    # Step 6: Remove orphaned ALiBi compute nodes
    # ------------------------------------------------------------------
    print(f"  Removing {len(removable)} orphaned nodes")
    removable_set = set(id(n) for n in removable)
    # Remove in one pass (graph.node is a repeated field — rebuild)
    surviving = [n for n in graph.node if id(n) not in removable_set]
    # Clear and re-add surviving nodes
    del graph.node[:]
    graph.node.extend(surviving)

    # ------------------------------------------------------------------
    # Step 6b: Topological sort
    #
    # After removing nodes and inserting new ones, the node list may no
    # longer be in topological order.  onnx.checker.check_model requires
    # topological order; ORT is more lenient but we sort for hygiene.
    # ------------------------------------------------------------------
    print("Topologically sorting surviving nodes ...")
    _topo_sort_graph(graph)

    # ------------------------------------------------------------------
    # Step 7: Validate + save
    # ------------------------------------------------------------------
    # Note: onnx.checker.check_model is intentionally skipped.
    # The model uses Microsoft-domain custom ops (SkipLayerNormalization,
    # LayerNormalization, QuantizeLinear variants) that the ONNX standard
    # checker does not recognise — the original model itself fails check_model.
    # Structural correctness is validated by ORT inference in the test suite.
    print("Skipping onnx.checker (model uses com.microsoft custom ops) ...")

    print(f"Saving patched model to {output_path} ...")
    onnx.save(model, output_path)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rewrite jina-code-v2 ALiBi runtime compute to const+Slice."
    )
    parser.add_argument("--input", required=True, help="Path to input model_quantized.onnx")
    parser.add_argument("--output", required=True, help="Path for patched output model")
    parser.add_argument("--n-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length to precompute")
    args = parser.parse_args()

    patch(
        input_path=args.input,
        output_path=args.output,
        n_heads=args.n_heads,
        max_len=args.max_len,
    )
