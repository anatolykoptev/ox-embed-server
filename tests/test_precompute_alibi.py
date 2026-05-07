"""
Tests for scripts/precompute_alibi.py — ALiBi ONNX graph rewrite.

RED: these tests fail until precompute_alibi.py is implemented.
"""
import sys
import os

import numpy as np
import pytest

# Add scripts/ to path so we can import precompute_alibi as a module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

ORIG_MODEL = "/home/krolik/deploy/krolik-server/models/jina-code-v2/model_quantized.onnx"
PATCHED_MODEL = "/tmp/jina_alibi_patched.onnx"

# ---------------------------------------------------------------------------
# Unit: slopes formula
# ---------------------------------------------------------------------------


def test_slopes_formula_count():
    """get_alibi_slopes must return exactly n_heads slopes."""
    from precompute_alibi import get_alibi_slopes

    slopes = get_alibi_slopes(12)
    assert len(slopes) == 12


def test_slopes_formula_decreasing():
    """Slopes must be positive and decreasing within each geometric sub-sequence.

    Press et al. for non-power-of-2 heads interleaves two geometric sequences:
    - base: 8 slopes (power-of-2 heads)
    - extra: 4 slopes (every other from doubled-head sequence)
    Both sub-sequences are individually decreasing; the concatenated result is not
    globally monotone (extra[0] = 0.707 > base[-1] = 0.0039).  We assert:
      1. All slopes are positive.
      2. base sub-sequence is decreasing.
      3. extra sub-sequence is decreasing.
    """
    from precompute_alibi import get_alibi_slopes
    import numpy as np

    slopes = get_alibi_slopes(12)
    assert all(s > 0 for s in slopes), "All slopes must be positive"

    # For n=12: closest power-of-2 is 8 (base), extra = 4 from 16-head seq
    base = slopes[:8]
    extra = slopes[8:]
    for i in range(1, len(base)):
        assert base[i] < base[i - 1], f"base[{i}] not decreasing"
    for i in range(1, len(extra)):
        assert extra[i] < extra[i - 1], f"extra[{i}] not decreasing"


def test_slopes_match_model():
    """Slopes must match the negative values stored in jina-code-v2 ONNX model.

    The model stores slopes as negative floats in /encoder/Constant_7.
    """
    from precompute_alibi import get_alibi_slopes

    model_slopes = np.array(
        [
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.03125,
            -0.015625,
            -0.0078125,
            -0.00390625,
            -0.70710677,
            -0.35355338,
            -0.17677669,
            -0.08838835,
        ],
        dtype=np.float32,
    )
    computed = np.array(get_alibi_slopes(12), dtype=np.float32)
    # The model negates the formula slopes (stored as negative → dist-based penalty)
    np.testing.assert_allclose(
        computed,
        np.abs(model_slopes),
        rtol=1e-5,
        err_msg="ALiBi slopes formula does not match model constants",
    )


# ---------------------------------------------------------------------------
# Unit: build_alibi_const shape and values
# ---------------------------------------------------------------------------


def test_build_alibi_const_shape():
    """build_alibi_const must return [1, H, max_len, max_len]."""
    from precompute_alibi import build_alibi_const

    a = build_alibi_const(n_heads=12, max_len=64)
    assert a.shape == (1, 12, 64, 64), f"Expected (1,12,64,64) got {a.shape}"


def test_build_alibi_const_dtype():
    """build_alibi_const must return float32."""
    from precompute_alibi import build_alibi_const

    a = build_alibi_const(n_heads=12, max_len=16)
    assert a.dtype == np.float32, f"Expected float32 got {a.dtype}"


def test_build_alibi_const_diagonal_zero():
    """Diagonal (i == j) must be zero for all heads (dist=0)."""
    from precompute_alibi import build_alibi_const

    a = build_alibi_const(n_heads=12, max_len=32)
    for h in range(12):
        diag = np.diag(a[0, h])
        np.testing.assert_array_equal(diag, 0.0, err_msg=f"head {h} diagonal not zero")


def test_build_alibi_const_negative_off_diagonal():
    """Off-diagonal values must be negative (penalty on distance)."""
    from precompute_alibi import build_alibi_const

    a = build_alibi_const(n_heads=12, max_len=16)
    for i in range(16):
        for j in range(16):
            if i != j:
                assert a[0, 0, i, j] < 0.0, f"[0,0,{i},{j}]={a[0,0,i,j]} not negative"


def test_build_alibi_const_symmetric():
    """ALiBi bias matrix must be symmetric (abs distance is symmetric)."""
    from precompute_alibi import build_alibi_const

    a = build_alibi_const(n_heads=12, max_len=32)
    for h in range(12):
        np.testing.assert_array_equal(
            a[0, h],
            a[0, h].T,
            err_msg=f"head {h} bias matrix is not symmetric",
        )


def test_build_alibi_const_slice_equivalence():
    """Slicing const[:, :, :S, :S] must equal const built with max_len=S."""
    from precompute_alibi import build_alibi_const

    full = build_alibi_const(n_heads=12, max_len=64)
    small = build_alibi_const(n_heads=12, max_len=16)
    np.testing.assert_array_equal(
        full[0, :, :16, :16],
        small[0, :, :16, :16],
        err_msg="Slice of full const must equal independently computed small const",
    )


# ---------------------------------------------------------------------------
# Integration: patched model must exist after patch()
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def patched_model_path():
    """Run patch() once and return path to patched model."""
    if not os.path.exists(ORIG_MODEL):
        pytest.skip(f"Original model not found at {ORIG_MODEL}")

    from precompute_alibi import patch

    patch(ORIG_MODEL, PATCHED_MODEL, n_heads=12, max_len=512)
    assert os.path.exists(PATCHED_MODEL), "patch() did not produce output file"
    return PATCHED_MODEL


@pytest.fixture(scope="module")
def sessions(patched_model_path):
    """Load both inference sessions."""
    import onnxruntime as ort

    orig = ort.InferenceSession(ORIG_MODEL)
    new = ort.InferenceSession(patched_model_path)
    return orig, new


# ---------------------------------------------------------------------------
# Integration: embedding equivalence across seq lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [16, 64, 128, 256, 512])
def test_embedding_equivalence(sessions, seq_len):
    """Patched model must produce hidden states within cosine+L∞ tolerance.

    The model outputs last_hidden_state shape [B, S, 768].  We compare the
    mean-pooled sequence representation (shape [768]) for cosine similarity
    and use L∞ on the full token matrix for absolute error.
    """
    orig, new = sessions
    np.random.seed(42)
    feeds = {
        "input_ids": np.random.randint(100, 30000, (1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
    }
    e_o = orig.run(None, feeds)[0]  # [1, S, 768]
    e_n = new.run(None, feeds)[0]   # [1, S, 768]

    # Mean-pool across sequence for cosine similarity
    mean_o = e_o[0].mean(axis=0)  # [768]
    mean_n = e_n[0].mean(axis=0)  # [768]
    cos = np.dot(mean_o, mean_n) / (np.linalg.norm(mean_o) * np.linalg.norm(mean_n))

    # Max absolute error across all token embeddings
    err = np.max(np.abs(e_o - e_n))

    assert cos > 0.9999, f"S={seq_len}: cosine={cos:.6f} (need >0.9999)"
    assert err < 5e-4, f"S={seq_len}: max_abs_err={err:.2e} (need <5e-4)"


def test_batch_size_2(sessions):
    """Patched model must handle batch_size=2."""
    orig, new = sessions
    np.random.seed(7)
    seq_len = 64
    feeds = {
        "input_ids": np.random.randint(100, 30000, (2, seq_len), dtype=np.int64),
        "attention_mask": np.ones((2, seq_len), dtype=np.int64),
    }
    e_o = orig.run(None, feeds)[0]  # [2, S, 768]
    e_n = new.run(None, feeds)[0]   # [2, S, 768]

    for b in range(2):
        mean_o = e_o[b].mean(axis=0)
        mean_n = e_n[b].mean(axis=0)
        cos = np.dot(mean_o, mean_n) / (np.linalg.norm(mean_o) * np.linalg.norm(mean_n))
        assert cos > 0.9999, f"batch={b}: cosine={cos:.6f}"


def test_patched_model_loadable_by_ort(patched_model_path):
    """Patched model must be loadable by onnxruntime (the real validator).

    Note: onnx.checker.check_model is intentionally not used here.  The model
    relies on com.microsoft custom ops (SkipLayerNormalization, quantized
    LayerNormalization) that the ONNX standard checker does not recognise —
    the original model itself fails check_model.  ORT inference is the correct
    structural validator for this model family.
    """
    import onnxruntime as ort

    # Loading alone validates op availability and graph connectivity
    sess = ort.InferenceSession(patched_model_path)
    assert sess is not None


def test_alibi_nodes_removed(patched_model_path):
    """Patched model must not contain the runtime ALiBi compute chain."""
    import onnx

    model = onnx.load(patched_model_path)
    node_names = {n.name for n in model.graph.node}
    # These nodes should be gone after rewrite
    alibi_chain = [
        "/encoder/Range",
        "/encoder/Sub",
        "/encoder/Abs",
        "/encoder/Unsqueeze_2",
        "/encoder/Expand",
        "/encoder/Cast_1",
        "/encoder/Mul_1",
        "/encoder/Unsqueeze_3",
    ]
    for name in alibi_chain:
        assert name not in node_names, f"ALiBi node '{name}' still present in patched model"


def test_alibi_const_initializer_present(patched_model_path):
    """Patched model must contain the precomputed ALiBi const initializer."""
    import onnx

    model = onnx.load(patched_model_path)
    init_names = {i.name for i in model.graph.initializer}
    assert "alibi_precomputed_const" in init_names, (
        "alibi_precomputed_const initializer not found in patched model"
    )
