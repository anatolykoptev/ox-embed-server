# Known Issues & Bugs

## BUG-001: ort crate 1000x slowdown on ARM for BERT models with token_type_ids

**Status:** UNRESOLVED (workaround deployed)
**Severity:** Critical
**Date:** 2026-03-06
**Component:** pykeio/ort v2.0.0-rc.12, ONNX Runtime 1.24.x, ARM Neoverse-N1

### Problem

The Rust `ort` crate produces 1000x slower inference for certain BERT-family ONNX models on ARM (Neoverse-N1 / Oracle Cloud A1). Specifically:

| Model | ort (Rust) | Python onnxruntime | Go onnxruntime_go |
|-------|-----------|-------------------|-------------------|
| multilingual-e5-large (XLM-RoBERTa) | 100ms | ~80ms | ~200ms |
| jina-code-v2 (BERT with token_type_ids) | **30,000ms** | **60ms** | ~150ms |

Same ONNX files, same ORT version (1.24.3), same machine. The slowdown is specific to:
- ARM architecture (not reproduced on x86)
- Models requiring `token_type_ids` input (BERT-family)
- `ort` crate specifically (Python and Go bindings work fine)

### What was tested

1. **ORT versions**: 1.24.1 (even slower: 43s jina, 2.2s e5), 1.24.3 (30s jina)
2. **token_type_ids=false**: Model crashes ("Missing Input: token_type_ids")
3. **Internalized token_type_ids in ONNX graph**: Still slow (10-19s)
4. **fp32 model (no quantization)**: Progressively slower (25s->153s, memory thrashing)
5. **jina-only (no e5 loaded)**: Still 24-33s with only 604MiB memory
6. **Thread configs** (auto, 4 intra + 1 inter): No improvement
7. **ONNX graph optimization** (Gelu/LayerNorm/SkipLayerNorm fusions): No improvement for ort

### Workaround

Deployed a Python sidecar (`embed-jina`) for jina-code-v2 using native `onnxruntime` package. The Rust `embed-server` handles only e5-large (which works fine in ort).

### Root cause hypothesis

Likely a bug in ort crate's tensor memory layout or session execution path for models with 3 named inputs on ARM. The Python and Go ORT bindings use different FFI approaches that don't trigger this issue.

### To monitor

- pykeio/ort releases: check if fixed in future versions
- Test with ort v3.x when available
- Consider filing upstream issue at https://github.com/pykeio/ort/issues

---

## BUG-002: ONNX models require graph optimization for ARM

**Status:** RESOLVED
**Severity:** High
**Date:** 2026-03-06

### Problem

Unoptimized ONNX models run ~50x slower on ARM without AVX instructions. The ONNX Runtime graph optimizer fuses operations (Gelu, LayerNormalization, SkipLayerNormalization) that are critical for ARM NEON performance.

### Solution

Pre-optimize models using `onnxruntime.transformers.optimizer`:

```python
from onnxruntime.transformers.optimizer import optimize_model

# jina-code-v2
m = optimize_model('model_quantized.onnx', model_type='bert',
                   num_heads=12, hidden_size=768, opt_level=0)
m.save_model_to_file('model_optimized.onnx')

# multilingual-e5-large
m = optimize_model('model_quantized.onnx', model_type='bert',
                   num_heads=16, hidden_size=1024, opt_level=0)
m.save_model_to_file('model_optimized.onnx')
```

Note: `opt_level=0` skips torch-dependent ORT optimization. The graph fusions alone provide the speedup. The `onnx` pip package is required (not just `onnxruntime`).

### Fusions applied (jina-code-v2)

- Gelu: 12
- LayerNormalization: 24
- SkipLayerNormalization: 37

---

## BUG-003: Python wsgiref.simple_server doesn't flush responses with ThreadingMixIn

**Status:** RESOLVED
**Severity:** High
**Date:** 2026-03-06

### Problem

`wsgiref.simple_server.WSGIServer` combined with `socketserver.ThreadingMixIn` accepts HTTP requests and processes them, but never sends the response back to the client. The client times out despite the server completing the work.

This happens because:
1. `WSGIServer` defaults to HTTP/1.0
2. `ThreadingMixIn` spawns threads that don't properly close connections
3. Without explicit `Content-Length` headers, HTTP/1.0 relies on connection close to signal end of response
4. The threading layer interferes with connection lifecycle

### Solution

Replace `wsgiref` WSGI app with `http.server.BaseHTTPRequestHandler`:

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # Important: use HTTP/1.1

    def do_POST(self):
        # Always set Content-Length explicitly
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
```

Key requirements:
- Use `HTTP/1.1` protocol version
- Always send explicit `Content-Length` header
- Use `BaseHTTPRequestHandler` (not WSGI)
