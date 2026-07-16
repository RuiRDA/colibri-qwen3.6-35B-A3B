# Qwen3.6 implementation notes

This document describes the Colibri CPU implementation for the text portion of
`Qwen/Qwen3.6-35B-A3B-FP8`. For the quick start and measured results, see the
[root README](README.md).

## Scope

Implemented:

- The checkpoint's nested `text_config` (`qwen3_5_moe_text`).
- 40 decoder layers.
- The 3:1 Gated DeltaNet/full-attention layer pattern.
- 256 routed experts, top 8, and the gated shared expert.
- Centered RMSNorm, partial split-half RoPE, and gated query output.
- FP32 DeltaNet recurrent state and an FP32 full-attention KV cache.
- Row-wise Q4 dense/expert matrices and Q8 embedding/LM-head matrices.
- Per-layer LRU expert streaming.
- Chat, one-shot generation, batch protocol, and OpenAI-compatible serving.

Not implemented:

- The vision encoder.
- Multimodal input.
- The MTP draft head or speculative decoding.
- GPU backends.

The omissions keep the first version focused on faithful text generation within
a 16 GB CPU-only machine.

## Runtime design

[`c/qwen36.c`](c/qwen36.c) is a dedicated sequential inference engine. Dense
weights, routers, recurrent state, and context state stay resident. Routed
expert matrices are loaded from the converted safetensors files only when the
router selects them.

Each decoder layer owns an LRU of expert slots. `--cap N` sets the maximum
slots per layer; it is not a global count. A hit reuses the already decoded
packed Q4 matrices. A miss evicts the least recently used expert for that
layer, reads the selected expert, and updates the cache counters.

The engine has AVX2 paths for FP32, Q8, and packed Q4 matrix-vector products and
uses OpenMP across output rows. It intentionally remains a small C runtime
rather than depending on PyTorch or a general inference framework.

## Quantized checkpoint layout

[`c/tools/convert_qwen36.py`](c/tools/convert_qwen36.py) directly reads the
official safetensors format.

| Tensor class | Stored format |
|---|---|
| RMSNorm and routing weights | FP32 |
| Token embedding and LM head | Row-wise symmetric int8 + FP32 row scale |
| Dense projections | Packed row-wise symmetric int4 + FP32 row scale |
| Routed expert projections | Packed row-wise symmetric int4 + FP32 row scale |
| Vision and MTP tensors | Skipped |

Packed int4 values use two signed quantized values per byte. Rows are converted
in bounded chunks so a large tensor is never expanded into one full FP32 copy.

The conversion loop processes one official shard at a time. Downloads support
HTTP range resume, output shards are written through a temporary file and
renamed only after completion, and validated completed shards are skipped.
The temporary FP8 shard is removed after its Q4 destination has been written.

## Build and conversion on Windows

The tested system used portable w64devkit at
`D:\desktop\colibri_test\.tools\w64devkit`. The launcher detects the
matching Make/GCC pair in that parent-level `.tools` directory.

```powershell
cd D:\desktop\colibri_test\colibri-qwen36
python c\coli build

python c\coli convert `
  --model C:\Models\qwen36-colibri-q4 `
  --workdir D:\colibri-convert-cache
```

NumPy is the converter's only required Python package. Conversion is resumable
by running the same command again.

The completed model on this machine occupied about 18.05 GB decimal
(16.81 GiB). At least 35 GB free on the destination drive is recommended before
starting. Keeping conversion scratch on D: and the final model on the C: SSD
worked well.

## Run

```powershell
$env:OMP_NUM_THREADS = "8"

python c\coli chat `
  --model C:\Models\qwen36-colibri-q4 `
  --ctx 4096 `
  --cap 64
```

One-shot generation:

```powershell
python c\coli run `
  --model C:\Models\qwen36-colibri-q4 `
  --ctx 4096 `
  --cap 64 `
  --ngen 128 `
  "Explain photosynthesis simply."
```

API server:

```powershell
python c\coli serve `
  --model C:\Models\qwen36-colibri-q4 `
  --ctx 4096 `
  --cap 64
```

Keep `--topk 0` for the checkpoint's configured top 8. Reducing top-k can
lower I/O and compute, but changes the model and is considered experimental.

## Memory behavior

On the tested i7-6700HQ laptop:

- Dense resident weights reported by the engine: 1.81 GB.
- RSS immediately after startup: 1.76 GB.
- RSS during the longest measured run: 5.94 GB.
- A cap of 64 produced approximately 70–75% expert-cache hits in the measured
  workloads.

One extra expert slot in every layer costs roughly 63 MB for this shape.
Demand filling means the cache reaches its real working-set size gradually.

A 4K context is a practical starting point. Full-attention KV storage grows
linearly with context, while DeltaNet layers retain fixed-size recurrent state.
The checkpoint's maximum advertised context is not a practical target on this
machine.

## Measured throughput

All measurements used the Q4 model on SSD, `OMP_NUM_THREADS=8`,
`--ctx 4096`, and `--cap 64`.

| Tokens | Prompt processing | Generation | Hit rate | RSS |
|---|---:|---:|---:|---:|
| 73 prompt, 32 generated | 1.401 tok/s | 1.258 tok/s | 75.1% | 5.94 GB |
| 146 prompt, 32 generated | 0.868 tok/s | 1.356 tok/s | 74.9% | 5.24 GB |
| 1,640 prompt, 128 generated | 1.151 tok/s | 1.225 tok/s | 70.4% | 5.94 GB |

The long-run script filename referred to PP512, but the actual prompt encoded
to 1,640 tokens. Results here use the engine-reported count.

## Validation

```powershell
cd D:\desktop\colibri_test\colibri-qwen36\c
python -m unittest discover -s tests
```

The end-to-end tiny-model regression creates a checkpoint with official tensor
names, converts it, builds/loads the real C engine, exercises both sequence
layer types and MoE paths, and compares deterministic output with its oracle.
Tokenizer tests cover both modern string-form merges and legacy pair arrays.
