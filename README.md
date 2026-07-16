# Colibri Qwen3.6-35B-A3B

CPU-only, dependency-light inference for the text portion of
[Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8), converted
to a compact 4-bit Colibri layout.

This repository is an experimental fork of
[JustVugg/colibri](https://github.com/JustVugg/colibri). It keeps Colibri's
central idea—dense weights in RAM and routed MoE experts streamed from
storage—but adds a dedicated engine for Qwen3.6's hybrid Gated DeltaNet and
full-attention architecture.

## What works

- Text-only Qwen3.6-35B-A3B inference on CPU.
- Official FP8 checkpoint conversion to row-wise Q4/Q8 safetensors.
- Windows AVX2/OpenMP native engine with no runtime Python dependency.
- Interactive chat and one-shot generation.
- OpenAI-compatible HTTP API.
- Resumable, one-shard-at-a-time conversion.
- Per-layer LRU expert cache controlled with `--cap`.
- Separate prompt-processing and token-generation timing.

Vision input and the MTP draft head are intentionally not implemented. They
are unnecessary for the text CPU baseline and would increase storage and
memory use.

## Architecture implemented

| Component | Qwen3.6-35B-A3B support |
|---|---|
| Decoder | 40 layers |
| Hybrid sequence model | 30 recurrent Gated DeltaNet layers and 10 full-attention layers |
| MoE | 256 routed experts per layer, top 8, plus one gated shared expert |
| Normalization | Centered Qwen RMSNorm |
| Attention details | Partial split-half RoPE and gated query output |
| Recurrent state | FP32 DeltaNet state |
| Context state | Bounded full-attention KV cache |
| Dense and expert matrices | Row-wise symmetric int4 |
| Embedding and LM head | Row-wise int8 |
| Expert residency | On-disk safetensors with a per-layer LRU cache |

The new engine is [`c/qwen36.c`](c/qwen36.c). The original GLM engine remains
available as [`c/glm.c`](c/glm.c), but the `coli` launcher in this fork
targets Qwen3.6.

## How the conversion works

The converter, [`c/tools/convert_qwen36.py`](c/tools/convert_qwen36.py), does
not require PyTorch or a second full copy of the source checkpoint.

For every official FP8 shard it:

1. Resumes or downloads that source shard into the work directory.
2. Reads safetensors metadata and tensor rows directly.
3. Decodes FP8 E4M3FN or BF16 values with NumPy.
4. Keeps normalization and router tensors in FP32.
5. Quantizes embeddings and the language-model head to row-wise int8.
6. Quantizes dense and expert matrices to packed row-wise int4.
7. Writes a completed Colibri safetensors shard atomically.
8. Deletes the temporary source shard and continues with the next one.

Each quantized row stores its own FP32 scale. Conversion is resumable:
completed destination shards are validated and skipped on the next run.

## Tested machine

The implementation was built and measured on:

- Intel Core i7-6700HQ, 4 cores / 8 threads, AVX2.
- 15.9 GB system RAM.
- Windows with GCC/OpenMP from portable w64devkit.
- Converted model stored on the C: SSD.
- Q4 model size: approximately 18.05 GB decimal / 16.81 GiB.

Allow at least 35 GB free on the destination drive before conversion so the
final model, an in-progress shard, and normal filesystem headroom all fit.
An HDD works, but cold expert-cache misses are sensitive to random seek
latency; an SSD is strongly preferred.

## Build

Requirements:

- Python 3.10 or newer for the launcher and converter.
- NumPy for conversion.
- GNU Make and GCC with OpenMP for building the C engine.

On Windows, the launcher also detects w64devkit placed at
`<repository-parent>\.tools\w64devkit`.

```powershell
git clone https://github.com/RuiRDA/colibri-qwen3.6-35B-A3B.git
cd colibri-qwen3.6-35B-A3B
python c\coli build
```

The resulting executable is `c\qwen36.exe`. The executable itself does not
need Python or NumPy.

## Convert the model

The recommended layout uses the SSD for the final model and another drive for
temporary source shards:

```powershell
python c\coli convert `
  --model C:\Models\qwen36-colibri-q4 `
  --workdir D:\colibri-convert-cache
```

The defaults are 4-bit dense matrices, 4-bit expert matrices, and 8-bit
embedding/output matrices. Re-run the same command after an interruption; it
resumes completed shards and partial downloads.

## Chat

```powershell
$env:OMP_NUM_THREADS = "8"
python c\coli chat `
  --model C:\Models\qwen36-colibri-q4 `
  --ctx 4096 `
  --cap 64
```

Inside chat, use `:more` to continue a response and `:q` to quit.

For a single prompt:

```powershell
python c\coli run `
  --model C:\Models\qwen36-colibri-q4 `
  --ctx 4096 `
  --cap 64 `
  --ngen 128 `
  "Explain quantum computing simply."
```

### What `--cap 64` means

`--cap` is the number of routed expert slots cached **per layer**. With
`--cap 64`, each of the 40 layers can retain up to 64 recently used experts.

- A higher value consumes more RAM but avoids more storage reads.
- A lower value saves RAM but causes more expert reloads.
- `64` was a useful balance on the tested 16 GB machine.
- Try `--cap 32` if other applications need more RAM.

At this model shape, one additional cache slot across all layers costs roughly
63 MB. The cache is demand-filled, so startup does not immediately allocate
the maximum.

## OpenAI-compatible API

```powershell
python c\coli serve `
  --model C:\Models\qwen36-colibri-q4 `
  --ctx 4096 `
  --cap 64 `
  --host 127.0.0.1 `
  --port 8000
```

The server selects the Qwen ChatML renderer from the converted model config and
exposes the existing Colibri chat-completions interface.

## Results from the tested PC

Settings for all runs below: Q4 model on the C: SSD, `OMP_NUM_THREADS=8`,
`--ctx 4096`, `--cap 64`, and greedy generation where noted.

| Run | Prompt processing | Token generation | Expert-cache hit | RSS |
|---|---:|---:|---:|---:|
| Startup | Ready in 3.2 s; 1.81 GB dense resident | — | — | 1.76 GB |
| Short benchmark, 73 prompt + 32 generated tokens | 52.1 s, **1.401 tok/s** | 25.4 s, **1.258 tok/s** | 75.1% | 5.94 GB |
| Short benchmark, 146 prompt + 32 generated tokens | 168.2 s, **0.868 tok/s** | 23.6 s, **1.356 tok/s** | 74.9% | 5.24 GB |
| Long benchmark, 1,640 prompt + 128 generated tokens | 1,425.3 s, **1.151 tok/s** | 104.5 s, **1.225 tok/s** | 70.4% | 5.94 GB |

The long run was originally launched from a file named `pp512_tg128`, but the
constructed repeated prompt tokenized to **1,640 tokens**, not 512. The table
reports the engine's measured token count and should not be presented as a
standardized PP512 score.

These numbers are a baseline for this specific older AVX2 laptop CPU. Prompt
speed varies with routing, cache warmth, prompt structure, storage latency, and
thermal behavior. The model is usable for patient local experimentation, but
it is not an interactive-speed CPU configuration.

## Validation

The test suite covers:

- Tensor classification and row-wise Q4/Q8 conversion.
- Resumable safetensors output.
- Nested Qwen text configuration and resource planning.
- Modern string-form and legacy array-form tokenizer merges.
- A deterministic tiny hybrid Qwen model through the real C loader.
- Gated DeltaNet, full attention, shared/routed experts, chat rendering, and
  streamed generation.
- CLI and Windows-specific diagnostics.

Run it with:

```powershell
cd c
python -m unittest discover -s tests
```

The post-merge validation passed 74 tests with four expected platform/toolchain
skips on Windows.

## Tuning notes

- Keep `--topk 0` for the configured top 8 experts. Lower top-k values trade
  model quality for fewer expert reads and are experimental.
- Start with `--ctx 4096`; the checkpoint's maximum context is not a sensible
  starting point on a 16 GB machine.
- Use the SSD for the final converted model when possible.
- Compare 4 and 8 OpenMP threads on your system. Eight was used for the
  reported runs, while four may reduce contention on some workloads.
- Measure warmed multi-turn chat before changing the cache size.

For a more detailed implementation and machine setup guide, see
[QWEN36.md](QWEN36.md).

## Credits and license

This work is based on [JustVugg/colibri](https://github.com/JustVugg/colibri).
Qwen model weights are distributed separately under their applicable model
license. Repository source code remains under the license in [LICENSE](LICENSE).
