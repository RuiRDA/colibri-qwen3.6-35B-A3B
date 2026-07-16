#!/usr/bin/env python3
"""Stream-convert Qwen3.6-35B-A3B-FP8 to Colibri row-wise int4.

The converter deliberately has only one non-stdlib dependency: NumPy.  It reads
FP8/BF16 safetensors directly, downloads one source shard at a time, writes the
quantized shard, and removes the source shard.  This keeps the temporary space
bounded to one Hugging Face shard and makes a D: scratch / C: model layout safe.

Example (PowerShell):
  python c/tools/convert_qwen36.py `
    --outdir C:\\Models\\qwen36-colibri-q4 `
    --workdir D:\\colibri-convert-cache
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
INDEX_FILE = "model.safetensors.index.json"
AUX_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def fp8_e4m3fn_lut() -> np.ndarray:
    """Return the exact 256-entry float8_e4m3fn decode table."""
    out = np.empty(256, dtype=np.float32)
    for raw in range(256):
        sign = -1.0 if raw & 0x80 else 1.0
        exp = (raw >> 3) & 0xF
        mant = raw & 7
        if exp == 0:
            value = (mant / 8.0) * (2.0 ** -6)
        elif exp == 15 and mant == 7:
            value = math.nan
        else:
            value = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        out[raw] = sign * value
    return out


FP8_LUT = fp8_e4m3fn_lut()


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


class SafeTensorReader:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        raw = self.file.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        hlen = struct.unpack("<Q", raw)[0]
        header = json.loads(self.file.read(hlen))
        self.data_start = 8 + hlen
        self.tensors: dict[str, TensorInfo] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            self.tensors[name] = TensorInfo(
                meta["dtype"], tuple(meta["shape"]),
                self.data_start + begin, self.data_start + end,
            )

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "SafeTensorReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def raw(self, name: str) -> np.memmap:
        t = self.tensors[name]
        dtypes = {
            "F8_E4M3": np.uint8, "F8_E4M3FN": np.uint8,
            "BF16": np.uint16, "F16": np.float16, "F32": np.float32,
            "U8": np.uint8,
        }
        if t.dtype not in dtypes:
            raise ValueError(f"unsupported source dtype {t.dtype} for {name}")
        return np.memmap(self.path, mode="r", dtype=dtypes[t.dtype],
                         offset=t.start, shape=t.shape)

    def f32(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        a = self.raw(name)
        if t.dtype in ("F8_E4M3", "F8_E4M3FN"):
            return FP8_LUT[np.asarray(a, dtype=np.uint8)]
        if t.dtype == "BF16":
            u = np.asarray(a, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32)
        return np.asarray(a, dtype=np.float32)


def normalized_name(name: str) -> str:
    prefix = "model.language_model."
    return "model." + name[len(prefix):] if name.startswith(prefix) else name


def classify(name: str) -> str:
    """Return skip, f32, io, expert, or dense for an official checkpoint key."""
    if name.startswith("model.visual.") or name.startswith("visual."):
        return "skip"
    if name.startswith("mtp."):
        return "skip"                 # base autoregressive path first; no MTP draft head
    if name.endswith("_scale_inv"):
        return "skip"                 # consumed with its FP8 weight
    n = normalized_name(name)
    if n in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if ".mlp.experts." in n and n.endswith(".weight"):
        return "expert"
    if n.endswith("mlp.gate.weight"):
        return "f32"                  # router precision matters and it is small
    if n.endswith("norm.weight") or n.endswith("layernorm.weight"):
        return "f32"
    if n.endswith((".A_log", ".dt_bias")) or ".conv1d." in n:
        return "f32"
    if n.endswith(".weight"):
        return "dense"
    return "f32"


def dtype_nbytes(dtype: str) -> int:
    return {"U8": 1, "F32": 4}[dtype]


def plan_output(reader: SafeTensorReader, dense_bits: int, expert_bits: int,
                io_bits: int) -> tuple[list[dict], dict[str, tuple[str, int]]]:
    specs: list[dict] = []
    source: dict[str, tuple[str, int]] = {}
    for src, info in reader.tensors.items():
        kind = classify(src)
        if kind == "skip":
            continue
        dst = normalized_name(src)
        bits = io_bits if kind == "io" else expert_bits if kind == "expert" else dense_bits
        if kind in ("io", "expert", "dense"):
            if len(info.shape) != 2:
                raise ValueError(f"expected matrix for quantized tensor {src}, got {info.shape}")
            rows, cols = info.shape
            row_bytes = cols if bits == 8 else (cols + 1) // 2
            specs.append({"name": dst, "dtype": "U8", "shape": [rows * row_bytes]})
            specs.append({"name": dst + ".qs", "dtype": "F32", "shape": [rows]})
            source[dst] = (src, bits)
        else:
            specs.append({"name": dst, "dtype": "F32", "shape": list(info.shape)})
            source[dst] = (src, 32)
    return specs, source


class SafeTensorWriter:
    def __init__(self, path: Path, specs: list[dict], metadata: dict[str, str]):
        self.path = path
        header: dict[str, object] = {"__metadata__": metadata}
        offset = 0
        self.offsets: dict[str, tuple[int, int]] = {}
        for s in specs:
            n = int(np.prod(s["shape"], dtype=np.int64))
            size = n * dtype_nbytes(s["dtype"])
            header[s["name"]] = {
                "dtype": s["dtype"], "shape": s["shape"],
                "data_offsets": [offset, offset + size],
            }
            self.offsets[s["name"]] = (offset, offset + size)
            offset += size
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((8 - len(encoded) % 8) % 8)
        self.data_start = 8 + len(encoded)
        self.file = path.open("w+b")
        self.file.write(struct.pack("<Q", len(encoded)))
        self.file.write(encoded)
        self.file.truncate(self.data_start + offset)

    def write(self, name: str, data: np.ndarray, byte_offset: int = 0) -> None:
        start, end = self.offsets[name]
        payload = np.ascontiguousarray(data).tobytes()
        if byte_offset + len(payload) > end - start:
            raise ValueError(f"write past tensor {name}")
        self.file.seek(self.data_start + start + byte_offset)
        self.file.write(payload)

    def close(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


def source_rows_f32(reader: SafeTensorReader, name: str, r0: int, r1: int) -> np.ndarray:
    info = reader.tensors[name]
    raw = reader.raw(name)[r0:r1]
    if info.dtype in ("F8_E4M3", "F8_E4M3FN"):
        scale_name = name + "_scale_inv"
        if scale_name not in reader.tensors:
            raise ValueError(f"missing FP8 scale {scale_name}")
        values = FP8_LUT[np.asarray(raw, dtype=np.uint8)]
        scales = reader.f32(scale_name)
        col_scale = np.repeat(scales[r0 // 128:(r1 + 127) // 128], 128, axis=1)
        col_scale = col_scale[:, :info.shape[1]]
        row_index = np.arange(r0, r1) // 128 - r0 // 128
        return values * col_scale[row_index]
    if info.dtype == "BF16":
        u = np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    return np.asarray(raw, dtype=np.float32)


def quant_rows(w: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    if bits not in (4, 8):
        raise ValueError("only 4-bit and 8-bit output are supported")
    qmax = (1 << (bits - 1)) - 1
    scale = np.maximum(np.max(np.abs(w), axis=1) / qmax, 1e-8).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -qmax - 1, qmax).astype(np.int16)
    if bits == 8:
        return q.astype(np.int8).view(np.uint8), scale
    rows, cols = q.shape
    packed = np.zeros((rows, (cols + 1) // 2), dtype=np.uint8)
    packed[:, :q[:, 0::2].shape[1]] = (q[:, 0::2] + 8).astype(np.uint8)
    if cols > 1:
        packed[:, :q[:, 1::2].shape[1]] |= ((q[:, 1::2] + 8).astype(np.uint8) << 4)
    return packed, scale


def convert_shard(src: Path, dst: Path, dense_bits: int, expert_bits: int,
                  io_bits: int, row_chunk: int) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with SafeTensorReader(src) as reader:
        specs, mapping = plan_output(reader, dense_bits, expert_bits, io_bits)
        writer = SafeTensorWriter(tmp, specs, {
            "format": "colibri-qwen36-v1",
            "dense_bits": str(dense_bits), "expert_bits": str(expert_bits),
            "io_bits": str(io_bits),
        })
        try:
            for dst_name, (src_name, bits) in mapping.items():
                info = reader.tensors[src_name]
                if bits == 32:
                    writer.write(dst_name, reader.f32(src_name).astype(np.float32, copy=False))
                    continue
                rows, cols = info.shape
                row_bytes = cols if bits == 8 else (cols + 1) // 2
                for r0 in range(0, rows, row_chunk):
                    r1 = min(rows, r0 + row_chunk)
                    w = source_rows_f32(reader, src_name, r0, r1)
                    q, scales = quant_rows(w, bits)
                    writer.write(dst_name, q, r0 * row_bytes)
                    writer.write(dst_name + ".qs", scales, r0 * 4)
        finally:
            writer.close()
    os.replace(tmp, dst)


def hf_url(repo: str, filename: str) -> str:
    quoted = "/".join(urllib.parse.quote(p) for p in filename.split("/"))
    return f"https://huggingface.co/{repo}/resolve/main/{quoted}"


def request(url: str, start: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": "colibri-qwen36/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def download(repo: str, filename: str, dest: Path, retries: int = 50) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        try:
            with urllib.request.urlopen(request(hf_url(repo, filename), have), timeout=30) as r:
                mode = "ab" if have and getattr(r, "status", 200) == 206 else "wb"
                with part.open(mode) as f:
                    while True:
                        block = r.read(4 << 20)
                        if not block:
                            break
                        f.write(block)
            os.replace(part, dest)
            return dest
        except (OSError, urllib.error.URLError) as exc:
            if attempt + 1 == retries:
                raise
            print(f"  download interrupted ({exc}); resume in {min(30, attempt + 2)}s", flush=True)
            time.sleep(min(30, attempt + 2))
    raise AssertionError("unreachable")


def fetch_json(repo: str, filename: str, dest: Path) -> dict:
    download(repo, filename, dest)
    return json.loads(dest.read_text(encoding="utf-8"))


def validate_existing(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 16:
        return False
    try:
        with SafeTensorReader(path) as r:
            return bool(r.tensors)
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True,
                    help="temporary shard directory; D: is recommended")
    ap.add_argument("--dense-bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--expert-bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--io-bits", type=int, choices=(4, 8), default=8)
    ap.add_argument("--row-chunk", type=int, default=256)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    lock = args.outdir / ".convert.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"another conversion (or stale lock) exists: {lock}", file=sys.stderr)
        return 2
    try:
        index_path = args.workdir / INDEX_FILE
        index = fetch_json(args.repo, INDEX_FILE, index_path)
        shards = sorted(set(index["weight_map"].values()))
        for aux in AUX_FILES:
            try:
                download(args.repo, aux, args.outdir / aux, retries=3)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        manifest = {
            "format": "colibri-qwen36-v1", "source": args.repo,
            "dense_bits": args.dense_bits, "expert_bits": args.expert_bits,
            "io_bits": args.io_bits, "text_only": True, "mtp": False,
        }
        (args.outdir / "colibri.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{len(shards)} source shards; conversion is resumable", flush=True)
        for i, filename in enumerate(shards, 1):
            out = args.outdir / f"out-{i:05d}-of-{len(shards):05d}.safetensors"
            if validate_existing(out):
                print(f"[{i}/{len(shards)}] already converted: {out.name}", flush=True)
                continue
            if shutil.disk_usage(args.outdir).free / 1e9 < args.min_free_gb:
                raise OSError(f"less than {args.min_free_gb:g} GB free at {args.outdir}")
            src = args.workdir / Path(filename).name
            print(f"[{i}/{len(shards)}] download {filename}", flush=True)
            download(args.repo, filename, src)
            print(f"[{i}/{len(shards)}] quantize -> {out.name}", flush=True)
            convert_shard(src, out, args.dense_bits, args.expert_bits,
                          args.io_bits, args.row_chunk)
            src.unlink()
        total = sum(p.stat().st_size for p in args.outdir.glob("*.safetensors"))
        print(f"complete: {args.outdir} ({total / 1e9:.2f} GB)")
        return 0
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
