#!/usr/bin/env python3
"""
Convert an Any-Precision packed checkpoint into a dense HuggingFace checkpoint
(model.safetensors) that AutoModelForCausalLM -- and therefore eval_ppl.py --
can load directly.

Why this exists
---------------
cache/{packed,layerwise_packed}/<name>/ stores `*.qweight` (bit-plane packed,
GPU-warp-permuted int32) plus per-precision `*.lut{bit}` tables. None of those
key names match `model.layers.N.*.weight`, so a plain
AutoModelForCausalLM.from_pretrained() silently RANDOM-INITIALIZES every linear
layer and reports PPL in the millions. This script materializes real fp16
weights so the number means something.

Weight reconstruction (mirrors pack.py exactly):
    qweight : int32 [parent_precision, N, K//32], MSB-plane first, byte order
              permuted for the ap_gemv kernel.
    unpack  -> uint8 codes [N, K], each in [0, 2**parent_precision)
    truncate-> codes >> (parent_precision - bit)     # nested precisions
    lut{bit}: fp16 [N, 2**bit], one centroid table per output row
    W[n, k] = lut{bit}[n, codes[n, k] >> shift]

Usage
-----
    python dequant_to_hf.py <packed_path> <output_dir> [--bit 3]

    python dequant_to_hf.py \
        cache/layerwise_packed/layerwise-4e20-w3-c4_s128_blk2048_g1_iter3_cd4 \
        cache/dense/llama32-1b-w3

Then:
    python eval_ppl.py --model-path cache/dense/llama32-1b-w3 \
        --datasets wikitext2 c4 --seqlen 8192 --dtype fp16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Bit-plane unpacking. Imported from the repo when available so that any future
# change to the kernel layout is picked up automatically; the fallback is a
# verbatim copy of pack.py's inverse permutation.
# --------------------------------------------------------------------------- #
def _get_unpacker():
    try:
        from any_precision.quantization.pack import _permute_bitmaps
        return _permute_bitmaps
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"Could not import any_precision.quantization.pack ({exc}).\n"
            "Run this script from the GuidedQuant repo root, or add it to "
            "PYTHONPATH."
        )


def unpack_qweight(qweight: torch.Tensor, parent_precision: int) -> np.ndarray:
    """int32 [P, N, K//32] bit-planes -> uint8 codes [N, K]."""
    permute = _get_unpacker()

    w_bits, N, ints_per_row = qweight.shape
    if w_bits != parent_precision:
        # prune_precisions() truncates the leading dim; we load raw tensors so
        # this should not happen, but a mismatch here silently corrupts every
        # weight, so it is worth being loud about.
        raise ValueError(
            f"qweight has {w_bits} bit-planes but parent_precision="
            f"{parent_precision}. Refusing to guess."
        )

    total_bytes = ints_per_row * 4
    bitmaps = qweight.contiguous().view(torch.uint8).reshape(w_bits, N, total_bytes)
    bitmaps = permute(bitmaps.cpu().numpy(), inverse=True)

    K = ints_per_row * 32
    flat = np.zeros(N * K, dtype=np.uint8)
    planes = bitmaps.reshape(parent_precision, -1)
    for bit in range(parent_precision):
        bools = np.unpackbits(planes[bit])
        flat |= bools.astype(np.uint8) << (parent_precision - bit - 1)

    return flat.reshape(N, K)


def dequantize(qweight: torch.Tensor, lut: torch.Tensor,
               bit: int, parent_precision: int) -> torch.Tensor:
    """Reconstruct the dense fp16 weight matrix [N, K]."""
    codes = unpack_qweight(qweight, parent_precision)

    # Nested precisions: the b-bit code is the top b bits of the parent code.
    shift = parent_precision - bit
    if shift:
        codes = codes >> shift

    lut_np = lut.cpu().to(torch.float32).numpy()          # [N, 2**bit]
    N, n_centroids = lut_np.shape
    if n_centroids != 2 ** bit:
        raise ValueError(f"lut{bit} has {n_centroids} centroids, expected {2 ** bit}")
    if codes.max() >= n_centroids:
        raise ValueError(
            f"code {codes.max()} out of range for {bit}-bit LUT "
            f"({n_centroids} entries) -- bit-plane order is probably wrong"
        )

    # Per-row gather: row n indexes its own centroid table.
    dense = np.take_along_axis(lut_np, codes.astype(np.int64), axis=1)
    return torch.from_numpy(dense).to(torch.float16)


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #
def load_packed_state_dict(path: Path) -> dict:
    """Read every shard in the packed dir into one dict (kept on CPU)."""
    from safetensors.torch import load_file

    shards = sorted(path.glob("*.safetensors"))
    if shards:
        state = {}
        for shard in shards:
            state.update(load_file(str(shard), device="cpu"))
        return state

    bins = sorted(path.glob("*.bin"))
    if not bins:
        raise SystemExit(f"No .safetensors or .bin found in {path}")
    state = {}
    for b in bins:
        state.update(torch.load(str(b), map_location="cpu"))
    return state


def main():
    ap = argparse.ArgumentParser(
        description="Dequantize an Any-Precision packed model to dense fp16 safetensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("packed_path", type=str, help="cache/layerwise_packed/<name>")
    ap.add_argument("output_dir", type=str, help="destination for model.safetensors")
    ap.add_argument("--bit", type=int, default=None,
                    help="precision to materialize; default = seed_precision "
                         "(the lowest, i.e. the one you actually quantized to)")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max-shard-size", type=str, default="4GB")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    src = Path(args.packed_path)
    dst = Path(args.output_dir)
    if not src.is_dir():
        raise SystemExit(f"Not a directory: {src}")

    torch_dtype = {"fp16": torch.float16,
                   "bf16": torch.bfloat16,
                   "fp32": torch.float32}[args.dtype]

    # ---- config / precision -------------------------------------------------
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)
    anyprec = getattr(config, "anyprec", None)
    if anyprec is None:
        raise SystemExit(
            f"{src} has no `anyprec` block in config.json -- this does not look "
            "like a packed Any-Precision checkpoint. If it is already a dense "
            "HF model, point eval_ppl.py at it directly."
        )

    seed_precision = anyprec["seed_precision"]
    parent_precision = anyprec["parent_precision"]
    bit = args.bit if args.bit is not None else seed_precision
    if not (seed_precision <= bit <= parent_precision):
        raise SystemExit(
            f"--bit {bit} outside supported range "
            f"[{seed_precision}, {parent_precision}]"
        )

    print(f"Source:     {src}")
    print(f"Precisions: seed={seed_precision} parent={parent_precision} "
          f"-> materializing {bit}-bit")
    print(f"Output:     {dst}  ({args.dtype})")
    print("-" * 60)

    # ---- build the dense skeleton ------------------------------------------
    # Instantiate from config (random init) and overwrite every quantized
    # linear. Non-quantized tensors (embeddings, norms, lm_head) are copied
    # straight from the packed checkpoint, which stores them unquantized.
    base_cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    if hasattr(base_cfg, "anyprec"):
        del base_cfg.anyprec
    base_cfg.torch_dtype = torch_dtype

    model = AutoModelForCausalLM.from_config(base_cfg, trust_remote_code=True)
    model = model.to(torch_dtype)
    target = model.state_dict()

    packed = load_packed_state_dict(src)

    qweight_keys = [k for k in packed if k.endswith(".qweight")]
    if not qweight_keys:
        raise SystemExit(f"No .qweight tensors in {src}; nothing to dequantize.")
    print(f"Found {len(qweight_keys)} quantized modules")

    written, copied, missing = 0, 0, []

    # ---- dequantize every packed linear ------------------------------------
    for i, qk in enumerate(sorted(qweight_keys), 1):
        prefix = qk[: -len(".qweight")]
        weight_key = prefix + ".weight"
        lut_key = f"{prefix}.lut{bit}"

        if lut_key not in packed:
            raise SystemExit(
                f"{lut_key} missing. The checkpoint may have been pruned to a "
                f"subset of precisions; try --bit with one that exists."
            )
        if weight_key not in target:
            missing.append(weight_key)
            continue

        dense = dequantize(packed[qk], packed[lut_key], bit, parent_precision)

        expected = tuple(target[weight_key].shape)
        if tuple(dense.shape) != expected:
            # K is padded up to a multiple of 32 by the packer; trim it back.
            if dense.shape[0] == expected[0] and dense.shape[1] > expected[1]:
                dense = dense[:, : expected[1]]
            else:
                raise SystemExit(
                    f"shape mismatch for {weight_key}: got {tuple(dense.shape)}, "
                    f"expected {expected}"
                )

        target[weight_key] = dense.to(torch_dtype)
        written += 1
        if i % 25 == 0 or i == len(qweight_keys):
            print(f"  [{i}/{len(qweight_keys)}] {prefix}")

    # ---- copy the unquantized remainder ------------------------------------
    for k, v in packed.items():
        if k.endswith(".qweight") or re.search(r"\.lut\d+$", k):
            continue
        if k in target and tuple(v.shape) == tuple(target[k].shape):
            target[k] = v.to(torch_dtype)
            copied += 1

    print("-" * 60)
    print(f"Dequantized {written} weights, copied {copied} unquantized tensors")
    if missing:
        print(f"WARNING: {len(missing)} packed modules had no home in the dense "
              f"model, e.g. {missing[:3]}")

    # Any key still holding its random init is a silent corruption source.
    # Embeddings/lm_head are tied in Llama-3.2-1B and legitimately absent from
    # the checkpoint, so only flag the rest.
    untouched = [
        k for k in target
        if re.search(r"layers\.\d+\.", k)
        and k not in packed
        and k.replace(".weight", ".qweight") not in packed
    ]
    if untouched:
        print(f"WARNING: {len(untouched)} in-block tensors were never written "
              f"and remain randomly initialized, e.g. {untouched[:5]}")

    model.load_state_dict(target, strict=True)

    # ---- save ---------------------------------------------------------------
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True,
                          max_shard_size=args.max_shard_size)

    try:
        AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(dst)
    except Exception as exc:
        print(f"WARNING: could not copy tokenizer from {src} ({exc}). "
              f"Copy it from the base model manually or eval_ppl.py will fail.")

    with open(dst / "dequant_info.json", "w") as fh:
        json.dump({
            "source": str(src),
            "materialized_bit": bit,
            "seed_precision": seed_precision,
            "parent_precision": parent_precision,
            "dtype": args.dtype,
            "modules_dequantized": written,
            "note": "Dense reconstruction of a packed Any-Precision model. "
                    "Numerically equivalent to the quantized model; provides "
                    "no memory or speed benefit. For PPL evaluation only.",
        }, fh, indent=2)

    print(f"\nSaved to {dst}")
    print(f"\nNext:\n  python eval_ppl.py --model-path {dst} \\\n"
          f"      --datasets wikitext2 c4 --seqlen 8192 --dtype fp16")


if __name__ == "__main__":
    main()