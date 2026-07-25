#!/usr/bin/env python3
"""
Standalone perplexity evaluation.

PRIMARY METRIC: non-overlapping block PPL, matching the GPTQ / AWQ /
OmniQuant / SpinQuant protocol. The corpus is concatenated, tokenized once,
then split into blocks [0:seqlen], [seqlen:2*seqlen], ... Each block is scored
independently: no context carried over from the previous block, no masking.

    Sanity check: Llama-2-7B  fp16, wikitext2, seqlen=2048 -> 5.47 (166 blocks)
                  Llama-2-13B fp16, wikitext2, seqlen=2048 -> 4.88

Off by more than 0.02 means a bug in tokenization or model loading, NOT in the
quantizer. Stop and fix it before running the ablation ladder.

SECONDARY METRIC (--method sliding): overlapping sliding window. A closer
estimate of true PPL -- HuggingFace calls the non-overlapping split a
"suboptimal" approximation because the first tokens of each block are scored
with no context -- BUT it yields noticeably lower numbers. Never place these
side by side with numbers from a paper table. Always report ctx_len and stride.

Usage:
    python eval_ppl.py --model-path ./quantized_models/flexnu/C_divisor_only \
        --datasets wikitext2 c4 --seqlen 2048

    python eval_ppl.py --model-path meta-llama/Llama-2-7b-hf \
        --datasets wikitext2 --seqlen 2048 --dtype fp16    # -> must print 5.47
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Corpus loading
#
# CRITICAL - these three details decide whether the numbers match paper tables:
#   1. wikitext2 uses "\n\n".join and does NOT filter empty lines.
#   2. Tokenize ONCE over the whole corpus; do NOT insert BOS manually.
#   3. c4 uses get_c4_new (join first 1100 docs, truncate to 256*seqlen tokens).
# --------------------------------------------------------------------------- #
def _load_corpus_ids(name: str, tokenizer, seqlen: int,
                     cache_dir: Optional[Path] = None) -> torch.Tensor:
    """Return input_ids [1, N] for the entire test corpus."""
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # The cache key must capture everything that can change the token stream.
        # Keying on the model name alone is a silent-corruption trap: flip
        # use_fast or add_bos_token, rerun, and you would read back a stale
        # tokenization and report wrong numbers with no warning.
        fingerprint = "|".join([
            name,
            str(getattr(tokenizer, "name_or_path", "tok")),
            type(tokenizer).__name__,
            f"fast={bool(getattr(tokenizer, 'is_fast', False))}",
            f"bos={getattr(tokenizer, 'add_bos_token', None)}",
            f"eos={getattr(tokenizer, 'add_eos_token', None)}",
            f"vocab={len(tokenizer)}",
            f"seqlen={seqlen}" if name == "c4" else "seqlen=na",
            f"tfm={_transformers_version()}",
        ])
        digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
        cache_file = cache_dir / f"{name}_{digest}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)

    from datasets import load_dataset

    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # NO filtering: keep empty lines, exactly as in GPTQ datautils.py
        enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    elif name == "c4":
        # GPTQ get_c4_new (the --new-eval path, which is what later papers
        # follow). NOT the older get_c4, which samples 256 separate
        # seqlen-long excerpts at random.
        # revision pinned so every run sees the same text.
        ds = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        enc = tokenizer(" ".join(ds[:1100]["text"]), return_tensors="pt").input_ids
        enc = enc[:, : 256 * seqlen]

    elif name == "ptb-new":
        # GPTQ get_ptb_new: test split, " ".join. The older get_ptb uses the
        # validation split and "\n\n".join and gives different numbers.
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
        enc = tokenizer(" ".join(ds["sentence"]), return_tensors="pt").input_ids

    else:
        raise ValueError(f"Unknown dataset: {name}")

    if cache_file is not None:
        with open(cache_file, "wb") as fh:
            pickle.dump(enc, fh)
    return enc


# --------------------------------------------------------------------------- #
# PRIMARY: non-overlapping block PPL (GPTQ protocol)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_ppl(model, tokenizer, testcases: List[str], seqlen: int = 2048,
             cache_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    results: Dict[str, float] = {}

    # KV cache is useless for teacher-forced scoring and wastes VRAM.
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    for name in testcases:
        enc = _load_corpus_ids(name, tokenizer, seqlen, cache_dir)
        nsamples = enc.numel() // seqlen
        if nsamples == 0:
            if verbose:
                print(f"{name}: corpus shorter than seqlen, skipping")
            continue

        nlls = []
        for i in tqdm(range(nsamples), disable=not verbose, desc=f"{name}"):
            batch = enc[:, i * seqlen : (i + 1) * seqlen].to(device)
            out = model(batch, labels=batch)
            # out.loss is the mean over (seqlen - 1) scored tokens.
            # Multiplying by seqlen here and dividing by nsamples*seqlen below:
            # the factor cancels, so the result is identical to using
            # (seqlen - 1) in both places. Kept as seqlen to match the original
            # GPTQ repo verbatim.
            nlls.append(out.loss.float() * seqlen)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
        results[name] = ppl
        if verbose:
            print(f"{name}: PPL = {ppl:.4f}  "
                  f"({nsamples} blocks x {seqlen} tokens = {nsamples * seqlen:,})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    return results


# --------------------------------------------------------------------------- #
# SECONDARY: overlapping sliding window
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_ppl_sliding(model, tokenizer, testcases: List[str], ctx_len: int = 2048,
                     stride: int = 512, cache_dir: Optional[Path] = None,
                     verbose: bool = True) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    results: Dict[str, float] = {}

    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    for name in testcases:
        enc = _load_corpus_ids(name, tokenizer, ctx_len, cache_dir).to(device)
        seq_len = enc.size(1)
        if seq_len < 2:
            continue

        nll_sum = 0.0
        n_scored = 0
        prev_end = 0

        for begin in tqdm(range(0, seq_len, stride), disable=not verbose,
                          desc=f"{name} (sliding)"):
            end = min(begin + ctx_len, seq_len)
            trg_len = end - prev_end          # number of NEW tokens in this window
            if trg_len <= 0:
                # This window adds no new tokens (happens when stride > ctx_len,
                # or on a truncated final window) -> skip to avoid bad masking.
                if end == seq_len:
                    break
                continue
            chunk = enc[:, begin:end]

            target = chunk.clone()
            n_ctx = chunk.size(1) - trg_len   # context tokens to mask out
            if n_ctx > 0:
                target[:, :n_ctx] = -100

            out = model(chunk, labels=target)
            # HF shifts labels: token i is predicted from tokens < i.
            # Tokens actually contributing loss = labels != -100 after the shift.
            valid = int((target[:, 1:] != -100).sum().item())
            if valid == 0:
                prev_end = end
                if end == seq_len:
                    break
                continue

            nll_sum += out.loss.float().item() * valid
            n_scored += valid

            prev_end = end
            if end == seq_len:
                break

        if n_scored == 0:
            continue

        ppl = float(torch.exp(torch.tensor(nll_sum / n_scored)))
        results[name] = ppl
        if verbose:
            print(f"{name}: PPL = {ppl:.4f}  (ctx={ctx_len}, stride={stride}, "
                  f"{n_scored:,} scored tokens)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    p = argparse.ArgumentParser(description="Perplexity evaluation (GPTQ protocol)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--datasets", type=str, nargs="+", default=["wikitext2"],
                   choices=["wikitext2", "c4", "ptb-new"],
                   help="c4 uses GPTQ's get_c4_new loader; ptb-new uses "
                        "get_ptb_new (test split), which differs from the "
                        "older get_ptb.")
    p.add_argument("--seqlen", type=int, default=2048,
                   help="2048 for Llama-1/2; 8192 for Llama-3/Qwen3. "
                        "Numbers at different seqlen are NOT comparable.")
    p.add_argument("--method", type=str, default="block",
                   choices=["block", "sliding"],
                   help="block = non-overlapping (paper standard); "
                        "sliding = overlapping (lower numbers, secondary metric)")
    p.add_argument("--stride", type=int, default=512,
                   help="only used with --method sliding")
    p.add_argument("--dtype", type=str, default="fp16", choices=list(_DTYPES))
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--cache-dir", type=str, default="./dataset_cache",
                   help="cache for tokenized corpora; '' to disable")
    p.add_argument("--out-json", type=str, default=None,
                   help="default: <model-path>/ppl.json")
    p.add_argument("--tag", type=str, default=None,
                   help="label recorded in the json (e.g. the ablation cell)")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _DTYPES[args.dtype]
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    print(f"Model:    {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Method:   {args.method}"
          + (f" (stride={args.stride})" if args.method == "sliding" else ""))
    print(f"Seqlen:   {args.seqlen}   dtype: {args.dtype}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    if args.method == "block":
        ppls = eval_ppl(model, tokenizer, args.datasets,
                        seqlen=args.seqlen, cache_dir=cache_dir)
    else:
        ppls = eval_ppl_sliding(model, tokenizer, args.datasets,
                                ctx_len=args.seqlen, stride=args.stride,
                                cache_dir=cache_dir)

    results = {
        "model_path": args.model_path,
        "tag": args.tag,
        "method": args.method,
        "seqlen": args.seqlen,
        "dtype": args.dtype,
        **({"stride": args.stride} if args.method == "sliding" else {}),
        "ppl": ppls,
        # Recorded so a number can be traced back to the exact protocol later.
        "env": {
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "vocab_size": len(tokenizer),
            "transformers": _transformers_version(),
            "torch": torch.__version__,
        },
    }

    out = args.out_json or os.path.join(args.model_path, "ppl.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("-" * 60)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
