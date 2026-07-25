#!/usr/bin/env python3
"""
Why do LNQ and FlexNu give identical results?

Run from the repo root:

    python diagnose_flexnu.py
    python diagnose_flexnu.py --model Llama-3.2-1B --bits 3 --groups 1

Checks, in order of how cheaply they explain the symptom:

  1. Logs      -- did train_flexnu ever run? did it move any weights?
  2. Silent skips -- "already processed" / "skip packing" guards firing
  3. Artifacts -- are the two runs' quantized tensors actually different?
  4. Packed    -- are the two packed checkpoints different?
  5. Dense     -- are the two dequantized models different?

The first check that FAILS is the answer; later checks are then usually
consequences rather than independent problems.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

try:
    import torch
except ImportError:
    sys.exit("torch not found -- run this in the environment you quantize in.")


G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; B = "\033[1m"; N = "\033[0m"
if not sys.stdout.isatty():
    G = R = Y = B = N = ""

FINDINGS: list[str] = []


def head(t: str) -> None:
    print(f"\n{B}{'=' * 72}\n{t}\n{'=' * 72}{N}")


def ok(m: str) -> None:
    print(f"  {G}[ok]{N}   {m}")


def bad(m: str) -> None:
    print(f"  {R}[BAD]{N}  {m}")
    FINDINGS.append(m)


def warn(m: str) -> None:
    print(f"  {Y}[warn]{N} {m}")


def info(m: str) -> None:
    print(f"         {m}")


# --------------------------------------------------------------------------- #
# 1. Logs
# --------------------------------------------------------------------------- #
def check_logs() -> bool:
    """Returns True if train_flexnu demonstrably ran."""
    head("1. LOGS -- did train_flexnu actually run?")

    logs = sorted(glob.glob("logs_layer/*_flexnu_*.txt"))
    if not logs:
        bad("No logs_layer/*_flexnu_*.txt at all. FlexNu was never launched, "
            "or it died before logging.basicConfig().")
        info("Rerun and read stdout -- tracebacks do NOT reach the log file:")
        info("  bash scripts/run_flexnu_gq.sh <model> 3 1 2>&1 | tail -40")
        return False

    info(f"{len(logs)} flexnu log(s); inspecting the most recent 3")
    ran = False

    for lg in logs[-3:]:
        txt = open(lg, errors="ignore").read()
        hits = txt.count("[flexnu]")
        print(f"\n  {os.path.basename(lg)}")
        print(f"    lines={len(txt.splitlines()):<6} [flexnu]={hits}")

        # Early-return guards, in the order layerwise_main.py hits them.
        for pat, why in [
            ("Need to provide it",
             "SqueezeLLM init missing -> returned before seed(). "
             "Run scripts/run_sqllm.sh first, with the SAME --model_name and bits."),
            ("All layers have already been processed",
             "load_progress() skipped every layer -- stale files in the _flexnu "
             "quantized dir. rm -rf it and rerun."),
            ("already exists and is not empty",
             "Packing skipped -- stale _flexnu packed dir preserved. THIS ALONE "
             "explains identical results: you evaluated an old checkpoint."),
        ]:
            if pat in txt:
                bad(f"{os.path.basename(lg)}: {why}")

        if hits == 0:
            if len(txt.splitlines()) < 10:
                bad(f"{os.path.basename(lg)}: only "
                    f"{len(txt.splitlines())} lines -- the process died early. "
                    f"Rerun piping stdout to see the traceback.")
            continue

        ran = True
        moved = [float(m) for m in re.findall(r"moved=([\d.]+)%", txt)]
        drops = [float(m) for m in re.findall(r"drop=([-\d.]+)%", txt)]
        if moved:
            mx, av = max(moved), sum(moved) / len(moved)
            print(f"    moved: mean={av:.2f}%  max={mx:.2f}%  (n={len(moved)})")
            if mx == 0.0:
                warn("moved=0% everywhere -- the escape never fired. The guard "
                     "fell back to the SqueezeLLM init.")
                info("NOTE: that still differs from converged LNQ, so identical "
                     "PPL would mean you evaluated the wrong directory.")
                info("If H is near-diagonal this is CORRECT behaviour "
                     "(Proposition 1), not a bug.")
            else:
                ok(f"escape fired on {len(moved)} modules")
        if drops:
            print(f"    energy drop: mean={sum(drops)/len(drops):.2f}%  "
                  f"best={max(drops):.2f}%")

    if ran:
        ok("train_flexnu ran")
    else:
        bad("No [flexnu] line in any log -- the solver was never reached.")
    return ran


# --------------------------------------------------------------------------- #
# 2 + 3. Quantized artifacts
# --------------------------------------------------------------------------- #
def _pair(pattern: str, label: str):
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not dirs:
        bad(f"No {label} directories match {pattern}")
        return None, None
    for d in dirs:
        n = len(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True)) or \
            len(glob.glob(os.path.join(d, "*")))
        print(f"    {d}  ({n} files)")
    lnq = [d for d in dirs if not d.endswith("_flexnu")]
    fnu = [d for d in dirs if d.endswith("_flexnu")]
    if not lnq:
        bad(f"No LNQ {label} dir (one without _flexnu)")
    if not fnu:
        bad(f"No FlexNu {label} dir (one ending _flexnu)")
    if not lnq or not fnu:
        return None, None

    if len(lnq) > 1 or len(fnu) > 1:
        warn("Multiple candidates -- model names differ between runs? "
             "Using the first of each. Check for hash-named leftovers.")
    return lnq[0], fnu[0]


def check_quantized(model: str, bits: int, groups: int) -> bool | None:
    head("2/3. QUANTIZED ARTIFACTS -- do the two runs differ?")
    pat = f"cache/layerwise_quantized/*w{bits}*g{groups}_iter*"
    lnq, fnu = _pair(pat, "quantized")
    if not lnq:
        return None

    print(f"\n  LNQ    {lnq}\n  FlexNu {fnu}\n")

    any_diff = False
    checked = 0
    for l in range(3):
        for kind in ("weights", f"lut_{bits}"):
            a = os.path.join(lnq, kind, f"l{l}.pt")
            b = os.path.join(fnu, kind, f"l{l}.pt")
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            da, db = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
            for k in list(da.keys())[:2]:
                if k not in db:
                    continue
                x = torch.as_tensor(da[k]).float()
                y = torch.as_tensor(db[k]).float()
                checked += 1
                if x.shape != y.shape:
                    bad(f"l{l} {kind} {k}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
                    any_diff = True
                    continue
                same = torch.equal(x, y)
                frac = (x != y).float().mean().item()
                any_diff |= not same
                tag = f"{R}IDENTICAL{N}" if same else f"{G}differs {100*frac:5.2f}%{N}"
                print(f"    l{l} {kind:8s} {k:26s} {tag}")

    if checked == 0:
        bad("No comparable .pt files -- at least one run produced nothing.")
        return None
    print()
    if any_diff:
        ok("Quantized artifacts DIFFER -> the solver did run differently.")
        info("So the problem is downstream: packing, dequant, or which dense "
             "directory you evaluated.")
    else:
        bad("Quantized artifacts are IDENTICAL -> the solver did not run "
            "differently. Cause is in the logs above, not downstream.")
    return any_diff


# --------------------------------------------------------------------------- #
# 4. Packed
# --------------------------------------------------------------------------- #
def check_packed(bits: int, groups: int) -> None:
    head("4. PACKED CHECKPOINTS")
    pat = f"cache/layerwise_packed/*w{bits}*g{groups}_iter*"
    lnq, fnu = _pair(pat, "packed")
    if not lnq:
        return

    import datetime
    for d, nm in ((lnq, "LNQ"), (fnu, "FlexNu")):
        mt = max((os.path.getmtime(f) for f in
                  glob.glob(os.path.join(d, "**", "*"), recursive=True)
                  if os.path.isfile(f)), default=os.path.getmtime(d))
        print(f"    {nm:7s} mtime {datetime.datetime.fromtimestamp(mt):%Y-%m-%d %H:%M:%S}")
    info("If the FlexNu mtime PREDATES your last run, packing was skipped and "
         "you evaluated a stale checkpoint.")

    fa = sorted(glob.glob(os.path.join(lnq, "*.safetensors")) +
                glob.glob(os.path.join(lnq, "*.bin")))
    fb = sorted(glob.glob(os.path.join(fnu, "*.safetensors")) +
                glob.glob(os.path.join(fnu, "*.bin")))
    if fa and fb and os.path.getsize(fa[0]) == os.path.getsize(fb[0]):
        info(f"(identical file sizes -- expected, the format is fixed-width)")


# --------------------------------------------------------------------------- #
# 5. Dense
# --------------------------------------------------------------------------- #
def check_dense() -> None:
    head("5. DENSE MODELS -- did you evaluate two different directories?")
    dirs = sorted(d for d in glob.glob("cache/dense/*") if os.path.isdir(d))
    if len(dirs) < 2:
        warn(f"Only {len(dirs)} dense dir(s). Dequantize both arms to "
             f"SEPARATE output directories.")
        for d in dirs:
            print(f"    {d}")
        return

    for d in dirs:
        print(f"    {d}")

    try:
        from safetensors.torch import load_file
    except ImportError:
        warn("safetensors not available; skipping tensor comparison.")
        return

    def first_shard(d):
        f = sorted(glob.glob(os.path.join(d, "*.safetensors")))
        return f[0] if f else None

    a, b = first_shard(dirs[0]), first_shard(dirs[1])
    if not (a and b):
        warn("No .safetensors found; skipping comparison.")
        return

    ta, tb = load_file(a), load_file(b)
    key = next((k for k in ta if "layers.0" in k and "proj.weight" in k), None)
    if key is None or key not in tb:
        warn("No comparable projection weight found.")
        return

    x, y = ta[key].float(), tb[key].float()
    frac = (x != y).float().mean().item()
    print()
    if torch.equal(x, y):
        bad(f"{os.path.basename(dirs[0])} and {os.path.basename(dirs[1])} have "
            f"IDENTICAL {key} -- these are the same model.")
    else:
        ok(f"Dense models differ ({100*frac:.2f}% of {key}). If PPL is still "
           f"identical, you evaluated the same --model-path twice.")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Llama-3.2-1B")
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--groups", type=int, default=1)
    a = p.parse_args()

    print(f"{B}FlexNu vs LNQ -- why identical?{N}")
    print(f"model={a.model} bits={a.bits} groups={a.groups}  cwd={os.getcwd()}")

    if not os.path.isdir("cache"):
        sys.exit(f"\n{R}No ./cache -- run this from the repo root.{N}")

    ran = check_logs()
    diff = check_quantized(a.model, a.bits, a.groups)
    check_packed(a.bits, a.groups)
    check_dense()

    head("VERDICT")
    if not ran:
        print("  train_flexnu never ran. Fix that first; everything else is a "
              "consequence.\n")
        print("  Most likely, in order:")
        print("    1. SqueezeLLM init missing or under a different model name")
        print("    2. Stale _flexnu dirs causing silent skips")
        print("    3. A crash before seed() -- rerun and read stdout")
    elif diff is False:
        print("  Solver ran but produced identical output. Almost certainly the")
        print("  layer-skip guard: the _flexnu quantized dir already had files.")
    elif diff is True:
        print("  Solver ran AND produced different weights. The bug is downstream:")
        print("  stale packed dir, or the same --model-path evaluated twice.")
    else:
        print("  Inconclusive -- see the BAD lines above.")

    if FINDINGS:
        print(f"\n{R}{len(FINDINGS)} problem(s):{N}")
        for i, f in enumerate(FINDINGS, 1):
            print(f"  {i}. {f}")

    print(f"\n{B}Clean-slate rerun{N}")
    print("  rm -rf cache/layerwise_quantized/*_flexnu \\")
    print("         cache/layerwise_packed/*_flexnu cache/dense/*")
    print("  bash scripts/run_flexnu_gq.sh <model> <bits> <g> 2>&1 | tail -40")


if __name__ == "__main__":
    main()