#!/usr/bin/env python3
"""
Resolve a model reference to something transformers can load.

Accepts, in order of preference:

  1. A local directory                  -> returned unchanged
  2. A full HF repo id ("meta-llama/Llama-3.2-1B")
  3. A BARE model name ("Llama-3.2-1B") -> matched against the local HF cache

Case 3 is the point of this file. Snapshot paths like

    ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de36...

are unreadable, differ per machine, and the commit hash changes whenever the
repo is re-pulled -- which silently invalidates SALIENCY_DIR and the
calibration cache keys, since those are derived from basename(MODEL_PATH).
Passing a stable short name fixes that too.

Resolution never downloads. If nothing is cached, the input string is returned
unchanged so `from_pretrained` can fetch it (or fail) exactly as before.

Usage from bash:
    MODEL_PATH=$(python resolve_model.py "$MODEL_PATH")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def _hf_cache_dir() -> Path:
    """Honour the same env vars huggingface_hub does."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("TRANSFORMERS_CACHE"):
        return Path(os.environ["TRANSFORMERS_CACHE"])
    return Path.home() / ".cache" / "huggingface" / "hub"


def _newest_snapshot(model_dir: Path) -> Optional[str]:
    """Newest snapshot under a models--org--name/ cache entry.

    A repo can have several commits cached. Newest-by-mtime is the pragmatic
    choice; if that is not what you want, pass the explicit path.
    """
    snaps = model_dir / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [d for d in snaps.iterdir() if d.is_dir()]
    if not candidates:
        return None
    # A snapshot with no config.json is a partial/aborted download.
    complete = [d for d in candidates if (d / "config.json").exists()]
    pool = complete or candidates
    return str(max(pool, key=lambda d: d.stat().st_mtime))


def _cache_entries() -> List[Path]:
    cache = _hf_cache_dir()
    if not cache.is_dir():
        return []
    return [d for d in cache.iterdir()
            if d.is_dir() and d.name.startswith("models--")]


def _entry_repo_id(entry: Path) -> str:
    """models--meta-llama--Llama-3.2-1B -> meta-llama/Llama-3.2-1B"""
    return entry.name[len("models--"):].replace("--", "/")


def resolve(ref: str) -> str:
    ref = (ref or "").strip()
    if not ref:
        return ref

    # 1. Already a usable local directory.
    if Path(ref).expanduser().is_dir():
        return str(Path(ref).expanduser())

    entries = _cache_entries()

    # 2. Full repo id -- exact cache hit, else hand back for download.
    if "/" in ref:
        want = "models--" + ref.replace("/", "--")
        for e in entries:
            if e.name == want:
                snap = _newest_snapshot(e)
                if snap:
                    return snap
        return ref

    # 3. Bare name. Match the part after the org, case-insensitively.
    lowered = ref.lower()
    matches = [e for e in entries
               if _entry_repo_id(e).split("/")[-1].lower() == lowered]

    if not matches:
        # Nothing cached under that name. Returning it unchanged means
        # from_pretrained raises its own (clearer) error about the repo id.
        print(f"[resolve_model] no cached model named {ref!r} in "
              f"{_hf_cache_dir()}; passing through unchanged",
              file=sys.stderr)
        return ref

    if len(matches) > 1:
        names = ", ".join(sorted(_entry_repo_id(m) for m in matches))
        print(f"[resolve_model] {ref!r} is ambiguous: {names}. "
              f"Pass the full org/name to disambiguate.", file=sys.stderr)
        sys.exit(2)

    snap = _newest_snapshot(matches[0])
    if snap is None:
        print(f"[resolve_model] cache entry for {ref!r} has no usable "
              f"snapshot; passing through unchanged", file=sys.stderr)
        return ref
    return snap


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: resolve_model.py <model-name-or-path>", file=sys.stderr)
        sys.exit(1)
    print(resolve(sys.argv[1]))


if __name__ == "__main__":
    main()
