"""
LNQ + FlexNu layered solver.

Runs LNQ to convergence, then starts FlexNu from LNQ's committed codebook and
assignments. The point is to separate two contributions that are otherwise
confounded:

    SqueezeLLM init  --LNQ-->  LNQ solution  --FlexNu-->  final

LNQ closes the "better search within the nearest-codeword set" gap; whatever
FlexNu adds on top can only come from LEAVING that set, because LNQ has already
converged inside it (Prop. 4.1). So `moved` here is a cleaner measurement of the
escape than it is when FlexNu starts from SqueezeLLM, where the two effects mix.

THE RISK, STATED UP FRONT
-------------------------
This is structurally the staging failure FlexNu's own Section 3.5 warns about:
starting delta2 = 0 on a grid already fitted to W is "precisely the
nearest-neighbour point the divisor exists to escape", and full staging collapsed
the effect from 79.2% to 13.1% in their synthetic ablation.

LNQ's output is a much stronger nearest-codeword fixed point than the k-means
grid that produced that collapse, so the risk is if anything higher here. Two
mitigations, both exposed as options:

  * `delta_init_noise` (default 0.0): initialise delta2 ~ N(0, sigma) instead of
    exactly 0, so the divisor does not start exactly at the fixed point. Try
    1e-3 to 1e-2 if the escape does not fire.
  * `lr_cb` can be raised here more safely than in the standalone case, because
    the codebook starts at LNQ's *optimal* one (Eq. 9) rather than at k-means --
    there is less for it to gain by chasing W, and therefore less reason for it
    to drag the thresholds out from under the divisor.

If `moved` comes out at ~0% while the same settings give ~5% starting from
SqueezeLLM, the staging trap is real on real models and that is itself a result
worth recording.

ERROR ACCOUNTING
----------------
Three objective values are reported per module, all on LNQ's exact scale
(`objective_lnq_scale`, i.e. mean over groups), so they are directly comparable
to LNQ's own log lines:

    obj_init  -- SqueezeLLM init, nearest-codeword
    obj_lnq   -- after LNQ converges          <- the number to beat
    obj_final -- after FlexNu on top

and two reductions:

    vs_lnq    = (obj_final - obj_lnq) / obj_lnq     NEGATIVE = FlexNu improved
    vs_init   = (obj_final - obj_init) / obj_init   total, both stages combined

`vs_lnq` is the one that matters. `vs_init` is context: if LNQ alone already
took 90% of the total, a small `vs_lnq` is less disappointing than it looks.
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import torch

from .layerwise_flexnu import (
    _optimize_rows,
    objective_lnq_scale,
)


def train_lnq_flexnu(
    W: np.ndarray,               # [out, in]
    init_labels: np.ndarray,     # [out, in]   SqueezeLLM
    init_centroids: np.ndarray,  # [out, K]    SqueezeLLM
    H: np.ndarray,               # [g, in, in]
    *,
    # ---- LNQ stage ----
    num_iterations: int = 3,
    cd_cycles: int = 4,
    # ---- FlexNu stage ----
    iters: int = 300,
    lr_scale: float = 3e-3,
    lr_cb: float = 1e-5,
    tau_frac: float = 0.5,
    use_delta3: bool = True,
    freeze_codebook: bool = False,
    freeze_scale: bool = False,
    stage_frac: float = 0.0,
    eval_every: int = 1,
    row_block: int = 64,
    lambda_s2: float = 0.0,
    damp_hessian: bool = True,
    delta_init_noise: float = 0.0,
    signed_g: bool = False,
    signed_eps: float = 0.05,
    lambda_tv: float = 0.0,
    skip_lnq: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """LNQ to convergence, then FlexNu from LNQ's solution.

    Same signature and return as train_least_squares / train_flexnu, so it drops
    into seed_layer behind a one-line branch.
    """
    device = torch.device("cuda")
    t_start = time.time()

    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    Ht_raw = torch.tensor(H, dtype=torch.float32, device=device)
    out_dim, din = Wt.shape
    g = Ht_raw.shape[0]
    K = init_centroids.shape[-1]
    assert out_dim % g == 0, f"out_dim {out_dim} not divisible by g {g}"
    group_size = out_dim // g

    # ---- baseline: SqueezeLLM init, for the vs_init figure ------------------
    C_init = torch.tensor(init_centroids, dtype=torch.float32,
                          device=device).reshape(out_dim, K)
    L_init = torch.tensor(init_labels, dtype=torch.long,
                          device=device).reshape(out_dim, din)
    obj_init = objective_lnq_scale(Wt, Ht_raw, L_init, C_init)

    # ------------------------------------------------------------------ #
    # STAGE 1 -- LNQ
    # ------------------------------------------------------------------ #
    if skip_lnq:
        lnq_labels, lnq_C = init_labels, init_centroids
        lnq_log = {}
        t_lnq = 0.0
        logging.info("[lnq+flexnu] stage 1 SKIPPED (skip_lnq=True)")
    else:
        # Imported here, not at module scope: layerwise_quantize imports
        # THIS module, so a top-level import back into it is circular.
        # By call time layerwise_quantize is fully initialised.
        from .layerwise_quantize import train_least_squares

        t0 = time.time()
        lnq_labels, lnq_C, lnq_log = train_least_squares(
            W, init_labels, init_centroids, H,
            num_iterations=num_iterations, cd_cycles=cd_cycles,
        )
        t_lnq = time.time() - t0

    L_lnq = torch.tensor(np.asarray(lnq_labels), dtype=torch.long,
                         device=device).reshape(out_dim, din)
    C_lnq = torch.tensor(np.asarray(lnq_C, dtype=np.float32),
                         device=device).reshape(out_dim, K)
    obj_lnq = objective_lnq_scale(Wt, Ht_raw, L_lnq, C_lnq)

    logging.info(
        f"[lnq+flexnu] stage 1 (LNQ): obj {obj_init:.5f} -> {obj_lnq:.5f}  "
        f"({100.0 * (obj_lnq - obj_init) / max(obj_init, 1e-12):+.2f}% vs init) "
        f"| {t_lnq:.1f}s"
    )

    # ------------------------------------------------------------------ #
    # STAGE 2 -- FlexNu, initialised from LNQ
    # ------------------------------------------------------------------ #
    # Ridge, as in train_flexnu. FlexNu only uses H in a matmul so PSD suffices,
    # but a semidefinite H has null directions the divisor will wander into.
    Ht = Ht_raw.clone()
    if damp_hessian:
        diag = torch.arange(din, device=device)
        for i in range(g):
            Ht[i, diag, diag] += 1e-5 * torch.diagonal(Ht[i]).mean()

    labels_out = torch.empty(out_dim, din, dtype=torch.long, device=device)
    C_out = torch.empty(out_dim, K, dtype=torch.float32, device=device)

    t0 = time.time()
    rb = max(1, min(int(row_block), group_size))
    for k in range(g):
        H_g = Ht[k]
        lo = k * group_size
        for r0 in range(lo, lo + group_size, rb):
            r1 = min(r0 + rb, lo + group_size)
            cb, idx, _, _ = _optimize_rows(
                Wt[r0:r1], H_g,
                L_lnq[r0:r1],          # LNQ's CD-optimised assignments
                C_lnq[r0:r1],          # LNQ's closed-form-optimal codebook
                # Honour BOTH. Without use_init_labels the solver re-derives
                # nearest-codeword, discarding the ~20% non-nearest choices that
                # coordinate descent bought -- which starts stage 2 strictly
                # worse than LNQ's committed solution.
                use_init_labels=True,
                delta_init_noise=delta_init_noise,
                signed_g=signed_g, signed_eps=signed_eps, lambda_tv=lambda_tv,
                iters=iters, lr_scale=lr_scale, lr_cb=lr_cb,
                tau_frac=tau_frac, use_delta3=use_delta3,
                freeze_codebook=freeze_codebook, freeze_scale=freeze_scale,
                stage_frac=stage_frac, eval_every=eval_every,
                lambda_s2=lambda_s2,
            )
            C_out[r0:r1] = cb
            labels_out[r0:r1] = idx
            del cb, idx
        torch.cuda.empty_cache()
    t_flex = time.time() - t0

    obj_final = objective_lnq_scale(Wt, Ht_raw, labels_out, C_out)

    # ---- diagnostics --------------------------------------------------------
    with torch.no_grad():
        th = 0.5 * (C_out[:, 1:] + C_out[:, :-1])
        nn_idx = torch.searchsorted(th.contiguous(),
                                    Wt.contiguous()).clamp_(0, K - 1)
        moved = (nn_idx != labels_out).float().mean().item()
        # How far did FlexNu move from LNQ's assignment specifically?
        changed_vs_lnq = (labels_out != L_lnq).float().mean().item()

    vs_lnq = (obj_final - obj_lnq) / max(obj_lnq, 1e-12)
    vs_init = (obj_final - obj_init) / max(obj_init, 1e-12)
    lnq_share = ((obj_init - obj_lnq) / max(obj_init - obj_final, 1e-12)
                 if obj_init > obj_final else float("nan"))

    verdict = "WIN " if vs_lnq < 0 else "lose"
    logging.info(
        f"[lnq+flexnu] stage 2 (FlexNu): obj {obj_lnq:.5f} -> {obj_final:.5f}  "
        f"vs_lnq={100.0 * vs_lnq:+.2f}% {verdict} | "
        f"moved={100.0 * moved:.2f}% (vs LNQ assign: {100.0 * changed_vs_lnq:.2f}%) "
        f"| {t_flex:.1f}s"
    )
    logging.info(
        f"[lnq+flexnu] TOTAL: init {obj_init:.5f} -> final {obj_final:.5f}  "
        f"vs_init={100.0 * vs_init:+.2f}%  "
        f"(LNQ contributed {100.0 * lnq_share:.0f}% of the total reduction) "
        f"| {time.time() - t_start:.1f}s"
    )

    if moved < 1e-4:
        logging.warning(
            "[lnq+flexnu] moved=0% -- the divisor never escaped LNQ's "
            "nearest-codeword solution. This is the Section 3.5 staging trap: "
            "delta2=0 on an already-fitted grid. Try --lnqflex_delta_noise 1e-2, "
            "or compare against a from-SqueezeLLM FlexNu run to confirm."
        )

    log_dict = {
        "objective": [obj_init, obj_lnq, obj_final],
        "iteration": [0, 1, 2],
        "stage": ["init", "lnq", "flexnu"],
        "obj_init": obj_init,
        "obj_lnq": obj_lnq,
        "obj_final": obj_final,
        "vs_lnq_rel": vs_lnq,
        "vs_init_rel": vs_init,
        "lnq_share_of_reduction": lnq_share,
        "moved_frac": moved,
        "changed_vs_lnq_frac": changed_vs_lnq,
        "time_lnq_s": t_lnq,
        "time_flexnu_s": t_flex,
        "lnq_log": lnq_log,
    }

    labels_np = labels_out.detach().cpu().numpy().astype(np.uint8)
    C_np = C_out.detach().cpu().numpy().astype(np.float32)
    del Wt, Ht, Ht_raw, C_out, labels_out, L_lnq, C_lnq
    torch.cuda.empty_cache()

    return labels_np, C_np, log_dict