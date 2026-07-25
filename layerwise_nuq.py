"""
Uniform LNQ + GuidedQuant solver.
========================================================================

Drop-in sibling of `train_least_squares` (LNQ) in layerwise_quantize.py, but
for a UNIFORM affine grid instead of a free non-uniform codebook.

Key differences vs LNQ (non-uniform):

  * NO SqueezeLLM init. The codebook is not free, so there is nothing to seed
    from an external k-means. We build the uniform grid directly from W with an
    H-weighted MSE scale search (see `_init_uniform`). `init_labels` and
    `init_centroids` passed in by the caller are IGNORED.

  * The codebook C is CONSTRAINED to an affine lattice per output channel j:

        C[j, q] = s_j * level[q] + z_j            (asymmetric)
        C[j, q] = s_j * level[q]                   (symmetric,  z_j = 0)

    where `level` is the FIXED integer grid  {qmin, ..., qmax},
    m = 2**bit levels.  Only (s_j, z_j) are optimized -- 1 or 2 dof per row,
    replacing LNQ's m free codebook values.

  * update_C  -->  update_scale : the free least-squares codebook solve
        c* = (P^T H P)^-1 P^T H w        (LNQ Eq. 9)
    collapses to the exact minimizer over the affine params. Stacking
        A_j = [ q_j , 1 ]   (asymmetric)   or   A_j = [ q_j ]  (symmetric),
        [s_j, z_j]^T = (A_j^T H A_j)^-1 A_j^T H w_j.
    This is Eq. (9) restricted to a 1- or 2-column basis.

  * update_P (the CD assignment sweep) is REUSED UNCHANGED from LNQ. Because we
    always materialize the current affine grid into a dense `C` of shape
    (output_dim, m) before calling update_P, the coordinate-wise closed-form CD
    update (with precompute + lazy batch-updates) rounds to the correct uniform
    levels automatically -- "nearest of m codewords" == "nearest uniform level"
    when the m codewords ARE the uniform levels. No kernel edit needed.

GuidedQuant enters ONLY through H (the saliency-weighted group Hessian
H_k = X^T diag(s_k) X). Both update_P and update_scale consume H, so they are
saliency-weighted automatically; this file is unaware of it.

Output contract is identical to LNQ: returns (labels, C, log_dict) with
  labels : (output_dim, input_dim)  uint-ish int assignments into the grid
  C      : (output_dim, m)           the *materialized* affine grid (fp32)
so packing / dequant / eval are untouched.
"""

import logging
import time
import numpy as np
import torch

# Reuse LNQ's exact CD assignment sweep and objective -- do NOT reimplement.
from .layerwise_quantize import update_P, objective_function


# --------------------------------------------------------------------------- #
#  Grid helpers
# --------------------------------------------------------------------------- #
def _levels(bit: int, symmetric: bool, device, dtype=torch.float32) -> torch.Tensor:
    """
    Fixed integer levels of the uniform grid, shape (m,), m = 2**bit.

    symmetric:  {-2^(b-1)+1, ..., 2^(b-1)-1, ...}  centered, includes 0
                (we use the range [-(2^(b-1)-1), 2^(b-1)-1] padded to m entries
                 by including -2^(b-1); standard signed range)
    asymmetric: {0, 1, ..., m-1}
    """
    m = 2 ** bit
    if symmetric:
        qmin = -(2 ** (bit - 1))
        qmax = 2 ** (bit - 1) - 1
        lv = torch.arange(qmin, qmax + 1, device=device, dtype=dtype)
    else:
        lv = torch.arange(0, m, device=device, dtype=dtype)
    assert lv.numel() == m, (lv.numel(), m)
    return lv


def _materialize_C(level: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Build dense per-channel codebook from affine params.
      level : (m,)
      s     : (output_dim, 1)
      z     : (output_dim, 1)
    returns C : (output_dim, m),   C[j,q] = s_j * level[q] + z_j
    """
    return s * level.unsqueeze(0) + z  # (output_dim, m)


# --------------------------------------------------------------------------- #
#  Uniform init  (replaces SqueezeLLM init)  -- H-weighted MSE scale search
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _init_uniform(
    W: torch.Tensor,          # (output_dim, input_dim)  on cuda
    H: torch.Tensor,          # (num_groups, input_dim, input_dim) on cuda
    level: torch.Tensor,      # (m,)
    symmetric: bool,
    n_scale_grid: int = 24,
    alpha_min: float = 0.55,
    alpha_max: float = 1.0,
):
    """
    Per output channel, pick (s_j, z_j) minimizing the H-weighted reconstruction
    error over a small grid of clipping ratios alpha, then round to get labels.
    Returns (labels, s, z, C) with s,z shape (output_dim,1) and C materialized.
    """
    device = W.device
    out_dim, in_dim = W.shape
    num_groups = H.shape[0]
    group_size = out_dim // num_groups
    lvmin, lvmax = level.min(), level.max()

    if symmetric:
        wmax = W.abs().amax(dim=1, keepdim=True)                  # (out,1)
        base_s = wmax / lvmax
        z_fixed = torch.zeros_like(base_s)
    else:
        wmin = W.amin(dim=1, keepdim=True)
        wmax = W.amax(dim=1, keepdim=True)
        base_s = (wmax - wmin) / (lvmax - lvmin)
        z_fixed = wmin - lvmin * base_s  # so that level=lvmin maps to wmin

    base_s = base_s.clamp_min(1e-12)

    alphas = torch.linspace(alpha_max, alpha_min, n_scale_grid, device=device)

    best_err = torch.full((out_dim,), float("inf"), device=device)
    best_s = base_s.clone()
    best_z = z_fixed.clone()

    # diag(H) per group, broadcast to rows -- cheap proxy inside the search loop
    # (full H-weighted error is used for the *final* selection below).
    for a in alphas:
        s = base_s * a
        if symmetric:
            z = torch.zeros_like(s)
        else:
            # keep zero-point consistent with clipped range
            z = wmin - lvmin * s
        # round to grid
        q = torch.clamp(torch.round((W - z) / s), lvmin, lvmax)   # (out,in)
        W_hat = s * q + z
        dW = (W_hat - W).reshape(num_groups, group_size, in_dim)
        # H-weighted per-row error:  sum_i dW_i^T H dW_i  (diagonal-of-groups)
        err = torch.einsum('nij,njk,nik->ni', dW, H, dW).reshape(out_dim)
        improved = err < best_err
        best_err = torch.where(improved, err, best_err)
        best_s = torch.where(improved.unsqueeze(1), s, best_s)
        best_z = torch.where(improved.unsqueeze(1), z, best_z)

    # final labels under the chosen (s,z)
    q = torch.clamp(torch.round((W - best_z) / best_s), lvmin, lvmax)   # (out,in)
    labels = (q - lvmin).round().long()                                # index into level[]
    C = _materialize_C(level, best_s, best_z)
    return labels, best_s, best_z, C


# --------------------------------------------------------------------------- #
#  Scale/zero-point update  (replaces update_C)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def update_scale(
    W: torch.Tensor,        # (output_dim, input_dim)  cuda
    H: torch.Tensor,        # (num_groups, input_dim, input_dim) cuda
    labels: torch.Tensor,   # (output_dim, input_dim)  long, index into level[]
    level: torch.Tensor,    # (m,)
    symmetric: bool,
    sub_channel_size: int = 64,
):
    """
    Exact minimizer of  sum_i (s q_i + z - w_i)^T H (s q_i + z - w_i)  over
    (s_j, z_j) per output channel, using the same Cholesky-reduced formulation
    as LNQ's update_C:  with H = L L^T, reduced_X = L^T, solve least squares on
    A_j = [reduced_X q_j , reduced_X 1]  vs  reduced_X w_j.

    Returns (s, z) each (output_dim, 1) and the materialized C (output_dim, m).
    """
    device = W.device
    out_dim, in_dim = W.shape
    num_groups = H.shape[0]
    group_size = out_dim // num_groups

    # integer code values q_ij = level[labels_ij]
    q = level[labels]                       # (out, in)  float

    # Cholesky per group  (H = L L^T), reduced_X = L^T   -> whitening
    L = torch.empty_like(H)
    for g in range(num_groups):
        L[g] = torch.linalg.cholesky(H[g])
    reduced_X = L.transpose(-2, -1)         # (num_groups, in, in)

    s_out = torch.empty((out_dim, 1), device=device)
    z_out = torch.empty((out_dim, 1), device=device)
    ncol = 1 if symmetric else 2
    ones_col = torch.ones((in_dim,), device=device)

    for st in range(0, out_dim, sub_channel_size):
        g = st // group_size
        Xr = reduced_X[g]                   # (in, in)   L^T for this group
        en = min(st + sub_channel_size, out_dim)
        qb = q[st:en]                        # (bsz, in)
        wb = W[st:en]                        # (bsz, in)
        bsz = en - st

        # b = Xr @ w_j   -> (bsz, in)   (whitened target)
        b = torch.einsum('ij,bj->bi', Xr, wb)          # (bsz, in)
        # a_scale = Xr @ q_j -> (bsz, in)
        a_scale = torch.einsum('ij,bj->bi', Xr, qb)    # (bsz, in)

        if symmetric:
            # s* = <a_scale, b> / <a_scale, a_scale>
            num = (a_scale * b).sum(dim=1)
            den = (a_scale * a_scale).sum(dim=1).clamp_min(1e-12)
            s = (num / den).unsqueeze(1)               # (bsz,1)
            s_out[st:en] = s
            z_out[st:en] = 0.0
        else:
            # a_z = Xr @ 1  (same for all rows in this group)
            a_z = (Xr @ ones_col).unsqueeze(0).expand(bsz, -1)   # (bsz, in)
            # 2x2 normal equations per row:  [ <as,as> <as,az> ; . <az,az> ]
            saa = (a_scale * a_scale).sum(dim=1)
            saz = (a_scale * a_z).sum(dim=1)
            zaa = (a_z * a_z).sum(dim=1)
            bs = (a_scale * b).sum(dim=1)
            bz = (a_z * b).sum(dim=1)
            det = (saa * zaa - saz * saz).clamp_min(1e-12)
            s = (zaa * bs - saz * bz) / det
            z = (saa * bz - saz * bs) / det
            s_out[st:en] = s.unsqueeze(1)
            z_out[st:en] = z.unsqueeze(1)

    s_out = s_out.clamp_min(1e-12)
    C = _materialize_C(level, s_out, z_out)
    return s_out, z_out, C


# --------------------------------------------------------------------------- #
#  Main alternating loop  (mirrors train_least_squares)
# --------------------------------------------------------------------------- #
def train_uniform(
    W: np.ndarray,               # (output_dim, input_dim)
    init_labels: np.ndarray,     # IGNORED (kept for signature compatibility)
    init_centroids: np.ndarray,  # IGNORED
    H: np.ndarray,               # (num_groups, input_dim, input_dim)
    seed_bit: int,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    symmetric: bool = True,
):
    device = torch.device("cuda")

    W = torch.tensor(W, dtype=torch.float32, device=device)
    H = torch.tensor(H, dtype=torch.float32, device=device)

    # --- PD damping (same as LNQ) ---
    diag = torch.arange(H.shape[1], device=device)
    for i in range(H.shape[0]):
        avg_diag = torch.mean(torch.diag(H[i]))
        damp, prev_damp = 1e-5, 0.
        while True:
            try:
                torch.linalg.cholesky(H[i])
                logging.info(f"{i+1}-th H is PD, dampening factor={prev_damp:.2e}")
                break
            except Exception as e:
                logging.info(f"{i+1}-th H not PD, dampening factor={damp:.2e}")
                H[i, diag, diag] += (damp - prev_damp) * avg_diag
                prev_damp = damp
                damp *= 10
                if damp > 1e0:
                    raise RuntimeError("H could not be made PD")

    level = _levels(seed_bit, symmetric, device)     # (m,)

    # --- Uniform init (replaces SqueezeLLM) ---
    labels, s, z, C = _init_uniform(W, H, level, symmetric)
    labels = labels.to(device)

    best_obj = objective_function(W, H, labels.cpu(), C.cpu()).item()
    best_labels, best_C = labels.detach().cpu().clone(), C.detach().cpu().clone()
    logging.info(f"[uniform] Initial objective: {best_obj:.6f}")

    log_dict = {"objective": [best_obj], "iteration": [0]}

    for iteration in range(num_iterations):
        t0 = time.time()

        # ----- Update P (CD assignment sweep) : LNQ's update_P, UNCHANGED -----
        if iteration > 0:
            # update_P expects C on the current grid; pass materialized C.
            labels = update_P(W, H, labels, C, cd_cycles=cd_cycles)

        obj_p = objective_function(W, H, labels.cpu(), C.cpu()).item()
        logging.info(f"[uniform] Iter {iteration+1} (P): {obj_p:.4f}")
        log_dict["objective"].append(obj_p)
        log_dict["iteration"].append(iteration + 1)

        # ----- Update scale/zero-point (replaces update_C) -----
        s, z, C = update_scale(W, H, labels, level, symmetric)

        obj_c = objective_function(W, H, labels.cpu(), C.cpu()).item()
        log_dict["objective"].append(obj_c)
        log_dict["iteration"].append(iteration + 1)

        if obj_c < best_obj:
            best_obj = obj_c
            best_labels = labels.detach().cpu().clone()
            best_C = C.detach().cpu().clone()
            logging.info(f"[uniform] Iter {iteration+1} (S): {obj_c:.4f} | improved")
        else:
            logging.info(f"[uniform] Iter {iteration+1} (S): {obj_c:.4f} | not improved, stop")
            labels, C = best_labels.to(device), best_C.to(device)
            break

        logging.info(f"[uniform] Iter {iteration+1}/{num_iterations} "
                     f"done in {time.time()-t0:.2f}s")

    labels = best_labels.detach().cpu().numpy()   # (out, in) int, index into level[]
    C = best_C.detach().cpu().numpy().astype(np.float32)
    return labels, C, log_dict