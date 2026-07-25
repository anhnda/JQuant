"""
Staged B-opt on top of LNQ + GuidedQuant.

Strengthens the assignment fixed point of LNQ from 1-opt to B-opt, at <= 2x the
current CD cost, without touching the codebook solve (Eq. 9) and without breaking
monotonicity. Implements the spec `Staged B-opt on top of LNQ + GuidedQuant`.

Layout note
-----------
The rest of this codebase carries weights as W[out, in] and the Hessian as
H[g, in, in] (see layerwise_quantize.py). The spec is written for W[d, dout]
(coordinates on axis 0, channels on axis 1). This module works in the CODEBASE
layout: a "channel" is a row (out index), a "coordinate" is a column (in index),
and everything runs per group, since H_g is shared across all rows of a group
(the structural gift of GuidedQuant that makes the pass affordable, spec 2.2).

Within a group of R rows sharing H_g (in x in):

    W_g   : (R, d)      FP weights for this group's rows
    idx   : (R, d)      current assignment into the per-channel codebook  (== P)
    cb    : (R, m)      per-channel codebook
    Wq    : (R, d)      current quantised weight, gather(cb, idx)
    E     : (R, d)      residual  Wq - W_g
    G     : (R, d)      E @ H_g   (note: rows are channels, so G[r] = H_g @ E[r])
    Hdiag : (d,)        diag(H_g)          -- shared across rows
    H_g   : (d, d)      raw group Hessian   -- shared across rows

Exact single-flip gain for row r, coordinate i, residual change delta:
    dE = 2 delta G[r,i] + delta^2 Hdiag[i]
Exact group gain on coordinate set T with residual change vector d_T:
    dE_T = 2 <d_T, G[r,T]> + d_T^T H_g[T,T] d_T
This holds for arbitrary d_T (any levels, any jump). It is the only formula used.

Monotonicity: every accepted move is committed only if its EXACT dE < 0 (below a
noise floor). G is refreshed from scratch (G = E @ H_g) after each batch of
commits, so no move is ever evaluated against a stale residual.

Public entry point
------------------
    bopt_refine(W, H, labels, C, ...) -> (labels, C, log)

W[out,in] np/torch, H[g,in,in], labels[out,in], C[out,K]. Returns refined labels
and the re-solved codebook C, plus an instrumentation dict (spec 2.6).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _as_t(x, dtype, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


def _gather_cb(cb: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """cb:(R,m), idx:(R,d) -> Wq:(R,d)."""
    return torch.gather(cb, 1, idx)


def _group_energy(E: torch.Tensor, H_g: torch.Tensor) -> torch.Tensor:
    """Exact per-row energy e^T H e. E:(R,d), H_g:(d,d) -> (R,)."""
    return torch.einsum('ri,ij,rj->r', E, H_g, E)


# --------------------------------------------------------------------------- #
# noise-floor constants (spec 2.4)
#   kappa_B = kappa_1 * sqrt(log N_B / log N_1)
# --------------------------------------------------------------------------- #
def _kappa(kappa1: float, log_n1: float, log_nB: float) -> float:
    return kappa1 * math.sqrt(max(log_nB, 1e-9) / max(log_n1, 1e-9))


def _log_stage(k: int, name: str, s: dict, f_before: float, f_after: float,
               d: int, kappa: float,
               mse_hold_before: Optional[float] = None,
               mse_hold_after: Optional[float] = None,
               R: int = 1) -> None:
    """Emit the §2.6 instrumentation for one stage of one group.

    f_before / f_after are MEAN per-channel energies (for the readable % drop).
    s['dE_released'] is a SUM over channels, so to express it as a fraction of
    the objective we compare it against the TOTAL energy (mean * R).

    Item 5 (calib-vs-holdout) is the one that matters: calibration energy MUST
    drop (monotonicity). If held-out energy does not follow, the stage is
    harvesting estimation noise and kappa should be raised, not celebrated.
    """
    rel = (f_before - f_after) / max(f_before, 1e-12)            # mean-scale drop
    f_tot = max(f_before * R, 1e-12)                             # total energy
    rel_released = s["dE_released"] / f_tot                      # committed frac
    msg = (
        f"[bopt][g{k}] {name}: "
        f"accept={s['n_accept']} "                          # item 1
        f"chans={100*s['frac_channels']:.1f}% "             # item 2
        f"dE_released={s['dE_released']:.4e} "
        f"({100*rel_released:.3f}% of f, obj drop {100*rel:.3f}%) "  # item 3
        f"kappa={kappa:.2f}"
    )
    if "jump_hist" in s:                                    # item 4
        jh = s["jump_hist"]
        multi = sum(jh[2:]) if len(jh) > 2 else 0
        tot = max(sum(jh), 1)
        msg += (f" | level-jumps={jh} "
                f"(multi-level={100*multi/tot:.1f}% <- LNQ-only, TFIC cannot)")
    if "cluster_size_hist" in s:                            # item 4 (B=3)
        msg += f" | cluster_sizes={s['cluster_size_hist']}"
    if "depth_hist" in s:                                   # item 4 (chains)
        dh = s["depth_hist"]
        deep = sum(i * c for i, c in enumerate(dh))
        msg += f" | chain_depths(hist)={dh}"
    if "funnel" in s:                                       # why accept is what it is
        fn = s["funnel"]
        msg += (f" | funnel: cand={fn['candidates']} "
                f"-> prop={fn['proposals']} -> below_thr={fn['below_thresh']} "
                f"-> accept={fn['accepted']}")
        if s["n_accept"] == 0:
            if fn["candidates"] == 0:
                msg += "  [no coords near a barrier -- c_cand too tight or truly 1-opt-deep]"
            elif fn["proposals"] == 0:
                msg += "  [candidates found but no synergistic pairs -- couplings weak]"
            elif fn["below_thresh"] == 0:
                msg += "  [pairs exist but none beat kappa*tau -- lower kappa1 to probe]"
    logging.info(msg)
    # item 5: calibration vs holdout — the noise-harvesting check
    if mse_hold_before is not None and mse_hold_after is not None:
        rel_h = (mse_hold_before - mse_hold_after) / max(mse_hold_before, 1e-12)
        follow = rel_h / rel if rel > 1e-12 else float("nan")
        flag = ""
        if rel > 1e-9 and rel_h < 0.5 * rel:
            flag = "  <-- HOLDOUT LAGS: likely noise, RAISE kappa"
        logging.info(
            f"[bopt][g{k}] {name} item5: calib -{100*rel:.3f}%  "
            f"holdout -{100*rel_h:.3f}%  (holdout/calib={follow:.2f}){flag}"
        )


# =========================================================================== #
# Prerequisites (spec 1)  -- confound controls, all cheap
# =========================================================================== #
@torch.no_grad()
def cd_to_fixed_point(
    W_g: torch.Tensor,   # (R, d)
    H_g: torch.Tensor,   # (d, d)
    Hdiag: torch.Tensor, # (d,)
    cb: torch.Tensor,    # (R, m)
    idx: torch.Tensor,   # (R, d)
    max_sweeps: int = 20,
    block: int = 128,
) -> Tuple[torch.Tensor, int, int]:
    """Run coordinate descent (any-level, exact) to an actual 1-opt fixed point.

    Mirrors update_P's arithmetic but keeps sweeping until no assignment moves,
    so that any B-opt gain measured afterwards is a barrier, not CD truncation
    (spec 1.1). Returns (idx, n_flips_last_sweep, sweeps_run, flip_trend,
    polished_1opt). polished_1opt is True iff the returned point is a genuine
    1-opt (always True on clean convergence; True after greedy polish otherwise).
    """
    R, d = W_g.shape
    Wq = _gather_cb(cb, idx)                       # (R,d)
    # Normalised H for the CD linear term, exactly as update_P does.
    Hn = H_g / Hdiag.view(1, -1)                   # divide columns by diag
    n_flips = -1
    sweeps = 0
    flip_trend = []                                # per-sweep flip counts
    # Guard against oscillation: keep the LOWEST-energy assignment seen, not the
    # last (which may be a mid-cycle iterate). CD on an indefinite H is not
    # guaranteed to converge; committing the best iterate keeps monotonicity.
    best_E = (Wq - W_g)
    best_energy = _group_energy(best_E, H_g).sum().item()
    best_idx = idx.clone()
    for sweep in range(max_sweeps):
        sweeps += 1
        E = Wq - W_g                               # (R,d)
        # B[:, j] = sum_{i<j} E[:,i] Hn[i,j]  (strictly-lower accumulation)
        B = E @ torch.tril(Hn, diagonal=-1)        # (R,d)
        moved = 0
        for s in range(0, d, block):
            e = min(s + block, d)
            for j in range(s, e):
                sol = W_g[:, j:j+1] - B[:, j:j+1]          # (R,1)
                dist = (sol - cb).abs()                    # (R,m)
                new = dist.argmin(dim=1)                   # (R,)
                old = idx[:, j]
                moved += int((new != old).sum().item())
                idx[:, j] = new
                newq = torch.gather(cb, 1, new.unsqueeze(1)).squeeze(1)
                Wq[:, j] = newq
                if j < e - 1:
                    B[:, j+1:e] += (newq - W_g[:, j]).unsqueeze(1) * Hn[j, j+1:e].unsqueeze(0)
            if e < d:
                B[:, e:] += (Wq[:, s:e] - W_g[:, s:e]) @ Hn[s:e, e:]
        n_flips = moved
        flip_trend.append(moved)
        # keep best-energy iterate
        cur_energy = _group_energy(Wq - W_g, H_g).sum().item()
        if cur_energy < best_energy:
            best_energy = cur_energy
            best_idx = idx.clone()
        if moved == 0:
            break
    # If CD never hit 0 flips, return the best iterate seen (not the last), so
    # downstream energy is a true lower bound of what CD reached.
    if n_flips != 0:
        idx = best_idx
        idx = _greedy_1opt(W_g, H_g, Hdiag, cb, idx, max_sweeps)
        # verify the polish actually reached 1-opt (it should, being greedy)
        E = _gather_cb(cb, idx) - W_g
        a, _, _ = _alt_gains(W_g, E, E @ H_g, Hdiag, cb, idx)
        polished_1opt = bool((a >= -1e-6).all())
    else:
        polished_1opt = True
    return idx, n_flips, sweeps, flip_trend, polished_1opt


@torch.no_grad()
def _greedy_1opt(W_g, H_g, Hdiag, cb, idx, max_iters=20):
    """Strictly-greedy single-flip descent (best flip per channel per iter).

    Unlike fixed-order cyclic CD this cannot oscillate: each iteration strictly
    lowers exact energy or stops. Used only to polish an oscillating CD result
    into a true 1-opt point. Returns idx (modified copy semantics).
    """
    idx = idx.clone()
    for _ in range(max_iters):
        E = _gather_cb(cb, idx) - W_g
        G = E @ H_g
        a, dstar, q = _alt_gains(W_g, E, G, Hdiag, cb, idx)   # a<0 => improving
        gbest, jbest = a.min(dim=1)                           # best flip per chan
        take = gbest < -1e-9
        if not bool(take.any()):
            break
        rows = take.nonzero(as_tuple=True)[0]
        idx[rows, jbest[rows]] = q[rows, jbest[rows]]
    return idx


@torch.no_grad()
def empty_codeword_audit(idx: torch.Tensor, m: int) -> Tuple[torch.Tensor, float]:
    """Spec 1.2. Returns (dead_mask:(R,m) bool, dead_rate)."""
    R, d = idx.shape
    counts = torch.zeros(R, m, device=idx.device, dtype=torch.long)
    counts.scatter_add_(1, idx, torch.ones_like(idx))
    dead = counts == 0
    return dead, dead.float().mean().item()


@torch.no_grad()
def reseed_dead_codewords(
    W_g: torch.Tensor, H_g: torch.Tensor, Hdiag: torch.Tensor,
    cb: torch.Tensor, idx: torch.Tensor, dead: torch.Tensor,
) -> torch.Tensor:
    """Spec 1.2. Reseed dead codewords at the weight with largest H_ii * e_i^2.

    Only the codebook `cb` is nudged; the caller re-runs the codebook solve
    (Eq. 9) + CD afterwards, so this is just a live restart of the dead level.
    """
    if not bool(dead.any()):
        return cb
    Wq = _gather_cb(cb, idx)
    e2 = (Wq - W_g).pow(2) * Hdiag.view(1, -1)     # (R,d) H_ii e_i^2
    rows = dead.any(dim=1).nonzero(as_tuple=True)[0]
    for r in rows.tolist():
        target = W_g[r, e2[r].argmax()].item()
        for c in dead[r].nonzero(as_tuple=True)[0].tolist():
            cb[r, c] = target
    return cb


# =========================================================================== #
# Stage 1 -- B = 2  (spec 2)
# =========================================================================== #
@torch.no_grad()
def _alt_gains(
    W_g: torch.Tensor, E: torch.Tensor, G: torch.Tensor,
    Hdiag: torch.Tensor, cb: torch.Tensor, idx: torch.Tensor,
    dout_chunk: int = 512,   # chunk over ROWS here (rows == channels)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spec 2.3 Step A. Best non-current level per coordinate (exact 1-flip).

    Returns
        a     : (R, d) best (most negative) single-flip gain; >=0 at 1-opt fp
        dstar : (R, d) residual change delta of that best flip
        q     : (R, d) chosen level index
    Chunked over rows to bound the (R,d,m) intermediate.
    """
    R, d = W_g.shape
    m = cb.shape[1]
    a = torch.empty(R, d, device=W_g.device)
    dstar = torch.empty(R, d, device=W_g.device)
    q = torch.empty(R, d, dtype=torch.long, device=W_g.device)
    for r0 in range(0, R, dout_chunk):
        r1 = min(r0 + dout_chunk, R)
        Wq = W_g[r0:r1] + E[r0:r1]                          # (r,d)
        delta = cb[r0:r1].unsqueeze(1) - Wq.unsqueeze(-1)   # (r,d,m)
        dE = (2 * delta * G[r0:r1].unsqueeze(-1)
              + delta.pow(2) * Hdiag.view(1, -1, 1))        # (r,d,m)
        dE.scatter_(2, idx[r0:r1].unsqueeze(-1), float('inf'))  # exclude current
        aa, qq = dE.min(dim=2)
        a[r0:r1] = aa
        q[r0:r1] = qq
        dstar[r0:r1] = torch.gather(delta, 2, qq.unsqueeze(-1)).squeeze(-1)
    return a, dstar, q


@torch.no_grad()
def _neighbour_lists(
    H_g: torch.Tensor, Hdiag: torch.Tensor, nu: int, rank_normalised: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spec 2.3 Step D. Once per group. Rank by correlation magnitude, but the
    returned H values are RAW H (that is what enters the energy).

    Returns  nb_idx:(d,nu) long,  H_nb:(d,nu) raw H[i, nb_idx[i]].
    """
    d = H_g.shape[0]
    s = Hdiag.clamp_min(1e-30).sqrt()
    if rank_normalised:
        Hn = H_g / (s.view(-1, 1) * s.view(1, -1))
    else:
        Hn = H_g.clone()
    Hn = Hn.clone()
    Hn.fill_diagonal_(0.0)
    nu = min(nu, d - 1)
    _, nb_idx = Hn.abs().topk(nu, dim=1)               # (d,nu)
    H_nb = torch.gather(H_g, 1, nb_idx)                # (d,nu) raw
    return nb_idx, H_nb


@torch.no_grad()
def _stage_pair_pass(
    W_g, H_g, Hdiag, G, cb, idx, E, nb_idx, H_nb,
    tau, kappa2, c_cand=8.0, top_p=200, row_chunk=512,
) -> Tuple[torch.Tensor, dict]:
    """Spec 2.3 Steps C,E,F,G. One B=2 pass. Mutates idx in place; returns the
    updated idx and a stats dict. Does NOT refresh G (caller does, Step H).

    Approximate proposals (synergy score), EXACT acceptance (m^2 enumeration).
    """
    R, d = W_g.shape
    m = cb.shape[1]
    nu = nb_idx.shape[1]
    dev = W_g.device

    # best single-flip proposal per coordinate
    a, dstar, qbest = _alt_gains(W_g, E, G, Hdiag, cb, idx, dout_chunk=row_chunk)

    n_accept = 0
    dE_released = 0.0
    jump_hist = torch.zeros(m, dtype=torch.long)   # level-jump distances
    frac_rows = torch.zeros(R, dtype=torch.bool)
    n_cand = 0            # coords passing the candidate mask
    n_prop = 0           # finite pair proposals enumerated
    n_below = 0          # proposals whose EXACT best gain beats -kappa*tau

    ar = torch.arange(d, device=dev)
    for r0 in range(0, R, row_chunk):
        r1 = min(r0 + row_chunk, R)
        C = r1 - r0
        a_c = a[r0:r1]                      # (C,d)
        dstar_c = dstar[r0:r1]              # (C,d)
        idx_c = idx[r0:r1]                  # (C,d)
        Wq_c = W_g[r0:r1] + E[r0:r1]        # (C,d)

        # candidate mask: pressed against a barrier (Step C)
        cand = a_c <= c_cand * tau         # (C,d)  a>=0 at fp; small a => tight
        n_cand += int(cand.sum())

        # pair synergy S[c, i, n] = -2 dstar[c,i] dstar[c,nb[i,n]] H_nb[i,n]  (Step E)
        # nb neighbour delta for each channel: (C, d, nu)
        d_nb = dstar_c[:, nb_idx]                                 # (C,d,nu)
        S = -2.0 * dstar_c.unsqueeze(-1) * d_nb * H_nb.unsqueeze(0)  # (C,d,nu)
        # mask: both endpoints candidates, and i<k dedupe
        cand_nb = cand[:, nb_idx]                                 # (C,d,nu)
        both = cand.unsqueeze(-1) & cand_nb
        dedupe = (nb_idx > ar.view(-1, 1)).unsqueeze(0)           # keep i<k only
        valid = both & dedupe
        S = S.masked_fill(~valid, float('-inf'))

        # take top_p pairs per channel by S (Step F: proposals only)
        Sflat = S.reshape(C, d * nu)
        P = min(top_p, Sflat.shape[1])
        top_val, top_lin = Sflat.topk(P, dim=1)                  # (C,P)
        i_idx = top_lin // nu                                     # (C,P) coord i
        n_sel = top_lin % nu
        k_idx = nb_idx[i_idx, n_sel]                              # (C,P) coord k
        valid_p = torch.isfinite(top_val)                        # (C,P)

        # EXACT m^2 enumeration for each proposed pair (Step F)
        # residual changes for all target levels at endpoints
        cb_c = cb[r0:r1]                                          # (C,m)
        # gather endpoint quantities
        cc = torch.arange(C, device=dev).unsqueeze(1).expand(C, P)
        Wqi = Wq_c[cc, i_idx]                                     # (C,P)
        Wqk = Wq_c[cc, k_idx]
        Gi = G[r0:r1][cc, i_idx]
        Gk = G[r0:r1][cc, k_idx]
        Hii = Hdiag[i_idx]                                       # (C,P)
        Hkk = Hdiag[k_idx]
        Hik = H_g[i_idx, k_idx]                                  # (C,P)
        lv = cb_c.unsqueeze(1).expand(C, P, m)                    # (C,P,m) codebook
        di = lv - Wqi.unsqueeze(-1)                               # (C,P,m)
        dk = lv - Wqk.unsqueeze(-1)                               # (C,P,m)
        dEi = 2 * di * Gi.unsqueeze(-1) + di.pow(2) * Hii.unsqueeze(-1)   # (C,P,m)
        dEk = 2 * dk * Gk.unsqueeze(-1) + dk.pow(2) * Hkk.unsqueeze(-1)   # (C,P,m)
        cross = 2 * Hik.unsqueeze(-1).unsqueeze(-1) * di.unsqueeze(-1) * dk.unsqueeze(-2)  # (C,P,m,m)
        dE_pair = dEi.unsqueeze(-1) + dEk.unsqueeze(-2) + cross   # (C,P,m,m)
        dE_flat = dE_pair.reshape(C, P, m * m)
        best_gain, best_lin = dE_flat.min(dim=2)                 # (C,P)
        li = best_lin // m                                       # level for i
        lk = best_lin % m                                        # level for k
        best_gain = best_gain.masked_fill(~valid_p, float('inf'))

        # Step G: accept ONLY the single best pair per channel this pass.
        # -------------------------------------------------------------------
        # Committing multiple disjoint pairs in the SAME channel is NOT safe
        # when H is dense: two pairs with disjoint support (a) and (b) still
        # interact through 2 * delta_a^T H delta_b, which does not vanish just
        # because the supports are disjoint. Two individually-improving pairs
        # can therefore make the union WORSE. The only exact-safe batch is:
        # one move per channel, all channels in parallel (channels are
        # independent objectives). For more moves, iterate the pass (caller's
        # restricted-VND loop) with a fresh G each time.
        thr = -kappa2 * tau
        n_prop += int(torch.isfinite(best_gain).sum())
        n_below += int(((best_gain < thr) & torch.isfinite(best_gain)).sum())
        # best proposal per channel
        gbest, pbest = best_gain.min(dim=1)                     # (C,), (C,)
        take = (gbest < thr) & torch.isfinite(gbest)           # (C,)
        rows_ok = take.nonzero(as_tuple=True)[0]
        for rr in rows_ok.tolist():
            gr = r0 + rr
            p = int(pbest[rr])
            ci, ck = int(i_idx[rr, p]), int(k_idx[rr, p])
            if ci == ck:
                continue
            old_i, old_k = int(idx[gr, ci]), int(idx[gr, ck])
            new_i, new_k = int(li[rr, p]), int(lk[rr, p])
            idx[gr, ci] = new_i
            idx[gr, ck] = new_k
            n_accept += 1
            dE_released += -float(gbest[rr])
            frac_rows[gr] = True
            jump_hist[abs(new_i - old_i)] += 1
            jump_hist[abs(new_k - old_k)] += 1

    stats = {
        "n_accept": n_accept,
        "frac_channels": frac_rows.float().mean().item(),
        "dE_released": dE_released,
        "jump_hist": jump_hist.tolist(),
        "funnel": {"candidates": n_cand, "proposals": n_prop,
                   "below_thresh": n_below, "accepted": n_accept},
    }
    return idx, stats


# =========================================================================== #
# Stage 2 -- B = 3  (spec 3): cluster growth + m^3 enumeration
# =========================================================================== #
@torch.no_grad()
def _stage_triple_pass(
    W_g, H_g, Hdiag, G, cb, idx, E, nb_idx, H_nb,
    tau, kappa3, c_cand=8.0, top_p=200, row_chunk=512,
) -> Tuple[torch.Tensor, dict]:
    """Spec 3.1. One B=3 pass. Grow a triple {i,k,l} from the best pair, then
    enumerate all m^3 level assignments exactly on the 3x3 submatrix H[T,T].

    For clarity and correctness this uses the "simpler variant" discipline
    (spec 2.3 Step G note): propose per channel, accept greedily-disjoint.
    """
    R, d = W_g.shape
    m = cb.shape[1]
    nu = nb_idx.shape[1]
    dev = W_g.device

    a, dstar, _ = _alt_gains(W_g, E, G, Hdiag, cb, idx, dout_chunk=row_chunk)

    n_accept = 0
    dE_released = 0.0
    size_hist = {2: 0, 3: 0}
    frac_rows = torch.zeros(R, dtype=torch.bool)
    ar = torch.arange(d, device=dev)
    thr = -kappa3 * tau

    for r0 in range(0, R, row_chunk):
        r1 = min(r0 + row_chunk, R)
        C = r1 - r0
        a_c = a[r0:r1]
        dstar_c = dstar[r0:r1]
        Wq_c = W_g[r0:r1] + E[r0:r1]
        cb_c = cb[r0:r1]
        G_c = G[r0:r1]
        cand = a_c <= c_cand * tau                                # (C,d)

        # best pair seed per channel (reuse synergy)
        d_nb = dstar_c[:, nb_idx]                                 # (C,d,nu)
        S = -2.0 * dstar_c.unsqueeze(-1) * d_nb * H_nb.unsqueeze(0)
        both = cand.unsqueeze(-1) & cand[:, nb_idx]
        dedupe = (nb_idx > ar.view(-1, 1)).unsqueeze(0)
        S = S.masked_fill(~(both & dedupe), float('-inf'))
        Sflat = S.reshape(C, d * nu)
        P = min(top_p, Sflat.shape[1])
        top_val, top_lin = Sflat.topk(P, dim=1)
        i_idx = top_lin // nu
        k_idx = nb_idx[i_idx, top_lin % nu]
        valid_p = torch.isfinite(top_val)

        # One cluster per channel per pass (see the dense-H interaction hazard
        # documented in _stage_pair_pass Step G). Iterate the pass for more.
        for c in range(C):
            gr = r0 + c
            order = torch.argsort(-top_val[c])       # best synergy first
            for pi in order.tolist():
                if not bool(valid_p[c, pi]):
                    break
                i = int(i_idx[c, pi]); k = int(k_idx[c, pi])
                if i == k:
                    continue
                # grow: third coord l maximising aggregate synergy with {i,k}
                cand_l = torch.cat([nb_idx[i], nb_idx[k]]).unique()
                cand_l = cand_l[cand[c, cand_l]]
                cand_l = cand_l[(cand_l != i) & (cand_l != k)]
                T = [i, k]
                if cand_l.numel() > 0:
                    d_i = dstar_c[c, i]; d_k = dstar_c[c, k]
                    d_l = dstar_c[c, cand_l]
                    s_il = (-2.0 * d_i * d_l * H_g[i, cand_l]).clamp_min(0)
                    s_kl = (-2.0 * d_k * d_l * H_g[k, cand_l]).clamp_min(0)
                    l = int(cand_l[(s_il + s_kl).argmax()])
                    T = [i, k, l]

                Ti = torch.tensor(T, device=dev)
                bsz = len(T)
                # exact m^bsz enumeration on submatrix H[T,T]
                Wq_T = Wq_c[c, Ti]                              # (bsz,)
                G_T = G_c[c, Ti]
                Hsub = H_g[Ti][:, Ti]                          # (bsz,bsz)
                lv = cb_c[c]                                   # (m,)
                # build delta grid (bsz axes of size m)
                grids = torch.meshgrid(*[torch.arange(m, device=dev)] * bsz,
                                       indexing='ij')
                lvl = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (bsz, m^bsz)
                dT = lv[lvl] - Wq_T.unsqueeze(1)               # (bsz, m^bsz)
                # dE = 2<dT,G_T> + dT^T Hsub dT   per column
                lin = 2 * (G_T.unsqueeze(1) * dT).sum(0)       # (m^bsz,)
                quad = torch.einsum('an,ab,bn->n', dT, Hsub, dT)
                dE = lin + quad
                best = int(dE.argmin())
                g_best = float(dE[best])
                if g_best >= thr:
                    continue
                # commit this one cluster, then move to the next channel
                for a_i, coord in enumerate(T):
                    idx[gr, coord] = int(lvl[a_i, best])
                n_accept += 1
                dE_released += -g_best
                frac_rows[gr] = True
                size_hist[len(T)] = size_hist.get(len(T), 0) + 1
                break   # at most one cluster per channel per pass

    stats = {
        "n_accept": n_accept,
        "frac_channels": frac_rows.float().mean().item(),
        "dE_released": dE_released,
        "cluster_size_hist": size_hist,
    }
    return idx, stats


# =========================================================================== #
# Stage 3 -- ejection chains (spec 4)
# =========================================================================== #
@torch.no_grad()
def _stage_chain_pass(
    W_g, H_g, Hdiag, G, cb, idx, E, nb_idx,
    tau, kappa, depth=18, n_chains=200, row_chunk=512,
) -> Tuple[torch.Tensor, dict]:
    """Spec 4. Lin-Kernighan-style ejection chains, variable depth, linear cost.

    Runs chains channel-serially (the sequential-in-depth pattern of spec 4.3);
    each chain either commits its best strictly-improving prefix or reverts
    entirely, so monotonicity holds (spec 4.4). G is updated by rank-1 within a
    chain and refreshed globally by the caller afterwards.
    """
    R, d = W_g.shape
    m = cb.shape[1]
    dev = W_g.device
    thr = -kappa * tau

    n_accept = 0
    dE_released = 0.0
    depth_hist = torch.zeros(depth + 1, dtype=torch.long)
    frac_rows = torch.zeros(R, dtype=torch.bool)

    for r in range(R):
        Wq_r = (W_g[r] + E[r]).clone()               # (d,)
        G_r = G[r].clone()                           # (d,)
        idx_r = idx[r].clone()
        # seeds: coordinates with largest single-flip pressure (small a)
        a_r, dstar_r, q_r = _alt_gains(
            W_g[r:r+1], (Wq_r - W_g[r]).unsqueeze(0), G_r.unsqueeze(0),
            Hdiag, cb[r:r+1], idx_r.unsqueeze(0))
        a_r = a_r[0]; 
        seed_order = a_r.argsort()[:n_chains]        # tightest first
        for seed in seed_order.tolist():
            locked = torch.zeros(d, dtype=torch.bool, device=dev)
            trail = []                               # (coord, old_lvl, new_lvl, delta)
            cum = 0.0
            best_cum = 0.0
            best_t = -1
            Wq_c = Wq_r.clone(); G_c = G_r.clone(); idx_c = idx_r.clone()
            cur = seed
            for t in range(depth):
                if locked[cur]:
                    break
                # best alternative level at cur (even if worsening)
                delta = cb[r] - Wq_c[cur]            # (m,)
                dE = 2 * delta * G_c[cur] + delta.pow(2) * Hdiag[cur]
                dE[idx_c[cur]] = float('inf')
                lvl = int(dE.argmin())
                dstep = float(delta[lvl])
                cum += float(dE[lvl])
                trail.append((cur, int(idx_c[cur]), lvl, dstep))
                # apply
                old = idx_c[cur].item()
                idx_c[cur] = lvl
                Wq_c[cur] = cb[r, lvl]
                # rank-1 G update: G += dstep * H[:,cur]
                G_c = G_c + dstep * H_g[:, cur]
                locked[cur] = True
                if cum < best_cum:
                    best_cum = cum
                    best_t = t
                # next coord: unlocked neighbour with best pressure
                nbrs = nb_idx[cur]
                nbrs = nbrs[~locked[nbrs]]
                if nbrs.numel() == 0:
                    break
                dl = cb[r].unsqueeze(0) - Wq_c[nbrs].unsqueeze(1)      # (nn,m)
                dEn = 2 * dl * G_c[nbrs].unsqueeze(1) + dl.pow(2) * Hdiag[nbrs].unsqueeze(1)
                dEn.scatter_(1, idx_c[nbrs].unsqueeze(1), float('inf'))
                cur = int(nbrs[dEn.min(1).values.argmin()])
            # accept best prefix if strictly improving past the floor
            if best_t >= 0 and best_cum < thr:
                for (coord, old_l, new_l, dstep) in trail[:best_t + 1]:
                    idx_r[coord] = new_l
                    Wq_r[coord] = cb[r, new_l]
                    G_r = G_r + dstep * H_g[:, coord]
                n_accept += 1
                dE_released += -best_cum
                frac_rows[r] = True
                depth_hist[best_t + 1] += 1
        idx[r] = idx_r

    stats = {
        "n_accept": n_accept,
        "frac_channels": frac_rows.float().mean().item(),
        "dE_released": dE_released,
        "depth_hist": depth_hist.tolist(),
    }
    return idx, stats


# =========================================================================== #
# Codebook re-solve (Eq. 9) -- untouched math, batched over group rows
# =========================================================================== #
@torch.no_grad()
def solve_codebook(
    W_g: torch.Tensor, H_g: torch.Tensor, idx: torch.Tensor, m: int,
    lambda_reg: float = 1e-7,
) -> torch.Tensor:
    """c = (P^T H P)^-1 P^T H w  per row, via the reduced-X least squares form
    used by update_C. Returns cb:(R,m).

    Uses Cholesky reduced_X = L^T so that ||reduced_X (Wq - w)||^2 == e^T H e.
    """
    R, d = W_g.shape
    L = torch.linalg.cholesky(H_g)
    Xr = L.transpose(-2, -1)                          # (d,d)  reduced X
    P = torch.nn.functional.one_hot(idx, num_classes=m).float()   # (R,d,m)
    A = torch.einsum('bj,rjc->rbc', Xr, P)            # (R,d,m)
    b = torch.einsum('bj,rj->rb', Xr, W_g).unsqueeze(-1)          # (R,d,1)
    I = math.sqrt(lambda_reg) * torch.eye(m, device=W_g.device).unsqueeze(0).expand(R, -1, -1)
    A = torch.cat([A, I], dim=1)
    b = torch.cat([b, torch.zeros(R, m, 1, device=W_g.device)], dim=1)
    cb = torch.linalg.lstsq(A, b).solution.squeeze(-1)           # (R,m)
    return cb


# =========================================================================== #
# Driver
# =========================================================================== #
@torch.no_grad()
def bopt_refine(
    W: np.ndarray,            # [out, in]
    H: np.ndarray,            # [g, in, in]
    labels: np.ndarray,       # [out, in]
    C: np.ndarray,            # [out, K]
    *,
    stages: int = 1,          # 1=B2 gating, 2=+B3, 3=+chains
    nu: int = 32,
    top_p: int = 200,
    c_cand: float = 8.0,
    kappa1: float = 2.0,
    max_cd_sweeps: int = 20,
    fix_dead: bool = True,
    chain_depth: int = 18,
    n_chains: int = 200,
    b2_max_passes: int = 8,   # restricted-VND passes for B=2 (1 move/chan/pass)
    gate_release_frac: float = 0.005,   # spec 2.7: 0.5%
    device: Optional[str] = None,
    verbose: bool = True,               # per-group / per-stage §2.6 instrumentation
    H_holdout: Optional[np.ndarray] = None,  # fresh calib split for §2.6 item 5
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Refine LNQ's (labels, C) with staged B-opt. Returns (labels, C, log).

    Operates group by group (H[k] shared across that group's rows). Preserves
    monotonicity: exact-energy acceptance + full G refresh between batches.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    Wt = _as_t(W, torch.float32, dev)               # (out,in)
    Ht = _as_t(H, torch.float32, dev)               # (g,in,in)
    idx_all = _as_t(labels, torch.long, dev)        # (out,in)
    cb_all = _as_t(C, torch.float32, dev)           # (out,K)
    out_dim, d = Wt.shape
    g = Ht.shape[0]
    m = cb_all.shape[1]
    assert out_dim % g == 0
    gs = out_dim // g

    # damp H copy ONLY for the Cholesky-based codebook solve; energy uses raw H.
    Ht_solve = Ht.clone()
    diag = torch.arange(d, device=dev)
    for k in range(g):
        Ht_solve[k, diag, diag] += 1e-5 * torch.diagonal(Ht[k]).mean()

    obj0 = _obj_lnq(Wt, Ht, idx_all, cb_all, g, gs)

    Ht_hold = _as_t(H_holdout, torch.float32, dev) if H_holdout is not None else None

    log = {"obj_init": obj0, "groups": [], "stages_run": stages}
    agg_jumps = torch.zeros(m, dtype=torch.long)   # rolled-up level-jump histogram

    if verbose:
        logging.info(
            f"[bopt] START  obj={obj0:.6f}  layer={out_dim}x{d}  m={m}  g={g}  "
            f"stages={stages}  nu={nu}  top_p={top_p}  kappa1={kappa1}"
            + ("  (+holdout H)" if Ht_hold is not None else "")
        )

    for k in range(g):
        lo, hi = k * gs, (k + 1) * gs
        W_g = Wt[lo:hi]
        idx = idx_all[lo:hi]
        cb = cb_all[lo:hi]
        H_g = Ht[k]                                  # raw, for energy
        H_solve = Ht_solve[k]
        Hdiag = torch.diagonal(H_g).clone()
        glog = {"group": k}

        # --- prereqs -------------------------------------------------------- #
        # 1.2 dead codeword audit + reseed
        dead, dead_rate = empty_codeword_audit(idx, m)
        glog["dead_rate_before"] = dead_rate
        if fix_dead and dead_rate > 0:
            cb = reseed_dead_codewords(W_g, H_g, Hdiag, cb, idx, dead)
            cb = solve_codebook(W_g, H_solve, idx, m)
        _, dead_rate_after = empty_codeword_audit(idx, m)
        glog["dead_rate_after"] = dead_rate_after
        if verbose and dead_rate > 0:
            logging.info(
                f"[bopt][g{k}] §1.2 dead codewords: {100*dead_rate:.2f}% -> "
                f"{100*dead_rate_after:.2f}%"
                + ("  <-- 2-bit capacity loss, reseeded" if m <= 4 else "")
            )

        # 1.1 CD to an actual 1-opt fixed point
        idx, n_flips, sweeps, flip_trend, polished_1opt = cd_to_fixed_point(
            W_g, H_g, Hdiag, cb, idx, max_sweeps=max_cd_sweeps)
        glog["cd_sweeps_to_fp"] = sweeps
        glog["cd_flips_last"] = n_flips
        glog["cd_converged"] = (n_flips == 0)
        glog["cd_polished_1opt"] = polished_1opt
        glog["cd_flip_trend"] = flip_trend
        cb = solve_codebook(W_g, H_solve, idx, m)
        if verbose:
            if n_flips != 0:
                # spec §1.1: "assert n_flips == 0, CD did not reach 1-opt;
                # B-opt measurement invalid". We do not hard-crash a real run,
                # but the measurement below is NOT a clean barrier probe.
                tr = flip_trend[-min(6, len(flip_trend)):]
                # Classify from the RECENT tail, not first-vs-last (the first
                # sweep always has huge flip counts, so last<first is useless).
                # Oscillating: the last value is above the tail's own minimum
                # (it bottomed out then bounced). Decreasing: last == tail min
                # and the tail is trending down.
                tail_min = min(tr)
                osc = len(tr) >= 3 and flip_trend[-1] > tail_min * 1.5
                still_down = (len(tr) >= 2 and flip_trend[-1] <= tail_min
                              and tr[-1] < tr[0])
                if osc:
                    diag = ("OSCILLATING (flips bounced back up) -- best iterate "
                            "kept; more sweeps will NOT converge this layer")
                elif still_down:
                    diag = "still decreasing -- raise --bopt_max_cd_sweeps"
                else:
                    diag = ("stalled near a plateau -- best iterate kept; "
                            "marginal gains from more sweeps")
                logging.warning(
                    f"[bopt][g{k}] §1.1 CD did NOT converge: {sweeps} sweeps, "
                    f"last flips={n_flips} (recent trend {tr}). {diag}. "
                    + ("Greedy polish recovered a true 1-opt point, so B-opt "
                       "below is on solid ground."
                       if polished_1opt else
                       "Greedy polish did NOT reach 1-opt -- B-opt gain is "
                       "CONFOUNDED with CD truncation.")
                )
                glog["cd_oscillating"] = bool(osc)
            else:
                trunc = "  <-- LNQ's K=4 would have TRUNCATED here" if sweeps > 4 else ""
                logging.info(
                    f"[bopt][g{k}] §1.1 CD->fixed point in {sweeps} sweeps "
                    f"(converged, 0 flips){trunc}"
                )

        # residual + G at the 1-opt point
        E = _gather_cb(cb, idx) - W_g
        G = E @ H_g                                  # (R,d): G[r] = H_g @ E[r]
        f_after_cd = _group_energy(E, H_g).mean().item()
        f_after_cd_tot = _group_energy(E, H_g).sum().item()   # for release fracs
        glog["f_after_cd"] = f_after_cd
        glog["f_after_cd_tot"] = f_after_cd_tot
        # §2.6 item 5: held-out energy baseline on a FRESH Hessian
        if Ht_hold is not None:
            mse_hold_before = _group_energy(E, Ht_hold[k]).mean().item()
            glog["mse_holdout_after_cd"] = mse_hold_before

        nb_idx, H_nb = _neighbour_lists(H_g, Hdiag, nu)
        # §2.6 item 6: sign histogram of H[i,k] over the neighbour lists
        if verbose:
            pos = int((H_nb > 0).sum()); neg = int((H_nb < 0).sum())
            tot = max(pos + neg, 1)
            glog["nb_sign_pos_frac"] = pos / tot
            logging.info(
                f"[bopt][g{k}] §2.6.6 neighbour H[i,k] sign: "
                f"+{100*pos/tot:.1f}% / -{100*neg/tot:.1f}%  (nu={nu})"
            )

        # noise floor (spec 2.3 Step B / 2.4)
        cand_pool = (torch.arange(d, device=dev),)   # use all coords for tau
        a0, _, _ = _alt_gains(W_g, E, G, Hdiag, cb, idx)
        tau = a0.abs().median().clamp_min(1e-30).item()
        glog["tau"] = tau

        # kappa constants
        log_n1 = math.log(d * (m - 1))
        kappa2 = _kappa(kappa1, log_n1, math.log(max(d * nu * m * m / 2, 2)))
        kappa3 = _kappa(kappa1, log_n1, math.log(max(d * nu * nu * m**3 / 6, 2)))

        # --- Stage 1: B=2 (restricted VND to fixed point) ------------------- #
        # One safe move per channel per pass; iterate with a fresh exact G until
        # no channel accepts (restricted 2-opt fixed point) or pass cap hit.
        # Refreshing G from scratch each pass is what makes the per-pass gains
        # exact -- every move is measured against the true current residual.
        f_pre = _group_energy(E, H_g).mean().item()
        s1 = {"n_accept": 0, "frac_channels": 0.0, "dE_released": 0.0,
              "jump_hist": [0]*m,
              "funnel": {"candidates": 0, "proposals": 0,
                         "below_thresh": 0, "accepted": 0},
              "passes": 0}
        for _pass in range(b2_max_passes):
            idx, sp = _stage_pair_pass(
                W_g, H_g, Hdiag, G, cb, idx, E, nb_idx, H_nb,
                tau, kappa2, c_cand=c_cand, top_p=top_p)
            s1["n_accept"] += sp["n_accept"]
            s1["dE_released"] += sp["dE_released"]
            s1["passes"] += 1
            for j in range(m):
                s1["jump_hist"][j] += sp["jump_hist"][j]
            for key in ("candidates", "proposals", "below_thresh", "accepted"):
                s1["funnel"][key] += sp["funnel"][key]
            s1["frac_channels"] = max(s1["frac_channels"], sp["frac_channels"])
            if sp["n_accept"] == 0:
                break
            # refresh exact G before the next pass (moves changed the residual)
            E = _gather_cb(cb, idx) - W_g
            G = E @ H_g
        cb = solve_codebook(W_g, H_solve, idx, m)     # spec 2.1 trailing re-solve
        E = _gather_cb(cb, idx) - W_g
        G = E @ H_g
        glog["stage1"] = s1
        f_post = _group_energy(E, H_g).mean().item()
        agg_jumps += torch.tensor(s1.get("jump_hist", [0]*m)[:m])
        if verbose:
            mhb = mha = None
            if Ht_hold is not None:
                mha = _group_energy(E, Ht_hold[k]).mean().item()
                mhb = glog.get("mse_holdout_after_cd")
            _log_stage(k, "S1(B=2)", s1, f_pre, f_post, d, kappa2, mhb, mha, R=gs)

        # --- Stage 2: B=3 --------------------------------------------------- #
        if stages >= 2:
            f_pre = _group_energy(E, H_g).mean().item()
            idx, s2 = _stage_triple_pass(
                W_g, H_g, Hdiag, G, cb, idx, E, nb_idx, H_nb,
                tau, kappa3, c_cand=c_cand, top_p=top_p)
            # energy attributable to the TRIPLE MOVES themselves (before re-sweep)
            E_tp = _gather_cb(cb, idx) - W_g
            f_triples = _group_energy(E_tp, H_g).mean().item()
            # spec 3.2: one CD re-sweep, then codebook solve
            idx, resweep_flips, _, _, _ = cd_to_fixed_point(
                W_g, H_g, Hdiag, cb, idx, max_sweeps=1)
            cb = solve_codebook(W_g, H_solve, idx, m)
            E = _gather_cb(cb, idx) - W_g
            G = E @ H_g
            s2["resweep_flips"] = resweep_flips
            glog["stage2"] = s2
            f_post = _group_energy(E, H_g).mean().item()
            if verbose:
                # attribute honestly: triple moves vs the trailing CD re-sweep
                rel_tp = (f_pre - f_triples) / max(f_pre, 1e-12)
                rel_rs = (f_triples - f_post) / max(f_pre, 1e-12)
                logging.info(
                    f"[bopt][g{k}] S2 attribution: triple-moves {100*rel_tp:.3f}% "
                    f"+ CD-resweep {100*rel_rs:.3f}% (flips={resweep_flips})"
                )
                mhb = mha = None
                if Ht_hold is not None:
                    mha = _group_energy(E, Ht_hold[k]).mean().item()
                    mhb = glog.get("mse_holdout_after_cd")
                _log_stage(k, "S2(B=3)", s2, f_pre, f_post, d, kappa3, mhb, mha, R=gs)

        # --- Stage 3: ejection chains -------------------------------------- #
        if stages >= 3:
            f_pre = _group_energy(E, H_g).mean().item()
            idx, s3 = _stage_chain_pass(
                W_g, H_g, Hdiag, G, cb, idx, E, nb_idx,
                tau, kappa2, depth=chain_depth, n_chains=n_chains)
            cb = solve_codebook(W_g, H_solve, idx, m)
            E = _gather_cb(cb, idx) - W_g
            G = E @ H_g
            glog["stage3"] = s3
            f_post = _group_energy(E, H_g).mean().item()
            if verbose:
                mhb = mha = None
                if Ht_hold is not None:
                    mha = _group_energy(E, Ht_hold[k]).mean().item()
                    mhb = glog.get("mse_holdout_after_cd")
                _log_stage(k, "S3(chain)", s3, f_pre, f_post, d, kappa2, mhb, mha, R=gs)

        f_final = _group_energy(E, H_g).mean().item()
        f_final_tot = _group_energy(E, H_g).sum().item()
        glog["f_final"] = f_final
        # total-energy drop (both numerator and denominator are sums)
        glog["dE_released_frac"] = (f_after_cd_tot - f_final_tot) / max(f_after_cd_tot, 1e-12)
        # Gate on B-opt's OWN committed gain only (spec 2.7): sum of dE_released
        # reported by the stages, which counts only exact-accepted moves and
        # excludes the CD re-sweep cleanup that would otherwise inflate the gate.
        # dE_released is a SUM over channels, so normalise by TOTAL energy.
        bopt_released = sum(
            glog.get(sk, {}).get("dE_released", 0.0)
            for sk in ("stage1", "stage2", "stage3")
        )
        glog["bopt_release_frac"] = bopt_released / max(f_after_cd_tot, 1e-12)

        idx_all[lo:hi] = idx
        cb_all[lo:hi] = cb
        log["groups"].append(glog)

    obj1 = _obj_lnq(Wt, Ht, idx_all, cb_all, g, gs)
    log["obj_final"] = obj1
    log["obj_rel"] = (obj1 - obj0) / max(obj0, 1e-12)
    # spec 2.7 gate signal (median over groups) — on B-opt's committed gain,
    # NOT total energy drop (which includes CD-resweep cleanup, a confound).
    rels = [gg["bopt_release_frac"] for gg in log["groups"]]
    rels_total = [gg["dE_released_frac"] for gg in log["groups"]]
    log["median_release_frac"] = float(np.median(rels)) if rels else 0.0
    log["median_total_drop_frac"] = float(np.median(rels_total)) if rels_total else 0.0
    log["gate_pass_release"] = log["median_release_frac"] >= gate_release_frac
    log["agg_jump_hist"] = agg_jumps.tolist()
    log["time_s"] = time.time() - t0

    # totals rolled up across groups/stages for a one-glance summary
    def _tot(stage_key):
        return sum(gg.get(stage_key, {}).get("n_accept", 0) for gg in log["groups"])
    tot1, tot2, tot3 = _tot("stage1"), _tot("stage2"), _tot("stage3")
    log["total_accepts"] = {"S1": tot1, "S2": tot2, "S3": tot3}
    jh = agg_jumps.tolist()
    multi = sum(jh[2:]) if len(jh) > 2 else 0
    tot_j = max(sum(jh), 1)

    if verbose:
        logging.info(
            f"[bopt] SUMMARY accepts: S1(B=2)={tot1}"
            + (f" S2(B=3)={tot2}" if stages >= 2 else "")
            + (f" S3(chain)={tot3}" if stages >= 3 else "")
            + f" | agg level-jumps={jh} (multi-level {100*multi/tot_j:.1f}%)"
        )
    reason = ("barriers present at reachable points"
              if log["gate_pass_release"]
              else "barriers SPARSE at points the pipeline reaches "
                   "(clean negative vs TFIC's central claim -- STOP, write it up)")
    logging.info(
        f"[bopt] obj {obj0:.6f} -> {obj1:.6f} ({100*log['obj_rel']:+.3f}%) | "
        f"B-opt release {100*log['median_release_frac']:.3f}% "
        f"(gate {'PASS' if log['gate_pass_release'] else 'FAIL'} @ "
        f"{100*gate_release_frac:.1f}%: {reason}) | "
        f"total energy drop {100*log['median_total_drop_frac']:.3f}% "
        f"(incl. CD-resweep) | {log['time_s']:.1f}s"
    )

    labels_out = idx_all.detach().cpu().numpy().astype(np.uint8)
    C_out = cb_all.detach().cpu().numpy().astype(np.float32)
    return labels_out, C_out, log


@torch.no_grad()
def _obj_lnq(Wt, Ht, idx, cb, g, gs):
    """LNQ-scale objective (mean over groups), matches objective_lnq_scale."""
    W_hat = torch.gather(cb.unsqueeze(1).expand(-1, idx.shape[1], -1),
                         2, idx.unsqueeze(-1)).squeeze(-1)
    dW = (W_hat - Wt).reshape(g, gs, -1)
    return torch.einsum('nij,njk,nik->i', dW, Ht, dW).mean().item()