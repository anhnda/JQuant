"""
B-opt test suite.

The critical test is Test 4 (positive control): a hand-built frustrated pair
that IS a 1-opt fixed point but whose joint two-coordinate move strictly lowers
energy. A correct B-opt MUST find and accept it. Without this test the suite is
vacuous -- monotonicity alone passes even if the accept path is never taken
(every stage can return accept=0 and the objective still drops from CD alone).
"""
import numpy as np, torch, logging, importlib.util, os
logging.basicConfig(level=logging.WARNING)
_p = os.path.join(os.path.dirname(__file__),
                  "any_precision/quantization/layerwise_bopt.py")
_spec = importlib.util.spec_from_file_location("layerwise_bopt", _p)
_bopt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bopt)
bopt_refine        = _bopt.bopt_refine
_stage_pair_pass   = _bopt._stage_pair_pass
_stage_triple_pass = _bopt._stage_triple_pass
_alt_gains         = _bopt._alt_gains
_neighbour_lists   = _bopt._neighbour_lists
_gather_cb         = _bopt._gather_cb
_group_energy      = _bopt._group_energy
solve_codebook     = _bopt.solve_codebook
cd_to_fixed_point  = _bopt.cd_to_fixed_point

torch.manual_seed(0); np.random.seed(0)

d, R, m, g = 64, 32, 4, 1
X = torch.randn(256, d)
A = torch.randn(d, d) * 0.3
H = (X.T @ X) / 256 + A @ A.T + torch.eye(d) * 0.05
H = 0.5 * (H + H.T)
Hg = H.numpy()[None]
W = torch.randn(R, d).numpy(); Wt = torch.tensor(W)
cb0 = torch.stack([torch.quantile(Wt[r], torch.linspace(0, 1, m)) for r in range(R)])
idx0 = (Wt.unsqueeze(-1) - cb0.unsqueeze(1)).abs().argmin(-1)
labels = idx0.numpy().astype(np.uint8); C = cb0.numpy().astype(np.float32)

def obj(labels, C):
    idx = torch.tensor(labels, dtype=torch.long); cb = torch.tensor(C)
    Wh = torch.gather(cb.unsqueeze(1).expand(-1, d, -1), 2, idx.unsqueeze(-1)).squeeze(-1)
    dW = (Wh - Wt).reshape(g, R // g, d)
    return torch.einsum('nij,njk,nik->i', dW, torch.tensor(Hg), dW).mean().item()


print("=== Test 1: monotonicity (objective must not increase) ===")
o0 = obj(labels, C)
for st in (1, 2, 3):
    lab, Cn, log = bopt_refine(W, Hg, labels, C, stages=st, device="cpu",
                               n_chains=20, chain_depth=10, verbose=False)
    o1 = obj(lab, Cn)
    assert o1 <= o0 + 1e-6, f"MONOTONICITY VIOLATED at stage {st}"
print(" OK: monotone at all stages\n")


print("=== Test 2: exact pair gain == brute-force energy delta ===")
Wg = Wt.clone(); Hgt = torch.tensor(Hg[0]); Hdiag = torch.diagonal(Hgt).clone()
cb = torch.tensor(C); idx = torch.tensor(labels, dtype=torch.long)
idx = cd_to_fixed_point(Wg, Hgt, Hdiag, cb, idx, max_sweeps=20)[0]
E = _gather_cb(cb, idx) - Wg; G = E @ Hgt
r, i, k = 0, 3, 7
new_i, new_k = (idx[r, i].item() + 1) % m, (idx[r, k].item() + 1) % m
Wq = _gather_cb(cb, idx)
di = cb[r, new_i] - Wq[r, i]; dk = cb[r, new_k] - Wq[r, k]
dE_formula = (2*di*G[r, i] + di**2*Hdiag[i] + 2*dk*G[r, k] + dk**2*Hdiag[k]
              + 2*di*dk*Hgt[i, k]).item()
e_before = _group_energy(E[r:r+1], Hgt).item()
idx2 = idx.clone(); idx2[r, i] = new_i; idx2[r, k] = new_k
e_after = _group_energy((_gather_cb(cb, idx2) - Wg)[r:r+1], Hgt).item()
assert abs(dE_formula - (e_after - e_before)) < 1e-4, "PAIR GAIN FORMULA WRONG"
print(f" formula dE={dE_formula:.6e}  actual={e_after-e_before:.6e}")
print(" OK: exact pair-gain formula verified\n")


print("=== Test 3: CD reaches a real 1-opt fixed point ===")
Wg = Wt.clone(); cb = torch.tensor(C); idx = torch.tensor(labels, dtype=torch.long)
idx, nflip, sweeps, trend, pol = cd_to_fixed_point(Wg, Hgt, Hdiag, cb, idx, max_sweeps=50)
assert nflip == 0, f"CD did not converge: {nflip} flips after {sweeps} sweeps"
E = _gather_cb(cb, idx) - Wg; G = E @ Hgt
a, _, _ = _alt_gains(Wg, E, G, Hdiag, cb, idx)
assert (a >= -1e-5).all(), "post-CD point is NOT 1-opt (negative single-flip gain)"
print(f" converged in {sweeps} sweeps, all single-flip gains >= 0 (1-opt)\n")


print("=== Test 4: POSITIVE CONTROL -- B-opt MUST cross a frustrated barrier ===")
dd = 8
Hf = torch.eye(dd); Hf[0, 1] = Hf[1, 0] = -0.5
Wf = torch.zeros(1, dd); Wf[0, 0] = 0.95; Wf[0, 1] = 0.95
cbf = torch.tensor([[0., 1.]])
idxf = torch.zeros(1, dd, dtype=torch.long)
Hdf = torch.diagonal(Hf).clone()
Wq = _gather_cb(cbf, idxf); Ef = Wq - Wf; Gf = Ef @ Hf
af, _, _ = _alt_gains(Wf, Ef, Gf, Hdf, cbf, idxf)
assert (af[0, :2] > 0).all(), "setup wrong: single flip already helps, not a barrier"
def E2(a, b_):
    ix = idxf.clone(); ix[0, 0] = a; ix[0, 1] = b_
    e = _gather_cb(cbf, ix) - Wf; return (e @ Hf * e).sum().item()
base = E2(0, 0); joint = E2(1, 1)
assert joint < base - 0.1, "setup wrong: joint move does not help"
tau = af.abs().median().clamp_min(1e-30).item()
nb_idx, H_nb = _neighbour_lists(Hf, Hdf, nu=dd-1)
idx_out, st = _stage_pair_pass(Wf, Hf, Hdf, Gf, cbf, idxf.clone(), Ef,
                               nb_idx, H_nb, tau, kappa2=0.5, c_cand=64.0, top_p=50)
assert st["n_accept"] >= 1, \
    f"POSITIVE CONTROL FAILED: B-opt did not accept a real barrier move (funnel={st['funnel']})"
assert idx_out[0, 0].item() == 1 and idx_out[0, 1].item() == 1, \
    f"accepted but wrong levels: {idx_out[0, :2].tolist()} != [1,1]"
e_after = ((_gather_cb(cbf, idx_out) - Wf) @ Hf * (_gather_cb(cbf, idx_out) - Wf)).sum().item()
assert e_after < base - 0.1, "energy did not drop after the accepted pair move"
print(f" barrier crossed: energy {base:.4f} -> {e_after:.4f}, accepts={st['n_accept']}, levels->[1,1]")
print(" OK: B-opt provably crosses a frustrated barrier (accept path exercised)\n")


print("=== Test 5: noise floor rejects a sub-threshold move ===")
idx_hi, st_hi = _stage_pair_pass(Wf, Hf, Hdf, Gf, cbf, idxf.clone(), Ef,
                                 nb_idx, H_nb, tau, kappa2=2.0, c_cand=64.0, top_p=50)
assert st_hi["n_accept"] == 0, "noise floor failed: accepted a move below kappa*tau"
print(f" kappa2=2.0 (thr={2.0*tau:.2f}) correctly REJECTS the {base-joint:.2f} gain")
print(" OK: noise floor gates as designed\n")


print("=== Test 6: held-out consistency (real fresh Hessian) ===")
torch.manual_seed(5)
Ac = torch.randn(d, d) * 0.4
Hc = Ac @ Ac.T + torch.eye(d) * 0.05; Hc = 0.5 * (Hc + Hc.T)
Xh = torch.randn(256, d)
Hh = (Xh.T @ Xh) / 256 + Ac @ Ac.T + torch.eye(d) * 0.05; Hh = 0.5 * (Hh + Hh.T)
Wc = torch.randn(R, d).numpy(); Wct = torch.tensor(Wc)
cbc = torch.stack([torch.quantile(Wct[r], torch.linspace(0, 1, m)) for r in range(R)])
idxc = (Wct.unsqueeze(-1) - cbc.unsqueeze(1)).abs().argmin(-1)
lc = idxc.numpy().astype(np.uint8); Cc = cbc.numpy().astype(np.float32)
lab, Cn, log = bopt_refine(Wc, Hc.numpy()[None], lc, Cc, stages=2, device="cpu",
                           kappa1=1.0, H_holdout=Hh.numpy()[None], verbose=False)
assert log["obj_final"] <= log["obj_init"] + 1e-6, "calib objective rose"
g0 = log["groups"][0]
print(f" calib obj {log['obj_init']:.4f} -> {log['obj_final']:.4f} | holdout tracked={'mse_holdout_after_cd' in g0}")
print(" OK\n")

print("=== Test 7: batch safety on DENSE H (per-pass energy must not rise) ===")
# Dense H with strong cross-block coupling: this is the regime where accepting
# multiple disjoint pairs per channel could raise energy via the cross term
# 2*delta_a^T H delta_b. bopt_refine must never let exact energy increase.
torch.manual_seed(11)
dd2 = 96
Ad = torch.randn(dd2, dd2) * 0.5           # dense, strong off-diagonal
Hd2 = Ad @ Ad.T + torch.eye(dd2) * 0.05; Hd2 = 0.5 * (Hd2 + Hd2.T)
assert torch.linalg.eigvalsh(Hd2).min() > 0, "H not PD"
Rr, mm = 48, 4
Wd = torch.randn(Rr, dd2).numpy(); Wdt = torch.tensor(Wd)
cbd = torch.stack([torch.quantile(Wdt[r], torch.linspace(0, 1, mm)) for r in range(Rr)])
idxd = (Wdt.unsqueeze(-1) - cbd.unsqueeze(1)).abs().argmin(-1)
ld = idxd.numpy().astype(np.uint8); Cd = cbd.numpy().astype(np.float32)

def exact_energy(labels, C, Hten):
    idx = torch.tensor(labels, dtype=torch.long); cb = torch.tensor(C)
    Wh = torch.gather(cb.unsqueeze(1).expand(-1, Hten.shape[0], -1), 2,
                      idx.unsqueeze(-1)).squeeze(-1)
    e = Wh - Wdt
    return torch.einsum('ri,ij,rj->', e, Hten, e).item()

e_in = exact_energy(ld, Cd, Hd2)
for kap in (2.0, 1.0, 0.3):   # including aggressive kappa that accepts a lot
    lab, Cn, log = bopt_refine(Wd, Hd2.numpy()[None], ld, Cd, stages=2,
                               device="cpu", kappa1=kap, verbose=False,
                               b2_max_passes=8)
    e_out = exact_energy(lab, Cn, Hd2)
    assert e_out <= e_in + 1e-4, \
        f"DENSE-H MONOTONICITY VIOLATED at kappa1={kap}: {e_in:.4f} -> {e_out:.4f}"
print(f" dense H ({dd2}x{dd2}), kappa in {{2,1,0.3}}: energy never rose (in={e_in:.2f})")
print(" OK: one-move-per-channel batch is exact-safe under dense H\n")

print("=== Test 8: B=3 POSITIVE CONTROL -- 2-opt but not 3-opt ===")
# From the review doc: equicorrelation H, a point that is exactly 2-opt (no
# single OR pair move helps) but whose TRIPLE flip strictly improves. B=2 must
# fail to move it; B=3 must solve it.
d3 = 3
H3 = torch.tensor([[1., -0.4, -0.4], [-0.4, 1., -0.4], [-0.4, -0.4, 1.]])
assert torch.linalg.eigvalsh(H3).min() > 0, "H3 not PD"
w3 = torch.tensor([[0.75, 0.75, 0.75]])
cb3 = torch.tensor([[0., 1.]])
idx3 = torch.zeros(1, d3, dtype=torch.long)
Hd3 = torch.diagonal(H3).clone()
def E3(ix):
    e = _gather_cb(cb3, ix) - w3; return (e @ H3 * e).sum().item()
base3 = E3(idx3); triple3 = E3(torch.ones(1, d3, dtype=torch.long))
assert triple3 < base3 - 0.1, "setup wrong: triple does not help"

Wq3 = _gather_cb(cb3, idx3); E3t = Wq3 - w3; G3 = E3t @ H3
a3, _, _ = _alt_gains(w3, E3t, G3, Hd3, cb3, idx3)
nb3, Hnb3 = _neighbour_lists(H3, Hd3, nu=2)
tau3 = a3.abs().median().clamp_min(1e-30).item()

# B=2 must NOT solve it (it is 2-opt): pair pass leaves it unmoved
idx_p, sp = _stage_pair_pass(w3, H3, Hd3, G3, cb3, idx3.clone(), E3t,
                             nb3, Hnb3, tau3, kappa2=0.3, c_cand=64.0, top_p=50)
assert sp["n_accept"] == 0, "B=2 wrongly moved a 2-opt point"

# B=3 MUST solve it
idx_t, st3 = _stage_triple_pass(w3, H3, Hd3, G3, cb3, idx3.clone(), E3t,
                                nb3, Hnb3, tau3, kappa3=0.3, c_cand=64.0, top_p=50)
assert st3["n_accept"] >= 1, f"B=3 failed to cross the triple barrier (stats={st3})"
assert (idx_t[0] == 1).all(), f"B=3 moved to wrong state: {idx_t[0].tolist()}"
e_after3 = E3(idx_t)
assert e_after3 < base3 - 0.1, "B=3 accepted but energy did not drop"
print(f" 2-opt point {base3:.3f}: B=2 leaves it (accept=0), "
      f"B=3 solves it {base3:.3f}->{e_after3:.3f}")
print(" OK: B=3 reaches what B=2 structurally cannot\n")

print("=== Test 9: greedy 1-opt polish reaches 1-opt from a bad start ===")
# Feed a deliberately non-1-opt assignment; the greedy polish must return a
# genuine 1-opt point (all single-flip gains >= 0) without relying on cyclic CD
# convergence. This is the guard for oscillating layers like v_proj.
_greedy_1opt = _bopt._greedy_1opt
torch.manual_seed(9)
dp = 48
Ap = torch.randn(dp, dp) * 0.4; Hp = Ap @ Ap.T + torch.eye(dp) * 0.05; Hp = 0.5*(Hp+Hp.T)
Hdp = torch.diagonal(Hp).clone()
Rp, mp = 16, 4
Wp = torch.randn(Rp, dp)
cbp = torch.stack([torch.quantile(Wp[r], torch.linspace(0, 1, mp)) for r in range(Rp)])
# start from a RANDOM (bad) assignment, not nearest-codeword
idxp = torch.randint(0, mp, (Rp, dp))
Ep = _gather_cb(cbp, idxp) - Wp
ap, _, _ = _alt_gains(Wp, Ep, Ep @ Hp, Hdp, cbp, idxp)
n_bad = int((ap < -1e-6).sum())
assert n_bad > 0, "test setup: start was already 1-opt"
idx_pol = _greedy_1opt(Wp, Hp, Hdp, cbp, idxp, max_iters=200)
Epol = _gather_cb(cbp, idx_pol) - Wp
apol, _, _ = _alt_gains(Wp, Epol, Epol @ Hp, Hdp, cbp, idx_pol)
assert (apol >= -1e-6).all(), "greedy polish did NOT reach 1-opt"
e0 = torch.einsum('ri,ij,rj->', Ep, Hp, Ep).item()
e1 = torch.einsum('ri,ij,rj->', Epol, Hp, Epol).item()
assert e1 <= e0 + 1e-6, "greedy polish raised energy"
print(f" {n_bad} improving flips at start -> 0 after polish, energy {e0:.2f}->{e1:.2f}")
print(" OK: polish guarantees a true 1-opt point (oscillation-proof)\n")

print("ALL TESTS PASSED")