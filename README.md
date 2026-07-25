<h1 align="center">FlexNu</h1>

<p align="center"><b>Loss-Aware Non-Nearest Assignment for Non-Uniform Weight Quantization</b><br>
Built on <a href="https://arxiv.org/abs/2505.07004">GuidedQuant</a>'s end-loss-guided objective.</p>

---

## What this is

Non-uniform weight quantizers fit a scalar codebook to the weight distribution and
then assign each weight to its **nearest** codeword. That rule is not optimal for
the objective these methods actually minimize — layer output reconstruction — and
the gap is governed entirely by the off-diagonal structure of the Hessian.

FlexNu keeps the codebook and the bit-rate exactly as they are, and changes only
the assignment. Following FlexRound, a learned element-wise divisor perturbs the
*query* but not the *reconstruction*, so the committed weight need not be the
nearest codeword. The divisor is training-time only and is discarded at commit.

The enabling observation is that **scalar codebooks are sorted**, so
nearest-codeword assignment is exactly a monotone step function of the query with
thresholds at codeword midpoints:

$$\hat{w} = c_0 + \sum_{j} \mathbf{1}[\,q > t_j\,]\,\Delta_j,
\qquad t_j = \tfrac{1}{2}(c_j + c_{j+1}),\quad \Delta_j = c_{j+1}-c_j > 0.$$

The forward pass uses the hard indicator — it *is* nearest-codeword assignment, no
relaxation. The backward pass substitutes a logistic bump. Because the map is
monotone, this straight-through estimator is consistent, with no softmax, no
temperature annealing, and no codebook collapse.

**What FlexNu contributes here** is the solver. **GuidedQuant contributes the
objective**: instead of the plain layer Gram $X^\top X$, we optimize against the
saliency-weighted Hessian $H_k = X^\top \mathrm{Diag}(s_k) X$, where $s_k$ is the
squared end-loss gradient averaged over a group of output channels. FlexNu's
escape is therefore aimed at *loss* reduction rather than *output* reduction.

> [!IMPORTANT]
> **This is a research prototype with unpublished results.** The synthetic
> validation is in the paper draft; real-model numbers are not yet established.
> Everything below is protocol, and the ablation ladder in
> [Validating a run](#validating-a-run) is not optional — cell A is a correctness
> check that must pass before any perplexity number means anything.

---

## When it helps, and when it provably does not

**Proposition 1.** *If $H_k$ is diagonal, the objective separates across input
coordinates and nearest-codeword assignment is provably optimal.*

Two consequences worth internalizing before running anything:

- **FlexNu buys exactly nothing when $H_k$ is well-conditioned.** This is a
  property of the objective, not a limitation of the implementation. The gain
  should track $\kappa(H_k)$; if it does not, the mechanism is falsified.
- **SqueezeLLM's objective cannot host this method.** It minimizes a *diagonal*
  Fisher, where Proposition 1 makes any non-nearest move strictly worse.
  SqueezeLLM's role here is initialization only.

The comparison target is **LNQ**, GuidedQuant's own solver. LNQ alternates a
closed-form codebook update with coordinate descent, and its CD step rounds to
nearest by construction — so every point LNQ can reach is a nearest-codeword map
of some codebook. It is a strong in-set solver. FlexNu is not a better search of
that set; it targets a different reachable set.

---

## Installation

Python 3.11 / CUDA 12.4 / pip 25.1.

```bash
pip install -r requirements.txt
```

Then the Any-Precision-LLM CUDA kernels, either from source

```bash
cd inference/ap_gemv && bash install.sh
```

or pre-built:

```bash
pip install ap-gemv -i https://jinukkim.me/whl/cu124   # CUDA 12.4
```

---

## Quick start

Four steps. Each reuses the previous one's cache, so the order matters.

```bash
MODEL=Llama-3.2-1B
BITS=3
G=1                      # GuidedQuant's output-channel group count

# 1. SqueezeLLM init -- REQUIRED, not optional.
#    Writes cache/quantized/, the source of init_labels + init_centroids for
#    BOTH solvers. For FlexNu it also seeds the best-iterate incumbent, i.e.
#    the floor guaranteeing the result is never worse than nearest-codeword.
#    The pipeline hard-returns without it ("Need to provide it").
bash scripts/run_sqllm.sh $MODEL $BITS $G

# 2. GuidedQuant Hessians -- solver-independent, cache once and share.
bash scripts/run_lnq.sh $MODEL $BITS $G -m hessians

# 3. LNQ baseline (cell E) -- the number FlexNu has to beat.
bash scripts/run_lnq.sh $MODEL $BITS $G

# 4. FlexNu.
bash scripts/run_flexnu_gq.sh $MODEL $BITS $G
```

Step 2 is optional as a separate call — step 3 would produce the Hessians anyway
— but it is the expensive amortizable part, and splitting it out means a failure
in step 3 does not cost you the Hessian pass.

### Evaluate

Dequantize to a dense HF checkpoint, then measure perplexity:

```bash
# LNQ baseline
python dequant_to_hf.py \
    cache/layerwise_packed/layerwise-$MODEL-w$BITS-c4_s128_blk2048_g${G}_iter3_cd4 \
    cache/dense/${MODEL}-w${BITS}-lnq
python eval_ppl.py --model-path cache/dense/${MODEL}-w${BITS}-lnq \
    --datasets wikitext2 c4 --seqlen 2048 --dtype fp16

# FlexNu -- note the _flexnu suffix
python dequant_to_hf.py \
    cache/layerwise_packed/layerwise-$MODEL-w$BITS-c4_s128_blk2048_g${G}_iter3_cd4_flexnu \
    cache/dense/${MODEL}-w${BITS}-flexnu
python eval_ppl.py --model-path cache/dense/${MODEL}-w${BITS}-flexnu \
    --datasets wikitext2 c4 --seqlen 2048 --dtype fp16
```

> [!WARNING]
> **Check the two packed paths differ only by `_flexnu`.** If they differ in the
> *model name* as well, one of the runs predates the `--model_name` fix and you
> are comparing artifacts from different pipeline states. See
> [Cache paths](#cache-paths-and-the-model_name-flag).

For Llama-3.2-1B you should see **112 quantized modules and 35 unquantized
tensors** — 7 projections × 16 layers, with layernorms, embeddings, `model.norm`,
and `lm_head` left in fp16. This is identical for LNQ and FlexNu; both read the
same `module_names` from `any_precision/analyzer/architectures/llama.yaml`. If
the two runs disagree, something is wrong.

---

## Validating a run

Do this before trusting any perplexity number.

### 1. Did FlexNu actually run?

```bash
grep -c "\[flexnu\]" logs_layer/*_flexnu_*.txt      # must be > 0
grep -m3 "\[flexnu\]" logs_layer/*_flexnu_*.txt
```

That log line is emitted only from inside `train_flexnu`. Zero hits means the
solver was never reached — check for `Need to provide it` or `already exists and
is not empty` in the log, and note that tracebacks go to **stdout**, not the log
file:

```bash
bash scripts/run_flexnu_gq.sh $MODEL $BITS $G 2>&1 | tail -40
```

### 2. The diagnostic that matters

Each `[flexnu]` line reports `moved=`, the fraction of weights whose committed
index differs from the nearest-codeword index. **This is the direct measurement
of the paper's claim.** If it is near zero on real layers, the escape is not
happening and any perplexity difference came from somewhere else.

### 3. The ablation ladder

Set `CELL` to select a rung.

| Cell | Command | Codebook | Divisor | Isolates |
|:--|:--|:--|:--|:--|
| **E** | `run_lnq.sh` | LNQ | — | baseline, **the number to beat** |
| **A** | `CELL=A run_flexnu_gq.sh` | frozen | frozen | correctness check |
| **B** | `CELL=B run_flexnu_gq.sh` | learned | frozen | fitting $W$ |
| **C** | `CELL=C run_flexnu_gq.sh` | frozen | learned | escaping via the loss |
| **D** | `CELL=D run_flexnu_gq.sh` | learned | learned | joint (default) |

**Cell A is not a result.** It sets $T=0$ and commits `searchsorted` on the
initialization codebook, so `E_init` and `E_final` in its log must be identical.
Any deviation is a fault in the Hessian handoff or the codebook layout — stop and
debug.

Note that A does **not** reproduce E. E is converged LNQ, which is substantially
better than the SqueezeLLM initialization A commits. The scientific comparison is
**B versus C**.

### 4. The falsification test

Proposition 1 predicts the gain tracks $\kappa(H_k)$. Compute it per layer
directly from the cached Hessians — no quantization run needed, seconds of
eigendecomposition — and correlate with per-layer `energy_rel_drop`. A null or
negative correlation falsifies the mechanism as stated.

A second measurement is specific to this combination. Compare $\kappa(H_k)$
against $\kappa(X^\top X)$, the latter obtainable from the same pipeline with
`--is_nosal true`:

- If saliency reweighting **increases** anisotropy, $H_k$ is a better substrate
  for FlexNu than the plain Gram — the whole argument for combining them.
- If it **flattens** the spectrum, expect FlexNu to underperform here relative to
  the plain layer-wise objective, and the honest conclusion is that the two
  techniques compete rather than compose.

This is cheap and it is worth running *before* spending GPU-hours.

---

## Tuning

All settings are environment variables read by `scripts/run_flexnu_gq.sh`.

| Variable | Default | Notes |
|:--|:--|:--|
| `ITERS` | `300` | Adam steps per row-block |
| `LR_SCALE` | `3e-3` | divisor learning rate |
| `LR_CB` | `1e-5` | codebook learning rate — **deliberately tiny** |
| `ROW_BLOCK` | `64` | memory knob; does not change the result |
| `TAU_FRAC` | `0.5` | backward-only STE width, as a fraction of mean gap |
| `STAGE_FRAC` | `0.0` | **leave at 0** — see below |
| `EVAL_EVERY` | `1` | **leave at 1** — see below |
| `CELL` | `D` | ablation rung |

### Three settings that are not free parameters

**`LR_CB` must stay far below `LR_SCALE`.** Raising it degrades results
monotonically past roughly $10^{-4}$: the codebook chases $W$ and drags the
thresholds out from under the divisor. Whether that cliff moves under $H_k$ —
where the codebook chases *loss-weighted* $W$ instead — is the most informative
single sweep available here, and is unresolved.

**`STAGE_FRAC=0` (joint) is not a default, it is a requirement.** Staging the
divisor after the codebook starts $\delta_2 = 0$ on a grid already fitted to $W$
— precisely the nearest-neighbour point the divisor exists to escape — and it
must then climb out. Full staging collapses the effect. The flag exists for the
ablation only.

**`EVAL_EVERY=1` is not tunable in practice.** The hard energy is
piecewise-constant and non-monotone along the STE trajectory: good assignments
appear and vanish within a few steps. Subsampling misses them and the best-iterate
guard then falls back to initialization. This roughly doubles per-step cost and
that cost is not optional.

### Memory

The STE forward materializes a `[row_block, d_in, K-1]` tensor and autograd
retains several. At `d_in=11008`, `K=8` (3-bit), fp32: 4.9 MB per tensor at
`ROW_BLOCK=16`, 39 MB at 128, with four to six live copies through the backward.
4-bit (`K=16`) roughly doubles it. Raise `ROW_BLOCK` until memory binds — rows are
independent given their group's Hessian, so it changes peak memory only.

---

## Design notes

### The best-iterate guard records, it never constrains

The STE gradient is exact for the *smoothed* objective, not the piecewise-constant
true one, so iterates keep moving after the hard energy bottoms out and momentum
walks them uphill. The last iterate is essentially never the best, so snapshotting
is necessary.

The subtlety: escaping nearest-neighbour requires **crossing** a threshold, and a
crossing is a transient in which the smoothed objective improves while the hard
energy briefly worsens. A guard that rejects, clips, or rewinds to such iterates
filters out exactly the moves the method depends on. Ours lets the optimizer run
unconstrained and only records what it passes through. The SqueezeLLM
initialization seeds the incumbent, so the committed result is never worse than
nearest-codeword — a floor on the *output*, not a leash on the *search*.

### Codebook width is forced to full row

`pack.py` reads `layer_lut[name][r_idx][0]`, with the accompanying comment that
the index assumes a single group: the Any-Precision-LLM kernel stores exactly one
codebook of $2^b$ entries per output row, spanning the full input dimension.
Block-wise codebooks are not representable in this format, so there is no
block-size flag.

This costs codebook expressiveness relative to a block-wise setup — one codebook
covers $d_{\text{in}}$ weights rather than 64 — but it is exactly what SqueezeLLM
and LNQ already do here, so the comparison is like-for-like and the bit-rate is
identical. Expect the synthetic magnitudes from the paper draft not to transfer.

### No Cholesky

LNQ factorizes $H_k$ for its closed-form codebook solve, which costs
$O(d_{\text{in}}^3)$ and requires positive *definiteness* — the reference
implementation escalates a damping factor and aborts if it cannot achieve it.
FlexNu touches $H_k$ only through the matmul, so positive *semi*definiteness
suffices. Both the cubic term and that failure mode disappear. A small ridge
($10^{-5}$ relative to the mean diagonal) is retained because a semidefinite
$H_k$ has null directions the divisor will otherwise wander into.

### Sortedness is structural

The codebook is parameterized as $c_0 = a$,
$c_{j+1} = c_j + \mathrm{softplus}(g_j)$, so $\Delta_j > 0$ holds at every point
of training with no sort, no projection, and no constraint violation. This keeps
the reconstruction monotone (hence the STE valid) and guarantees the committed
codebook is strictly increasing, which `searchsorted` at commit time requires.

---

## Cache paths and the `--model_name` flag

`resolve_model.py` turns a bare model name into an HF snapshot path ending in a
commit hash. Cache directories are derived from `basename(model_path)`, so
without intervention they are named after that hash — which changes on every
re-pull, silently invalidating every downstream cache.

All three scripts therefore forward `--model_name "$MODEL_REF"`, the clean name
you typed, and both pipelines respect it:

```python
model_name = model_name or model_string.split("/")[-1]
```

Resulting layout for `MODEL=Llama-3.2-1B`, `BITS=3`, `G=1`:

```
cache/quantized/Llama-3.2-1B-w3_orig3-c4_s128_blk2048            <- shared init
cache/hessians/Llama-3.2-1B-c4_s128_blk2048_g1                   <- shared Hessians
cache/layerwise_quantized/Llama-3.2-1B-w3-...-g1_iter3_cd4         <- LNQ
cache/layerwise_quantized/Llama-3.2-1B-w3-...-g1_iter3_cd4_flexnu  <- FlexNu
cache/layerwise_packed/layerwise-Llama-3.2-1B-w3-...-cd4           <- LNQ
cache/layerwise_packed/layerwise-Llama-3.2-1B-w3-...-cd4_flexnu    <- FlexNu
```

Two facts worth knowing before sweeping:

- **Precision must match** across steps 1, 3, and 4 —
  `initialization_cache_path` embeds `w{bits}_orig{bits}`.
- **`G` need not match between step 1 and steps 3–4.** Step 1's `--num_groups`
  affects only the saliency cache; `quantized_cache_path` has no `g` in it. One
  SqueezeLLM run serves every `G` downstream.

> [!CAUTION]
> The packing stage **skips silently** when its output directory exists and is
> non-empty. A stale directory from an earlier run will be preserved and
> evaluated, which looks exactly like "FlexNu produced identical results to LNQ."
> When in doubt:
> ```bash
> rm -rf cache/layerwise_quantized/*_flexnu cache/layerwise_packed/*_flexnu cache/dense/*
> ```

---

## Where the code lives

| Path | Role |
|:--|:--|
| `any_precision/quantization/layerwise_flexnu.py` | the solver — `train_flexnu` |
| `any_precision/quantization/layerwise_quantize.py` | dispatch; LNQ's `train_least_squares` |
| `any_precision/quantization/layerwise_main.py` | pipeline, cache paths |
| `any_precision/quantization/activations.py` | saliency-weighted Hessian accumulation |
| `scripts/run_flexnu_gq.sh` | runner + ablation ladder |

`train_flexnu` has the same signature and return as LNQ's `train_least_squares`,
so the two are interchangeable behind a one-line branch in `seed_layer`.

---

## Limitations

**Scope.** No benefit when $H_k$ is well-conditioned (Proposition 1). This
delimits where the method should be deployed.

**Rotation.** Incoherence processing (QuIP, QuaRot, SpinQuant) improves Gram
conditioning, which by Proposition 1 would reduce available headroom. Whether
rotation and FlexNu compose or compete is open, and is arguably the most important
question this work raises.

**Cost.** Substantially more expensive at quantization time than one-shot
methods: full $H_k$ per layer plus $T$ optimization steps per row-block, with
mandatory per-step energy evaluation. Inference cost and bit-rate are unchanged.

**No convergence guarantee.** LNQ is a descent method with a proof. The STE
trajectory here is non-monotone in the true objective; the guard makes results
reproducible and bounds them below by nearest-codeword, but the method remains
sensitive to `LR_SCALE` in a way we have not fully characterized.

**Scalar only.** The sortedness argument does not extend to unordered vector
codebooks (E8 lattices, AQLM's additive books), where the Voronoi topology is the
obstruction. Trellis-structured codebooks, being ordered along the chain, look
more promising.

---

## Acknowledgements

This work builds directly on **GuidedQuant** ([paper](https://arxiv.org/abs/2505.07004),
[code](https://github.com/snu-mllab/GuidedQuant)), whose saliency-weighted
objective and LNQ solver are the foundation and the baseline here. It also builds
on **FlexRound** (Lee et al., ICML 2023) for the divisor mechanism,
**SqueezeLLM** for initialization, and **Any-Precision-LLM** for the inference
kernels.

Model support (**Qwen3 dense, Gemma3, Llama 3, Llama 2**) is inherited from
GuidedQuant. To add architectures, modify
`any_precision/analyzer/architectures/` and
`any_precision/analyzer/splitted_models/`.