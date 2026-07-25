#!/usr/bin/env bash
# LNQ + staged B-opt solver.
#
#   bash scripts/run_lnqbopt.sh <model_ref> <bits> <num_groups> [-m mode]
#   bash scripts/run_lnqbopt.sh Llama-3.2-1B 2 1
#
# Runs LNQ to convergence, then strengthens LNQ's assignment fixed point from
# 1-opt to B-opt with the exact-energy staged search. LNQ closes the "better
# search within the nearest-codeword set" gap (Prop. 4.1); B-opt buys what is
# left by LEAVING the 1-opt neighbourhood -- multi-coordinate moves that no
# single flip can reach. The codebook solve (Eq. 9) is untouched and every
# accepted move is committed only on its EXACT dE < 0, so monotonicity holds.
#
# Per module it logs (STAGES=1, the gating pass):
#
#   [bopt] obj 0.14210 -> 0.14139 (-0.503%) | median release 0.71%
#          (gate PASS @ 0.5%) | 41.3s
#
# WHAT THE NUMBERS MEAN
#   obj            LNQ-scale objective (mean over groups), directly comparable
#                  to LNQ's own "Objective:" lines and to run_lnqflexnu's obj_*.
#   median release dE released by B-opt as a fraction of the energy AFTER CD
#                  reached its 1-opt fixed point -- i.e. barrier gain only, with
#                  CD-truncation removed by the §1.1 full-convergence prereq.
#   gate           median release >= GATE_FRAC (default 0.5%). This is the
#                  spec's §2.7 decision: if it FAILS, barriers are sparse at the
#                  points the pipeline actually reaches -- STOP, that is a clean
#                  negative result, do not escalate to B=3.
#
# STAGES (escalate ONLY when the previous gate says to):
#   1  B=2 pair pass          -- always. This is the gating experiment.
#   2  + B=3 triple pass      -- only if Stage 1 clears 0.5% (set STAGES=2).
#   3  + ejection chains      -- only if Stage 2's cluster sizes tail past 5,
#                                i.e. depth, not width, is binding (STAGES=3).
#
# PREREQUISITES (same as the other solvers):
#   bash scripts/run_sqllm.sh $MODEL $BITS $G          # init -- REQUIRED
#   bash scripts/run_lnq.sh   $MODEL $BITS $G -m hessians
#
# SCOPE: the plan targets 2-bit and 3-bit only. At 4 bits the published numbers
# already tie and there is no headroom to buy -- run this at BITS=2 or 3.
set -x

MODEL_REF=$1
BITS=$2
NUM_GROUPS=$3

MODEL_PATH=$(python resolve_model.py "$MODEL_REF") || exit $?

MODE_OPT=""
if [[ "$4" == "-m" && -n "$5" ]]; then
  MODE_OPT="--mode $5"
fi

DATASET=${DATASET:-c4}
SEQ_LEN=${SEQ_LEN:-2048}
NUM_EXAMPLES=${NUM_EXAMPLES:-128}

# ---- LNQ stage --------------------------------------------------------------
NUM_ITER=${NUM_ITER:-3}
CD_CYCLES=${CD_CYCLES:-4}

# ---- B-opt stage ------------------------------------------------------------
# 1 = B=2 gating pass (default), 2 = +B=3, 3 = +ejection chains. Do NOT jump to
# 2 or 3 without reading the gate line from the STAGES=1 run first.
STAGES=${STAGES:-1}
# Neighbour-list width (spec default 32). Ranked by correlation-normalised
# |H[i,k]| but the synergy uses raw H[i,k] -- that is what enters the energy.
NU=${NU:-32}
# Pairs enumerated exactly per channel (approximate proposal, exact m^2 accept).
# The score may be aggressive; nothing is committed on it.
TOP_P=${TOP_P:-200}
# Base noise-floor multiplier. kappa_B = kappa1 * sqrt(log N_B / log N_1) grows
# with search width automatically (§2.4). Raise it -- do NOT lower it -- if
# held-out MSE fails to follow calibration MSE down (§2.6 item 5).
KAPPA1=${KAPPA1:-2.0}
# CD is swept to an ACTUAL 1-opt fixed point before measuring barriers, so any
# gain is a barrier and not CD truncation. Cap on that convergence loop.
MAX_CD_SWEEPS=${MAX_CD_SWEEPS:-20}
# Stage-3 ejection-chain depth and count (linear in depth; §4.2). Ignored unless
# STAGES=3.
CHAIN_DEPTH=${CHAIN_DEPTH:-18}
N_CHAINS=${N_CHAINS:-200}

python layerwise_nuq.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --solver lnqbopt \
  --seed_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 \
  --num_iterations "$NUM_ITER" --cd_cycles "$CD_CYCLES" \
  --bopt_stages "$STAGES" \
  --bopt_nu "$NU" \
  --bopt_top_p "$TOP_P" \
  --bopt_kappa1 "$KAPPA1" \
  --bopt_max_cd_sweeps "$MAX_CD_SWEEPS" \
  --bopt_chain_depth "$CHAIN_DEPTH" \
  --bopt_n_chains "$N_CHAINS" \
  $MODE_OPT