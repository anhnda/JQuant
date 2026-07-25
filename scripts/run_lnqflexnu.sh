#!/usr/bin/env bash
# LNQ + FlexNu layered solver.
#
#   bash scripts/run_lnqflexnu.sh <model_ref> <bits> <num_groups> [-m mode]
#   bash scripts/run_lnqflexnu.sh Llama-3.2-1B 3 1
#
# Runs LNQ to convergence, then FlexNu from LNQ's solution. Per module it logs
# three objectives on LNQ's exact scale:
#
#   [lnq+flexnu] stage 1 (LNQ):    obj 0.14210 -> 0.12570  (-11.54% vs init)
#   [lnq+flexnu] stage 2 (FlexNu): obj 0.12570 -> 0.11003  vs_lnq=-12.47% WIN
#                                  moved=4.81% (vs LNQ assign: 4.81%)
#   [lnq+flexnu] TOTAL: init 0.14210 -> final 0.11003  vs_init=-22.57%
#                       (LNQ contributed 51% of the total reduction)
#
# vs_lnq is the number that matters -- NEGATIVE means FlexNu improved on
# converged LNQ. vs_init is context.
#
# PREREQUISITES (same as the other solvers):
#   bash scripts/run_sqllm.sh $MODEL $BITS $G     # init -- REQUIRED
#   bash scripts/run_lnq.sh   $MODEL $BITS $G -m hessians
#
# THE RISK: starting delta2=0 on LNQ's converged (nearest-codeword) grid is
# structurally the staging trap of FlexNu Section 3.5. If moved=0% everywhere,
# set DELTA_NOISE=1e-2 to break the fixed point, and compare against a
# from-SqueezeLLM run to confirm the trap is what is happening.
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

# ---- FlexNu stage -----------------------------------------------------------
ITERS=${ITERS:-300}
LR_SCALE=${LR_SCALE:-3e-3}
# Raised from the standalone 1e-5 default. The codebook now starts at LNQ's
# OPTIMAL solution (closed-form Eq. 9), so it has less to gain by chasing W and
# therefore less reason to drag thresholds out from under the divisor. Sweep it.
LR_CB=${LR_CB:-1e-5}
# 512 on a 24 GB card; the [row_block, d_in, K-1] tensor is the limit. 64 leaves
# the GPU ~80% idle on this model.
ROW_BLOCK=${ROW_BLOCK:-512}
TAU_FRAC=${TAU_FRAC:-0.5}
STAGE_FRAC=${STAGE_FRAC:-0.0}
EVAL_EVERY=${EVAL_EVERY:-1}
# Break the delta2=0 fixed point. 0.0 = exact FlexRound init.
DELTA_NOISE=${DELTA_NOISE:-0.0}

# ---- signed G ---------------------------------------------------------------
# G = exp(gamma+) - exp(gamma-) replaces the positive divisor exp(-delta2).
# The positive form CANNOT change sign, so it spans only sign-preserving
# reassignments. Measured on Llama-3.2-1B layer 0: of LNQ's 22.33% non-nearest
# choices, 29% cross zero and are unreachable -- 3.53% of weights costing ~33%
# energy, because a sign crossing is the strongest available cancellation.
# Signed G reaches 100% of targets.
SIGNED_G=${SIGNED_G:-true}
SIGNED_EPS=${SIGNED_EPS:-0.05}   # flip resistance; damping = (1+eps)/eps = 21x
LAMBDA_TV=${LAMBDA_TV:-0.0}      # TV penalty: shrinks toward G=0, not toward flip

FREEZE_CB=false
FREEZE_SC=false
case "${CELL:-D}" in
  A) FREEZE_CB=true;  FREEZE_SC=true  ;;   # must reproduce LNQ exactly
  B) FREEZE_CB=false; FREEZE_SC=true  ;;   # codebook refinement only
  C) FREEZE_CB=true;  FREEZE_SC=false ;;   # pure escape from LNQ
  D) FREEZE_CB=false; FREEZE_SC=false ;;   # joint
  *) echo "CELL must be one of A B C D"; exit 1 ;;
esac

python layerwise_nuq.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --solver lnqflexnu \
  --seed_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 \
  --num_iterations "$NUM_ITER" --cd_cycles "$CD_CYCLES" \
  --flexnu_iters "$ITERS" \
  --flexnu_lr_scale "$LR_SCALE" \
  --flexnu_lr_cb "$LR_CB" \
  --flexnu_row_block "$ROW_BLOCK" \
  --flexnu_tau_frac "$TAU_FRAC" \
  --flexnu_stage_frac "$STAGE_FRAC" \
  --flexnu_eval_every "$EVAL_EVERY" \
  --flexnu_delta_init_noise "$DELTA_NOISE" \
  --flexnu_signed_g "$SIGNED_G" \
  --flexnu_signed_eps "$SIGNED_EPS" \
  --flexnu_lambda_tv "$LAMBDA_TV" \
  --flexnu_freeze_codebook "$FREEZE_CB" \
  --flexnu_freeze_scale "$FREEZE_SC" \
  $MODE_OPT