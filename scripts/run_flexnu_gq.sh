#!/usr/bin/env bash
# FlexNu solver for the GuidedQuant objective.
# Same positional args as run_lnq.sh:  <model_ref> <bits> <num_groups> [-m mode]
#
#   bash scripts/run_flexnu_gq.sh meta-llama/Llama-2-7b-hf 3 1
#
# Hessians are solver-independent -- cache them once with run_lnq.sh -m hessians
# and both solvers reuse them.
#
# ABLATION LADDER (run before trusting any perplexity number):
#   CELL=A  ... freeze both  -> MUST reproduce init nearest-codeword energy
#   CELL=B  ... codebook only (fits W)
#   CELL=C  ... divisor only (escapes via the loss)
#   CELL=D  ... joint (default)
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

# ---- FlexNu hyperparameters -------------------------------------------------
# lr_cb is deliberately ~300x smaller than lr_scale: the reference measured
# monotone degradation as lr_cb rises, because the codebook chases W and drags
# the thresholds out from under the divisor.
ITERS=${ITERS:-300}
LR_SCALE=${LR_SCALE:-3e-3}
LR_CB=${LR_CB:-1e-5}
ROW_BLOCK=${ROW_BLOCK:-64}
TAU_FRAC=${TAU_FRAC:-0.5}
STAGE_FRAC=${STAGE_FRAC:-0.0}     # 0.0 = joint. Nonzero collapses the effect.
EVAL_EVERY=${EVAL_EVERY:-1}       # 1 is required: hard energy is non-monotone
                                  # and good assignments vanish in a few steps.

# ---- ablation cell ----------------------------------------------------------
FREEZE_CB=false
FREEZE_SC=false
case "${CELL:-D}" in
  A) FREEZE_CB=true;  FREEZE_SC=true  ;;   # sanity: must reproduce init exactly
  B) FREEZE_CB=false; FREEZE_SC=true  ;;   # codebook only
  C) FREEZE_CB=true;  FREEZE_SC=false ;;   # divisor only
  D) FREEZE_CB=false; FREEZE_SC=false ;;   # joint
  *) echo "CELL must be one of A B C D"; exit 1 ;;
esac

python layerwise_nuq.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --solver flexnu \
  --seed_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 \
  --flexnu_iters "$ITERS" \
  --flexnu_lr_scale "$LR_SCALE" \
  --flexnu_lr_cb "$LR_CB" \
  --flexnu_row_block "$ROW_BLOCK" \
  --flexnu_tau_frac "$TAU_FRAC" \
  --flexnu_stage_frac "$STAGE_FRAC" \
  --flexnu_eval_every "$EVAL_EVERY" \
  --flexnu_freeze_codebook "$FREEZE_CB" \
  --flexnu_freeze_scale "$FREEZE_SC" \
  $MODE_OPT