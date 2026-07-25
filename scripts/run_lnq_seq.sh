set -x
# ---------------------------------------------------------------------------
# Sequential dirty-stream LNQ / GuidedQuant.
#
# Same interface as run_lnq.sh:
#     bash scripts/run_lnq_seq.sh <MODEL_REF> <BITS> <NUM_GROUPS> [SAL_MODE]
#
# Difference from run_lnq.sh: instead of building all block Hessians from a
# single CLEAN forward pass and then quantizing every block independently,
# this sweeps front-to-back and quantizes each block against the DIRTY input
# stream (upstream blocks already fake-quantized + written back into the live
# model). The LNQ solver itself is unchanged.
#
#   MODEL_REF   e.g. Llama-3.2-1B   (resolved via resolve_model.py)
#   BITS        seed precision, e.g. 3
#   NUM_GROUPS  GuidedQuant Hessian groups g (1 = single saliency vector)
#   SAL_MODE    clean (default) | dirty   [dirty not yet implemented]
#
# Env overrides (same as run_lnq.sh): DATASET, SEQ_LEN, NUM_EXAMPLES
# ---------------------------------------------------------------------------

MODEL_REF=$1
BITS=$2
NUM_GROUPS=$3
SAL_MODE=${4:-clean}

MODEL_PATH=$(python resolve_model.py "$MODEL_REF") || exit $?

DATASET=${DATASET:-c4}
SEQ_LEN=${SEQ_LEN:-2048}
NUM_EXAMPLES=${NUM_EXAMPLES:-128}

python layerwise_nuq_seq.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --seed_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 \
  --sal_mode "$SAL_MODE"
