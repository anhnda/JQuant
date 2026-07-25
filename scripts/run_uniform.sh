set -x

# Uniform LNQ + GuidedQuant.
#   $1 = MODEL_REF (e.g. Llama-2-7b-hf)
#   $2 = BITS
#   $3 = NUM_GROUPS (g)   -- GuidedQuant Hessian grouping; 1 = no grouping
#   $4/$5 = optional  -m <mode>   (tokens|hessians|quantize|pack)
#
# NOTE: unlike LNQ, this does NOT need SqueezeLLM init. The uniform grid is
# built from W directly (H-weighted MSE scale search). You still need the
# GuidedQuant Hessians cached first:
#     bash scripts/run_lnq.sh <MODEL> <BITS> <G> -m hessians
# (hessian cache is solver-independent and shared.)

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
SYMMETRIC=${SYMMETRIC:-true}     # set SYMMETRIC=false for asymmetric (learned zero-point)

python layerwise_nuq.py "$MODEL_PATH" \
  --model_name "$MODEL_REF" \
  --seed_precision "$BITS" \
  --dataset "$DATASET" --seq_len "$SEQ_LEN" --num_examples "$NUM_EXAMPLES" \
  --num_groups "$NUM_GROUPS" --random_state 42 \
  --solver uniform --uniform_symmetric "$SYMMETRIC" $MODE_OPT
