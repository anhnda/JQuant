"""
CLI entry for SEQUENTIAL dirty-stream LNQ / GuidedQuant.

Mirrors layerwise_nuq.py but routes the quantize phase through the fused
sequential sweep in any_precision.quantization.layerwise_seq, so every block
is quantized against the real dirty input stream (upstream blocks already
fake-quantized) instead of the clean cached Hessians.

Pipeline:  [tokens] -> [saliency cache] -> [SEQUENTIAL: dirty-Hessian + LNQ + overwrite]
The standard clean Hessian phase is NOT used.
"""
import argparse
import functools
import logging
import os

import torch
torch.load = functools.partial(torch.load, weights_only=False)

from any_precision.analyzer import get_analyzer
from any_precision.quantization.config import (
    DEFAULT_SEED_PRECISION, DEFAULT_CACHE_DIR, DEFAULT_DATASET,
    DEFAULT_SEQ_LEN, DEFAULT_NUM_EXAMPLES,
)
from any_precision.quantization.datautils import get_tokens
from any_precision.quantization.layerwise_seq import seed_sequential


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def main():
    p = argparse.ArgumentParser("Sequential dirty-stream LNQ")
    p.add_argument("model", type=str)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--seed_precision", type=int, default=DEFAULT_SEED_PRECISION)
    p.add_argument("--yaml_path", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    p.add_argument("--num_groups", type=int, default=1)
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--cd_cycles", type=int, default=4)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--solver", type=str, default="lnq",
                   choices=["lnq", "flexnu", "lnqflexnu", "lnqbopt"])
    p.add_argument("--sal_mode", type=str, default="clean",
                   choices=["clean", "dirty"],
                   help="clean: cached end-loss saliency (GuidedQuant). "
                        "dirty: per-block backward (not implemented).")
    p.add_argument("--no_pack", action="store_true",
                   help="Stop after quantize; do not build the packed model.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    model_name = args.model_name or args.model.split("/")[-1]
    cd = args.cache_dir

    tokens_cache = (f"{cd}/tokens/"
                    f"{model_name}-{args.dataset}_s{args.num_examples}_blk{args.seq_len}.pt")
    saliency_cache = (f"{cd}/saliency/{model_name}"
                      f"-{args.dataset}_s{args.num_examples}_blk{args.seq_len}_g{args.num_groups}")
    init_cache = (f"{cd}/quantized/"
                  f"{model_name}-w{args.seed_precision}_orig{args.seed_precision}"
                  f"-{args.dataset}_s{args.num_examples}_blk{args.seq_len}")
    _sal_suffix = "_dsal" if args.sal_mode == "dirty" else ""
    out_cache = (f"{cd}/layerwise_quantized/"
                 f"{model_name}-w{args.seed_precision}"
                 f"-{args.dataset}_s{args.num_examples}_blk{args.seq_len}_g{args.num_groups}"
                 f"_iter{args.num_iterations}_cd{args.cd_cycles}"
                 f"{'' if args.solver == 'lnq' else '_' + args.solver}_seq{_sal_suffix}")

    logging.info(f"Tokens cache:       {tokens_cache}")
    logging.info(f"Saliency cache:     {saliency_cache}")
    logging.info(f"Initialization:     {init_cache}")
    logging.info(f"Output (seq):       {out_cache}")

    analyzer = get_analyzer(args.model, yaml_path=args.yaml_path, include_tokenizer=True)

    # ----- tokens -----
    logging.info("------------------- Get tokens -------------------")
    tokens = get_tokens(args.dataset, "train", analyzer.tokenizer,
                        args.seq_len, args.num_examples, tokens_cache, args.random_state)

    # ----- saliency cache: only needed for clean mode -----
    if args.sal_mode == "clean":
        need = [os.path.join(saliency_cache, f"l{i}.pt") for i in range(analyzer.num_layers)]
        if not all(os.path.exists(f) for f in need):
            logging.info("Saliency cache missing; generating clean end-loss saliency...")
            from any_precision.quantization.gradients import get_gradients
            os.makedirs(saliency_cache, exist_ok=True)
            get_gradients(
                analyzer, tokens,
                save_path=None, saliency_path=saliency_cache,
                num_groups=args.num_groups, skip_save_gradients=True,
            )
            # reload analyzer: gradient pass may have moved/hooked the model
            analyzer = get_analyzer(args.model, yaml_path=args.yaml_path, include_tokenizer=True)
    else:
        logging.info("sal_mode=dirty: saliency is computed LIVE per block via a "
                     "real end-loss backward through the partially-quantized model. "
                     "No clean saliency cache is used.")

    if not os.path.exists(init_cache):
        logging.error(f"Initialization cache {init_cache} missing. "
                      f"Run the SqueezeLLM-init step first (same as stock LNQ).")
        return

    # ----- sequential dirty-stream quantize -----
    logging.info("------------------- Sequential dirty-stream quantize -------------------")
    seed_sequential(
        analyzer=analyzer,
        module_names=analyzer.module_names,
        tokens=tokens,
        saliency_path=saliency_cache,
        initialization_path=init_cache,
        output_folder=out_cache,
        seed_precision=args.seed_precision,
        num_iterations=args.num_iterations,
        cd_cycles=args.cd_cycles,
        num_groups=args.num_groups,
        solver=args.solver,
        flexnu_kwargs={},
        sal_mode=args.sal_mode,
    )

    # ----- pack (unless --no_pack) -----
    if args.no_pack:
        logging.info(f"Quantize done. Packed model NOT built (--no_pack). "
                     f"LUT/labels are in: {out_cache}")
        return

    packed_out = (f"{cd}/layerwise_packed/"
                  f"layerwise-{model_name}-w{args.seed_precision}"
                  f"-{args.dataset}_s{args.num_examples}_blk{args.seq_len}_g{args.num_groups}"
                  f"_iter{args.num_iterations}_cd{args.cd_cycles}"
                  f"{'' if args.solver == 'lnq' else '_' + args.solver}_seq{_sal_suffix}")

    if os.path.exists(packed_out) and os.listdir(packed_out):
        logging.info(f"Packed model already exists at {packed_out}; skipping pack.")
        logging.info("Done.")
        return

    logging.info("------------------- Pack -------------------")
    # the overwrite step mutated in-memory weights; reload a clean analyzer so
    # pack reads the ORIGINAL fp16 shapes/metadata, then fills from the LUT cache.
    analyzer = get_analyzer(args.model, yaml_path=args.yaml_path, include_tokenizer=True)
    from any_precision.quantization.pack import pack
    # CRITICAL: drop original .weight keys before packing. pack() only ADDS
    # .qweight/.lut and never removes .weight, so if the originals stay in the
    # state_dict they (a) bloat the checkpoint and (b) get copied back over the
    # dequantized weights by dequant_to_hf.py's "copy unquantized remainder"
    # loop -> the exported HF model would silently be the ORIGINAL fp16 model.
    # The stock pipeline (layerwise_main.py) does this at line 193; the seq
    # path must too.
    analyzer.drop_original_weights()
    pack(
        analyzer=analyzer,
        lut_path=out_cache,
        output_model_path=packed_out,
        seed_precision=args.seed_precision,
        parent_precision=args.seed_precision,
        cpu_count=None,
    )
    logging.info(f"Done. Packed model: {packed_out}")


if __name__ == "__main__":
    main()