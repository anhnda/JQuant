"""
Sequential dirty-stream LNQ / GuidedQuant driver.
=================================================

The stock pipeline (`layerwise_main.py`) is *layer-wise*:

    1. Build ALL block Hessians from ONE clean forward pass  (activations.py)
    2. Quantize every block independently from the cached Hessians  (seed)

Every block is therefore quantized as if it will receive the *clean* FP16
input X^(l).  At inference it actually receives the *dirty* input
X~^(l) = output of the already-quantized upstream blocks, so the cached
Hessian H = X^T diag(s) X is built from the wrong input covariance.

This driver fuses the two phases into a single front-to-back sweep so that
each block is quantized against the REAL dirty stream:

    for block l in 0..L-1:
        X~^(l) = inps                       # dirty input (upstream already fake-quant)
        H~_k   = X~^(l)^T diag(s_k) X~^(l)   # dirty, saliency-weighted Hessian
        LNQ(H~_k, W_l) -> labels, codebook   # UNCHANGED solver (seed_layer)
        W_l  <- dequant(labels, codebook)    # fake-quant: snap to grid, keep fp16
        overwrite block l weights in the live model with W_l
        inps <- block_l(inps)                # forward THROUGH the quantized block
                                             #   -> this is X~^(l+1)

The LNQ solver itself (`seed_layer` / `train_least_squares`) is reused
verbatim.  Only the data feeding each Hessian changes, exactly as intended:
target stays the FP16 weight, end-loss enters only through the saliency s_k,
and there is no clean X*W output-reconstruction term.

Saliency modes
--------------
    --sal_mode clean  (default): s_k is the end-loss gradient at the ORIGINAL
        clean model, computed once and cached (this is the pure GuidedQuant
        quantity).  Only X~ is dirtied.  Cheap: no per-block backward.
    --sal_mode dirty : s_k is recomputed by a backward pass through the
        partially-quantized model at each block.  Consistent second-order
        object, but O(L) to-end backwards.  (Not implemented here yet; hook
        point marked below.)

Only the clean-saliency path is wired here because it fixes the dominant
term (input-covariance drift) at ~1x forward cost and reuses the existing
per-block saliency cache produced by the standard `hessians` phase.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from .activations import (
    get_inps,
    SaliencyEngine,
    _find_sublayers,
    _wrap_sublayers,
    _unwrap_sublayers,
)
from .layerwise_quantize import seed_layer, fix_hessian_shape, load_progress
from .utils import get_progress_bar


# --------------------------------------------------------------------------- #
# Hessian construction for ONE block from the CURRENT (dirty) inps            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _build_block_hessians(
    layer: nn.Module,
    module_names: List[str],
    inps: torch.Tensor,               # (N, seq_len, hidden) dirty input to this block
    layer_saliencies: Dict[str, torch.Tensor],   # {module -> (N, seq_len, G)}
    device: torch.device,
    **forward_args,
) -> Dict[str, torch.Tensor]:
    """
    Reuse SaliencyEngine to accumulate H_k = X~^T diag(s_k) X~ for every
    quantizable sub-module of `layer`, by running ONE forward of the block
    over `inps` with the sub-layers wrapped.  Different sub-modules
    (q/k/v vs o vs gate/up vs down) automatically get the correct input
    space because the wrapper sits on each nn.Linear and sees whatever that
    linear actually receives during the block forward.
    """
    layer = layer.to(device)
    found = _find_sublayers(layer)
    sublayers = {nm: found[nm] for nm in module_names if nm in found}

    engines: Dict[str, SaliencyEngine] = {}
    for nm, sub in sublayers.items():
        if nm not in layer_saliencies:
            raise ValueError(f"No saliency for sub-layer '{nm}'")
        engines[nm] = SaliencyEngine(
            sub.weight.shape[1], layer_saliencies[nm],
            dtype=torch.float32, device=device,
        )

    _wrap_sublayers(layer, engines)

    proc = {}
    for k, v in forward_args.items():
        if isinstance(v, torch.Tensor):
            proc[k] = v.to(device, non_blocking=True)
        elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
            proc[k] = tuple(x.to(device, non_blocking=True) for x in v)
        else:
            proc[k] = v

    for i in range(len(inps)):
        layer(inps[i].to(device).unsqueeze(0), **proc)

    _unwrap_sublayers(layer)

    return {nm: eng.XTX.detach().cpu().float() for nm, eng in engines.items()}


# --------------------------------------------------------------------------- #
# Dequantize (labels, codebook) -> fp16 weight, for fake-quant overwrite      #
# --------------------------------------------------------------------------- #
def _dequant(parent_weights: np.ndarray, lut: np.ndarray) -> torch.Tensor:
    """
    parent_weights (labels): (out, 1, in) uint8
    lut (codebook):          (out, 1, K)  float
    returns fp16 weight (out, in) = codebook gathered by label.
    """
    labels = torch.from_numpy(np.ascontiguousarray(parent_weights)).long().squeeze(1)  # (out, in)
    C = torch.from_numpy(np.ascontiguousarray(lut)).float().squeeze(1)                  # (out, K)
    W_hat = torch.gather(C, 1, labels)                                                  # (out, in)
    return W_hat


# --------------------------------------------------------------------------- #
# Forward the dirty input through the (now fake-quantized) block -> next inps #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _forward_block(
    layer: nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    device: torch.device,
    **forward_args,
):
    layer = layer.to(device)
    proc = {}
    for k, v in forward_args.items():
        if isinstance(v, torch.Tensor):
            proc[k] = v.to(device, non_blocking=True)
        elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
            proc[k] = tuple(x.to(device, non_blocking=True) for x in v)
        else:
            proc[k] = v
    for j in range(len(inps)):
        out = layer(inps[j].to(device).unsqueeze(0), **proc)[0]
        outs[j].copy_(out.reshape_as(outs[j]), non_blocking=True)


# --------------------------------------------------------------------------- #
# Main sequential sweep                                                        #
# --------------------------------------------------------------------------- #
def seed_sequential(
    analyzer,
    module_names: List[str],
    tokens,
    saliency_path: str,
    initialization_path: str,
    output_folder: str,
    seed_precision: int,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    num_groups: int = 1,
    solver: str = "lnq",
    flexnu_kwargs: dict = None,
    sal_mode: str = "clean",
):
    assert sal_mode in ("clean", "dirty"), f"unknown sal_mode {sal_mode}"
    if sal_mode == "dirty":
        return _seed_sequential_dirty_saliency(
            analyzer, module_names, tokens, initialization_path, output_folder,
            seed_precision, num_iterations, cd_cycles, num_groups, solver, flexnu_kwargs,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_count = 1

    from .layerwise_quantize import get_saver
    layer_saver = get_saver(output_folder, seed_precision, seed_precision, module_names)

    layers_to_process, completed = load_progress(
        output_folder, seed_precision, seed_precision, analyzer.num_layers
    )
    if completed:
        logging.info(f"[seq] resuming; already-done blocks skipped for QUANT, "
                     f"but still forwarded to keep the dirty stream correct: {completed}")

    # -- catch block-0 inputs (clean, since nothing quantized yet) --
    model_seqlen = tokens[0].shape[-1]
    data = tokens if tokens[0].dim() == 2 else [t.unsqueeze(0) for t in tokens]
    inps_list, forward_args = get_inps(
        analyzer=analyzer, data=data, model_seqlen=model_seqlen,
        devices=[device], offload_activations=True,
    )
    inps = inps_list[0]                      # single-device tensor (N, seq, hidden)
    outs = torch.zeros_like(inps)

    layers = analyzer.get_layers()
    num_layers = len(layers)

    pb = get_progress_bar(num_layers, "Sequential dirty-stream LNQ")

    for l in range(num_layers):
        layer = layers[l]

        # ---- 1. build DIRTY Hessians from current inps (skip if already quantized) ----
        if l not in completed:
            sal_file = os.path.join(saliency_path, f"l{l}.pt")
            loaded = torch.load(sal_file)
            orig_g = list(loaded.values())[0].shape[-1]
            assert orig_g % num_groups == 0
            sub = orig_g // num_groups
            loaded = {k: v.view(v.shape[0], v.shape[1], num_groups, sub).mean(-1)
                      for k, v in loaded.items()}

            hess = _build_block_hessians(
                layer, module_names, inps, loaded, device, **forward_args,
            )

            # ---- 2. run UNCHANGED LNQ solver on the dirty Hessians ----
            model_layer = [
                analyzer.get_layer_weights(l)[nm].float().numpy() for nm in module_names
            ]
            init_labels = torch.load(
                os.path.join(initialization_path, "weights", f"l{l}.pt"))
            init_centroids = torch.load(
                os.path.join(initialization_path, f"lut_{seed_precision}", f"l{l}.pt"))
            init_labels_layer = [init_labels[nm] for nm in module_names]
            init_centroids_layer = [init_centroids[nm].astype(np.float32) for nm in module_names]
            hessian_layer = [fix_hessian_shape(hess[nm]).float().numpy() for nm in module_names]

            luts_by_bit_by_module, parent_weights, log_dict = seed_layer(
                l, module_names, model_layer,
                init_labels_layer, init_centroids_layer, hessian_layer,
                seed_precision, group_count,
                num_iterations=num_iterations, cd_cycles=cd_cycles,
                solver=solver, flexnu_kwargs=flexnu_kwargs,
            )

            # ---- 3. persist results (same cache format as stock pipeline) ----
            layer_saver(luts_by_bit_by_module, parent_weights, log_dict, l)
        else:
            # already-quantized on a previous run: reload its labels+codebook
            luts_by_bit_by_module, parent_weights = _reload_quant(
                output_folder, seed_precision, module_names, l)

        # ---- 4. FAKE-QUANT OVERWRITE: dequant and write back into the live model ----
        modules = analyzer.get_modules(layer)
        for m_idx, nm in enumerate(module_names):
            W_hat = _dequant(parent_weights[m_idx], luts_by_bit_by_module[m_idx][0])
            mod = modules[nm]
            W_hat = W_hat.to(mod.weight.dtype).to(mod.weight.device)
            assert W_hat.shape == mod.weight.shape, \
                f"dequant shape {tuple(W_hat.shape)} != weight {tuple(mod.weight.shape)} ({nm})"
            with torch.no_grad():
                mod.weight.data.copy_(W_hat)
        # invalidate any cached fp32 copy of this layer's weights
        analyzer._model_weights.pop(l, None)

        # ---- 5. forward THROUGH the now-quantized block -> next block's dirty input ----
        _forward_block(layer, inps, outs, device, **forward_args)
        layer.to("cpu")
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        pb.update(1)

    pb.close()
    logging.info("[seq] Sequential dirty-stream quantization complete.")


def _reload_quant(output_folder, seed_precision, module_names, l):
    """Reload a previously-saved block's (codebook, labels) for overwrite on resume."""
    lut = torch.load(f"{output_folder}/lut_{seed_precision}/l{l}.pt")
    w = torch.load(f"{output_folder}/weights/l{l}.pt")
    luts_by_bit_by_module = [[lut[nm]] for nm in module_names]
    parent_weights = [w[nm] for nm in module_names]
    return luts_by_bit_by_module, parent_weights


# =========================================================================== #
#  DIRTY-SALIENCY sequential sweep                                            #
#  --------------------------------------------------------------------------- #
#  Per block l (input -> output order):                                        #
#    (a) REAL END LOSS: run the FULL model (blocks 0..l-1 already fake-quant-  #
#        overwritten, blocks l..L-1 still original) with labels=input_ids, so  #
#        outputs.loss is the genuine next-token causal-LM cross-entropy at the #
#        LM head -- the same end loss GuidedQuant uses. loss.backward() gives  #
#        d(end loss)/d z^(l) at every module output of block l, flowing back   #
#        through the (clean) downstream blocks and the head.                   #
#    (b) reduce that gradient exactly like gradients.py: (1e3*grad)^2, grouped #
#        channel-mean -> saliency (N, seq_len, num_groups).                    #
#    (c) build the DIRTY Hessian  H~ = X~^T diag(s_dirty) X~  from the current #
#        dirty input to block l and this dirty saliency.                       #
#    (d) run the UNCHANGED LNQ solver.                                         #
#    (e) FAKE-QUANT OVERWRITE block l (dequant -> .data.copy_), so the next    #
#        block's forward+backward flows through the now-quantized block l.     #
#                                                                              #
#  The whole model stays resident and differentiable throughout, because the  #
#  backward reaches the head at every step. Cost: L full fwd+bwd passes over   #
#  the calib set (fine for 1B; this is the expensive-but-faithful path).       #
# =========================================================================== #
@torch.enable_grad()
def _capture_block_saliency(
    analyzer,
    layer_idx: int,
    module_names: List[str],
    tokens,
    num_groups: int,
    device: torch.device,
):
    """
    Real end-loss saliency for ONE block, at the CURRENT (partially-quantized)
    operating point. Returns {module_name -> (N, seq_len, num_groups)} float,
    matching the layout that _build_block_hessians / SaliencyEngine expect.
    """
    model = analyzer.model
    layer = analyzer.get_layers()[layer_idx]
    modules = analyzer.get_modules(layer)

    # chunk-lists per module, exactly like gradients.py
    saliency_chunks: Dict[str, list] = {nm: [] for nm in module_names}

    hooks = []

    def make_fwd_hook(nm):
        def fwd_hook(module, inp, out):
            # `out` may be a tuple for some modules; linear returns a tensor
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()

            def grad_hook(grad):
                bsz, seq_len, hidden = grad.shape
                gsz = hidden // num_groups
                gsq = (grad.float() * 1e3).pow(2).view(bsz, seq_len, num_groups, gsz)
                saliency_chunks[nm].append(gsq.mean(-1).float().cpu())
            t.register_hook(grad_hook)
        return fwd_hook

    for nm, mod in modules.items():
        if nm in saliency_chunks:
            hooks.append(mod.register_forward_hook(make_fwd_hook(nm)))

    # Ensure params require grad so autograd builds the graph to the head.
    # (Quantized upstream weights were written via .data.copy_ under no_grad;
    #  they remain leaves and act as fixed constants in the graph -- exactly the
    #  dirty operating point. We do not differentiate through the quantizer.)
    prev_req = {}
    for n, p in model.named_parameters():
        prev_req[n] = p.requires_grad
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    was_training = model.training
    model.eval()  # eval mode (no dropout) but grad still flows

    data = tokens if tokens[0].dim() == 2 else [t.unsqueeze(0) for t in tokens]
    for seq in data:
        seq = seq.to(model.device if hasattr(model, "device") else device)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        out = model(input_ids=seq, labels=seq)   # REAL end loss (next-token CE at head)
        out.loss.backward()
        model.zero_grad(set_to_none=True)         # clear weight grads; saliency already captured

    for h in hooks:
        h.remove()
    for n, p in model.named_parameters():
        p.requires_grad_(prev_req[n])
    if was_training:
        model.train()

    return {nm: torch.cat(saliency_chunks[nm], dim=0) for nm in module_names}


def _seed_sequential_dirty_saliency(
    analyzer,
    module_names: List[str],
    tokens,
    initialization_path: str,
    output_folder: str,
    seed_precision: int,
    num_iterations: int,
    cd_cycles: int,
    num_groups: int,
    solver: str,
    flexnu_kwargs: dict,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_count = 1

    from .layerwise_quantize import get_saver
    layer_saver = get_saver(output_folder, seed_precision, seed_precision, module_names)

    layers_to_process, completed = load_progress(
        output_folder, seed_precision, seed_precision, analyzer.num_layers
    )
    if completed:
        logging.info(f"[seq-dirty] resuming; done blocks re-forwarded to keep "
                     f"the dirty stream + dirty backward correct: {completed}")

    # whole model must stay resident & differentiable: move it to GPU once
    model = analyzer.model
    model.to(device)

    # catch block-0 dirty input (clean at this point) for the Hessian geometry
    model_seqlen = tokens[0].shape[-1]
    data = tokens if tokens[0].dim() == 2 else [t.unsqueeze(0) for t in tokens]
    inps_list, forward_args = get_inps(
        analyzer=analyzer, data=data, model_seqlen=model_seqlen,
        devices=[device], offload_activations=True,
    )
    inps = inps_list[0]
    outs = torch.zeros_like(inps)

    layers = analyzer.get_layers()
    num_layers = len(layers)

    pb = get_progress_bar(num_layers, "Sequential dirty-stream + dirty-saliency LNQ")

    for l in range(num_layers):
        layer = layers[l]

        if l not in completed:
            # ---- (a)+(b) REAL end-loss DIRTY saliency at current operating point ----
            dirty_sal = _capture_block_saliency(
                analyzer, l, module_names, tokens, num_groups, device,
            )   # {nm -> (N, seq_len, num_groups)}

            # ---- (c) DIRTY Hessian from dirty input + dirty saliency ----
            hess = _build_block_hessians(
                layer, module_names, inps, dirty_sal, device, **forward_args,
            )

            # ---- (d) UNCHANGED LNQ solver ----
            model_layer = [
                analyzer.get_layer_weights(l)[nm].float().numpy() for nm in module_names
            ]
            init_labels = torch.load(
                os.path.join(initialization_path, "weights", f"l{l}.pt"))
            init_centroids = torch.load(
                os.path.join(initialization_path, f"lut_{seed_precision}", f"l{l}.pt"))
            init_labels_layer = [init_labels[nm] for nm in module_names]
            init_centroids_layer = [init_centroids[nm].astype(np.float32) for nm in module_names]
            hessian_layer = [fix_hessian_shape(hess[nm]).float().numpy() for nm in module_names]

            luts_by_bit_by_module, parent_weights, log_dict = seed_layer(
                l, module_names, model_layer,
                init_labels_layer, init_centroids_layer, hessian_layer,
                seed_precision, group_count,
                num_iterations=num_iterations, cd_cycles=cd_cycles,
                solver=solver, flexnu_kwargs=flexnu_kwargs,
            )
            layer_saver(luts_by_bit_by_module, parent_weights, log_dict, l)
        else:
            luts_by_bit_by_module, parent_weights = _reload_quant(
                output_folder, seed_precision, module_names, l)

        # ---- (e) FAKE-QUANT OVERWRITE block l ----
        modules = analyzer.get_modules(layer)
        for m_idx, nm in enumerate(module_names):
            W_hat = _dequant(parent_weights[m_idx], luts_by_bit_by_module[m_idx][0])
            mod = modules[nm]
            W_hat = W_hat.to(mod.weight.dtype).to(mod.weight.device)
            assert W_hat.shape == mod.weight.shape, \
                f"dequant shape {tuple(W_hat.shape)} != weight {tuple(mod.weight.shape)} ({nm})"
            with torch.no_grad():
                mod.weight.data.copy_(W_hat)
        analyzer._model_weights.pop(l, None)

        # ---- advance the dirty INPUT stream through the now-quantized block ----
        _forward_block(layer, inps, outs, device, **forward_args)
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        pb.update(1)

    pb.close()
    logging.info("[seq-dirty] Sequential dirty-stream + dirty-saliency quantization complete.")