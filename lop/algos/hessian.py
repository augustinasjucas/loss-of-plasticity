import torch
import torch.nn as nn
from typing import List, Tuple

def _effective_rank_from_eigvals(eigvals: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Given a 1D tensor of non-negative eigenvalues, compute the smallest k such that sum(eigvals[:k]) / sum(eigvals) >= threshold.
    This is the definition of effective.
    """
    vals = torch.sort(eigvals, descending=True).values
    cum = torch.cumsum(vals, dim=0)
    total = cum[-1]
    # find first index where cum/total >= threshold, i.e., cum >= threshold * total.
    k = torch.searchsorted(cum, threshold * total).item() + 1
    return k

def compute_effective_hessian_ranks(
    net: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    threshold: float = 0.99,
) -> Tuple[List[int], int]:
    """
    Compute per-layer and overall effective rank of the Hessian approximation.
    Adapted from https://arxiv.org/pdf/2312.00246.

    Returns:
        layer_ranks: List of ints, one per nn.Linear in net.layers
        net_rank:   Int, effective rank over all weights jointly
    """
    # Identify all weight-bearing layers and record their parameter shapes
    linear_layers = []
    for layer in net.layers:
        if isinstance(layer, nn.Linear):
            linear_layers.append(layer)
    
    n_samples = X.size(0)

    # per_layer_grads[layer_idx][weight_idx][sample_idx] = what effect on the loss does the weight "weight_idx" in layer "layer_idx" have on the loss for sample "sample_idx"
    per_layer_grads = [
        torch.zeros((layer.weight.numel(), n_samples), device=X.device)
        for layer in linear_layers
    ] 
    total_params = sum(layer.weight.numel() for layer in linear_layers)
    per_sample_all = torch.zeros((total_params, n_samples), device=X.device)

    # Loop over samples and accumulate gradients
    for i in range(n_samples):
        net.zero_grad()
        xi = X[i : i + 1]  # shape (1, x_dim)
        yi = y[i : i + 1]  # shape (1,)
        out = net.predict(xi)[0]
        loss = loss_fn(out, yi)
        loss.backward()
        
        # Save the gradiens both per-layer and for the whole network
        offset = 0
        for idx, layer in enumerate(linear_layers):
            g = layer.weight.grad.detach().reshape(-1)  # flatten
            per_layer_grads[idx][:, i] = g
            per_sample_all[offset : offset + g.numel(), i] = g
            offset += g.numel()
    
    # Compute effective ranks of layers
    layer_ranks = []
    for G in per_layer_grads:
        # The squared singular values of G are the eigenvalues of G^TG and GG^T.
        # https://math.stackexchange.com/questions/2152751/why-does-the-eigenvalues-of-aat-are-the-squares-of-the-singular-values-of-a
        # And all we need are the eigenvalues of GG^T, as the effective rank is computed from them.
        sv = torch.linalg.svdvals(G)
        eig = sv.pow(2)
        layer_ranks.append(_effective_rank_from_eigvals(eig, threshold))
    
    # Compute effective rank of the whole network
    sv_all = torch.linalg.svdvals(per_sample_all)
    eig_all = sv_all.pow(2)
    net_rank = _effective_rank_from_eigvals(eig_all, threshold)
    
    return layer_ranks, net_rank
