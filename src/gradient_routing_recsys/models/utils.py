"""Utility functions for gradient routing recommender models."""

import torch
from torch import Tensor


def sparsemax(input_tensor: Tensor, dim: int = -1) -> Tensor:
    """
    Sparsemax activation function.

    Sparsemax is a sparse alternative to softmax that can produce exactly zero probabilities.
    Reference: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    dim : int
        Dimension along which to apply sparsemax

    Returns
    -------
    torch.Tensor
        Sparsemax probabilities
    """
    # Translate input_tensor by max for numerical stability
    translated = input_tensor - input_tensor.max(dim=dim, keepdim=True)[0]

    # Sort input_tensor in descending order
    sorted_input, _ = torch.sort(translated, descending=True, dim=dim)

    # Calculate cumulative sum
    cumsum = torch.cumsum(sorted_input, dim=dim)

    # Create range vector
    num_classes = input_tensor.size(dim)
    k = torch.arange(
        1, num_classes + 1, device=input_tensor.device, dtype=input_tensor.dtype
    )

    # Reshape k to broadcast correctly with sorted_input
    k_shape = [1] * input_tensor.dim()
    k_shape[dim] = num_classes
    k = k.view(k_shape)

    # Find the support
    support = (k * sorted_input) > (cumsum - 1)

    # Get the support size
    support_size = support.sum(dim=dim, keepdim=True).float()

    # Calculate tau using index_select for each batch dimension
    # This is a simpler approach that avoids gather dimension issues
    batch_dims = input_tensor.dim() - 1

    # Reshape for easier indexing
    if dim == -1:
        dim = input_tensor.dim() - 1

    # Flatten all batch dimensions
    cumsum_flat = cumsum.reshape(-1, num_classes)
    support_size_flat = support_size.reshape(-1, 1)

    # Get indices (support_size - 1) for each sample, clamped to valid range
    idx = (support_size_flat - 1).clamp(min=0, max=num_classes - 1).long()

    # Use advanced indexing to get the values
    batch_idx = torch.arange(cumsum_flat.size(0), device=input_tensor.device).unsqueeze(
        1
    )
    tau_values = cumsum_flat[batch_idx, idx]

    # Reshape back to original shape
    tau_shape = list(input_tensor.shape)
    tau_shape[dim] = 1
    tau = tau_values.reshape(tau_shape)

    # Adjust tau calculation
    tau = (tau - 1) / (support_size + 1e-10)

    # Apply sparsemax
    output = torch.clamp(translated - tau, min=0)

    return output
