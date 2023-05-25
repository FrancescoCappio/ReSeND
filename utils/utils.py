import torch
from tqdm import tqdm

def get_coreset_idx(
    z_lib : torch.Tensor, 
    n : int = 1000,
    eps : float = 0.90,
    float16 : bool = True,
    force_cpu : bool = False,
) -> torch.Tensor:
    """Returns n coreset idx for given z_lib.
    
    Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
    CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
    Args:
        z_lib:      (n, d) tensor of patches.
        n:          Number of patches to select.
        eps:        Agression of the sparse random projection.
        float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
        force_cpu:  Force cpu, useful in case of GPU OOM.
    Returns:
        coreset indices
    """

    select_idx = 0
    last_item = z_lib[select_idx:select_idx+1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)
    # The line below is not faster than linalg.norm, although i'm keeping it in for
    # future reference.
    # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in range(n-1):
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) # broadcasting step
        # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
        min_distances = torch.minimum(distances, min_distances) # iterative step
        select_idx = torch.argmax(min_distances) # selection step

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    return torch.stack(coreset_idx)
