"""
Layer Residual PCA analyzer.

Given residual direction vectors saved from Residual Concept Detection,
compute singular values of the stacked matrix using PCA (via SVD).

Input directions format (Python side):
    {
        "layers.0.attn_out": [v0_0, v0_1, ..., v0_{d-1}],
        "layers.0.mlp_out":  [v1_0, v1_1, ..., v1_{d-1}],
        ...
    }

Output JSON-serializable dict:
    {
        "num_directions": N,
        "model_dim": D,
        "singular_values": [...],  # length = min(N, D), sorted desc
        "explained_variance": [...],  # optional, same length, sum to 1.0
    }
"""

from typing import Dict, Iterable, List, Mapping, Sequence

import torch  # type: ignore[import]


def _directions_to_matrix(directions: Mapping[str, Sequence[float]]) -> torch.Tensor:
    """Convert directions dict to (N, D) tensor on CPU."""
    rows: List[torch.Tensor] = []
    for vec in directions.values():
        try:
            t = torch.as_tensor(vec, dtype=torch.float32)
        except Exception:
            continue
        if t.ndim != 1:
            t = t.reshape(-1)
        rows.append(t)
    if not rows:
        raise ValueError("No valid direction vectors found.")
    # Pad shorter rows with zeros so all have same dim
    dim = max(r.shape[0] for r in rows)
    padded = []
    for r in rows:
        if r.shape[0] < dim:
          pad = torch.zeros(dim - r.shape[0], dtype=r.dtype)
          r = torch.cat([r, pad], dim=0)
        padded.append(r)
    return torch.stack(padded, dim=0)  # (N, D)


def compute_residual_pca_singular_values(
    directions: Mapping[str, Sequence[float]],
    max_components: int = 256,
) -> Dict[str, Iterable[float]]:
    """
    Compute singular values for residual directions.

    - Center matrix by subtracting mean over directions.
    - Use torch.linalg.svdvals on CPU.
    """
    mat = _directions_to_matrix(directions)  # (N, D)
    num_dirs, dim = int(mat.shape[0]), int(mat.shape[1])

    # Center
    mat = mat - mat.mean(dim=0, keepdim=True)

    # Compute singular values (sorted descending by default)
    with torch.no_grad():
        s = torch.linalg.svdvals(mat)  # (min(N, D),)

    # Truncate if requested
    k = min(len(s), int(max_components) if max_components else len(s))
    s = s[:k].cpu()

    # Explained variance (proportional to s^2)
    var = s**2
    total_var = float(var.sum().item()) or 1.0
    explained = (var / total_var).tolist()

    return {
        "num_directions": num_dirs,
        "model_dim": dim,
        "singular_values": s.tolist(),
        "explained_variance": explained,
    }

