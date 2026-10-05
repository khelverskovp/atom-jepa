"""Radius graphs for molecules and periodic crystals, plus minimum-image folding.

Edge convention (both kinds): edge_index is [E, 2] = (i=center, j=neighbor),
edge_vectors are unit vectors pointing i -> j, edge_lengths = |r_j - r_i|.
"""

import torch


def radius_graph(coords, cutoff, min_dist=0.5):
    """Non-periodic radius graph over all atom pairs with min_dist < d < cutoff."""
    n = coords.size(0)
    if n < 2:
        return (torch.zeros(0, 2, dtype=torch.long), torch.zeros(0), torch.zeros(0, 3))
    d = torch.cdist(coords, coords)
    mask = (d < cutoff) & (d > min_dist)
    i, j = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([i, j], dim=1)
    vec = coords[j] - coords[i]
    r = vec.norm(dim=1)
    u = vec / r.clamp_min(1e-9).unsqueeze(-1)
    return edge_index, r, u


def pbc_radius_graph(positions, cell, cutoff, pbc=True, numbers=None, min_dist=1e-3):
    """Periodic radius graph via ASE's neighbour list.

    ASE returns the displacement D = r_j(image) - r_i directly (image offsets
    folded in), so periodic images are handled exactly, including multiple
    images of the same neighbour for small cells / large cutoffs. Self-image
    neighbours (i == j across a cell boundary) are real periodic contacts and are
    kept; only the zero-distance self-pair is excluded.

    `numbers` only builds the ASE Atoms object; it does not affect the geometry.
    """
    import numpy as np
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    pos = positions.detach().cpu().numpy() if torch.is_tensor(positions) else np.asarray(positions)
    cl = cell.detach().cpu().numpy() if torch.is_tensor(cell) else np.asarray(cell)
    n = int(pos.shape[0])
    empty = (torch.zeros(0, 2, dtype=torch.long), torch.zeros(0), torch.zeros(0, 3))
    if n == 0:
        return empty

    if numbers is None:
        z = np.ones(n, dtype=np.int64)
    else:
        z = numbers.detach().cpu().numpy() if torch.is_tensor(numbers) else np.asarray(numbers)
    atoms = Atoms(numbers=z, positions=pos, cell=cl, pbc=pbc)

    i, j, D = neighbor_list("ijD", atoms, float(cutoff))   # D = r_j(image) - r_i, points i->j
    if len(i) == 0:
        return empty

    i = torch.from_numpy(i).long()
    j = torch.from_numpy(j).long()
    D = torch.from_numpy(np.ascontiguousarray(D)).float()
    r = D.norm(dim=1)
    keep = r > min_dist
    if not bool(keep.all()):
        i, j, D, r = i[keep], j[keep], D[keep], r[keep]
    u = D / r.clamp_min(1e-9).unsqueeze(-1)
    edge_index = torch.stack([i, j], dim=1)
    return edge_index, r, u


def min_image_delta(delta, cell, cell_index=None, refine=True):
    """Fold Cartesian displacements to their shortest periodic image.

    delta      : [M, 3] Cartesian displacements (r_target - r_reference)
    cell       : [3, 3] or [G, 3, 3] row-vector lattice (cart = frac @ cell)
    cell_index : [M] mapping each row of `delta` to a cell (required when G > 1
                 and the rows are not already cell-aligned)

    Returns [M, 3]: the representative of `delta` modulo the lattice with the
    smallest Euclidean norm. Rotation-equivariant and translation-consistent, so
    it is a valid geometric input to an equivariant predictor.
    """
    if delta.numel() == 0:
        return delta
    if cell.dim() == 2:
        cell, cell_index = cell.unsqueeze(0), None

    cell64 = cell.to(torch.float64)
    if bool((torch.linalg.det(cell64).abs() < 1e-8).any()):
        raise ValueError(
            "min_image_delta got a singular (zero-volume) cell; the minimum-image "
            "convention needs a full-rank lattice. Check that every structure in "
            "the dataset carries a real 3x3 cell."
        )
    cinv = torch.linalg.inv(cell64).to(delta.dtype)

    if cell_index is not None:
        C, Ci = cell[cell_index], cinv[cell_index]
    elif cell.size(0) == 1:
        C = cell.expand(delta.size(0), 3, 3)
        Ci = cinv.expand(delta.size(0), 3, 3)
    else:
        C, Ci = cell, cinv                      # already row-aligned

    frac = torch.einsum("mi,mij->mj", delta, Ci)
    frac = frac - frac.round()                  # into [-0.5, 0.5)^3
    if not refine:
        return torch.einsum("mi,mij->mj", frac, C)

    # Rounding the fractional vector is NOT always the shortest image for
    # non-orthogonal cells, so test the 27 neighbouring images and keep the
    # shortest. Exact for any cell within one lattice step of its reduced form.
    s = torch.tensor([-1.0, 0.0, 1.0], device=delta.device, dtype=delta.dtype)
    shifts = torch.cartesian_prod(s, s, s)                       # [27, 3]
    cand = torch.einsum("mki,mij->mkj", frac.unsqueeze(1) + shifts, C)   # [M,27,3]
    k = cand.norm(dim=-1).argmin(dim=1)
    return cand.gather(1, k.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
