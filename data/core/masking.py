"""k-hop ego partitions: split one sample into two complementary atom subsets.

The partition is chosen on a bond-scale TOPOLOGY graph, not the message-passing
graph: the dataset's covalent bonds (`bond_edge_index`) when it has them,
otherwise a radius graph at `topo_cutoff` (periodic for crystals). The two
resulting views are rebuilt afterwards at the model cutoff (see data/core/collate.py).
"""

from typing import Dict, List, Tuple

import torch

from data.core.graphs import pbc_radius_graph, radius_graph


def _build_adjacency(edge_index: torch.Tensor, n: int) -> List[List[int]]:
    """Undirected adjacency list from an [E,2] edge_index."""
    adj: List[List[int]] = [[] for _ in range(n)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[:, 0].tolist()
    dst = edge_index[:, 1].tolist()
    for a, b in zip(src, dst):
        adj[a].append(b)
        adj[b].append(a)  # treat hops as undirected
    return adj


def _bfs_hops(center: int, n: int, adj: List[List[int]]) -> List[int]:
    """Hop-distance from `center` to every node (unreachable -> n+1)."""
    INF = n + 1
    dist = [INF] * n
    dist[center] = 0
    frontier = [center]
    d = 0
    while frontier:
        d += 1
        nxt: List[int] = []
        for u in frontier:
            for v in adj[u]:
                if dist[v] == INF:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    return dist


def k_hop_ego_indices(center: int, edge_index: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Sorted node indices of the k-hop ego-net around `center`, CAPPED so that
    the complement is always non-empty.

    Takes the largest hop radius h <= k whose closed neighbourhood still leaves at
    least one node out, so on tiny / densely connected graphs k shrinks gracefully
    (down to h=0, just the center). Nodes unreachable from `center` always land
    in the complement.
    """
    if n <= 1:
        return torch.arange(n)
    adj = _build_adjacency(edge_index, n)
    dist = _bfs_hops(center, n, adj)
    reachable_max = max((d for d in dist if d <= n), default=0)
    hi = min(k, reachable_max)
    for h in range(hi, -1, -1):
        sel = [i for i in range(n) if dist[i] <= h]
        if len(sel) < n:  # non-empty complement
            return torch.tensor(sel, dtype=torch.long)
    return torch.tensor([center], dtype=torch.long)


def topology_edges(sample: Dict[str, torch.Tensor], topo_cutoff: float) -> torch.Tensor:
    """Bond-scale graph that defines the k-hop neighbourhoods.

    Uses sample["bond_edge_index"] when present; otherwise a radius graph at
    `topo_cutoff` (periodic when the sample has a `cell`). The radius version
    depends only on fixed geometry, so it is cached on the sample dict under the
    private key "_topo_proxy" -- a one-time cost when the dataset caches samples.
    """
    bonds = sample.get("bond_edge_index")
    if bonds is not None and bonds.numel() > 0:
        return bonds

    cached = sample.get("_topo_proxy")
    if cached is not None and cached[0] == topo_cutoff:
        return cached[1]

    if "cell" in sample:
        edges, _, _ = pbc_radius_graph(sample["node_coordinates"], sample["cell"], topo_cutoff,
                                       numbers=sample["atomic_numbers"])
    else:
        edges, _, _ = radius_graph(sample["node_coordinates"], topo_cutoff)
    sample["_topo_proxy"] = (topo_cutoff, edges)
    return edges


def ego_partition(sample: Dict[str, torch.Tensor], ego_hops, topo_cutoff: float
                  ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Split a sample into a k-hop ego subset and its complement.

    k is drawn uniformly from `ego_hops` (an int or a list), the center uniformly
    from the atoms. Returns (idx_a, idx_b, center): sorted atom indices of the ego
    subset, of its complement, and the ego center. A sample with fewer than two
    atoms cannot be split; both views are then the whole sample.
    """
    n = sample["node_coordinates"].size(0)
    if n < 2:
        idx = torch.arange(n)
        return idx, idx, 0

    topo = topology_edges(sample, topo_cutoff)
    ks = [int(ego_hops)] if isinstance(ego_hops, int) else [int(h) for h in ego_hops]
    k = ks[int(torch.randint(len(ks), (1,)))]
    center = int(torch.randint(n, (1,)))

    idx_a = k_hop_ego_indices(center, topo, n, k)
    mask = torch.ones(n, dtype=torch.bool)
    mask[idx_a] = False
    idx_b = torch.nonzero(mask, as_tuple=False).view(-1)
    return idx_a, idx_b, center
