"""Turn dataset samples into model-ready graph batches.

A SAMPLE (what a dataset's __getitem__ returns, see data/README.md) holds only
atoms and geometry; graphs are built here, at batching time, with the model's
cutoff. A sample with a `cell` is a periodic crystal and gets a periodic radius
graph; otherwise it is a molecule.

A BATCH is one flat graph over all samples:
    atomic_numbers   [N] long        node_coordinates [N, 3]
    edge_index       [E, 2]          (center, neighbor), offset per graph
    edge_lengths     [E]             edge_vectors     [E, 3] (unit, center -> neighbor)
    node_graph_index [N]             num_graphs       int
    cell             [G, 3, 3]       (crystals only)
plus any graph-level labels the samples carry (e.g. y [G, P]).

Two collators:
    GraphCollator  one full graph per sample (fine-tuning, probes)
    JEPACollator   two complementary k-hop ego views per sample plus the full
                   graph, for JEPA pretraining
"""

from typing import Dict, List, Optional

import torch

from atom_jepa.data.graphs import pbc_radius_graph, radius_graph
from atom_jepa.data.masking import ego_partition

# Per-atom / per-structure sample fields. Every other tensor field on a sample
# is treated as a graph-level label and stacked along a new batch dimension.
STRUCTURE_KEYS = ("atomic_numbers", "node_coordinates", "cell", "bond_edge_index")


def featurize(sample: Dict[str, torch.Tensor], cutoff: float,
              keep: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Graph for a sample (or for the atoms `keep` of it) at radius `cutoff`.

    Crystals keep their original cell, so an atom subset stays a valid periodic
    structure (a partially occupied cell)."""
    z = sample["atomic_numbers"]
    pos = sample["node_coordinates"]
    if keep is not None:
        z, pos = z[keep], pos[keep]
    out = {"atomic_numbers": z, "node_coordinates": pos}
    if "cell" in sample:
        out["cell"] = sample["cell"]
        edge_index, r, u = pbc_radius_graph(pos, sample["cell"], cutoff, numbers=z)
    else:
        edge_index, r, u = radius_graph(pos, cutoff)
    out.update(edge_index=edge_index, edge_lengths=r, edge_vectors=u)
    return out


def collate_graphs(graphs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Batch featurized graphs into one flat graph with per-graph edge offsets.

    Concatenates whichever atom-identity fields the graphs carry
    (`atomic_numbers`, and the legacy one-hot `node_features` some fine-tuning
    datasets still attach), stacks per-graph `cell`s, and stacks `y` if present.
    """
    coords, edges, r, u, gidx = [], [], [], [], []
    nf, an, cells = [], [], []
    offset = 0
    has_nf = len(graphs) > 0 and "node_features" in graphs[0]
    has_an = len(graphs) > 0 and "atomic_numbers" in graphs[0]
    has_cell = len(graphs) > 0 and "cell" in graphs[0]
    for g, graph in enumerate(graphs):
        n = graph["node_coordinates"].size(0)
        coords.append(graph["node_coordinates"])
        edges.append(graph["edge_index"] + offset)
        r.append(graph["edge_lengths"])
        u.append(graph["edge_vectors"])
        gidx.append(torch.full((n,), g, dtype=torch.long))
        if has_nf:
            nf.append(graph["node_features"])
        if has_an:
            an.append(graph["atomic_numbers"])
        if has_cell:
            cells.append(graph["cell"])
        offset += n
    batch = {
        "node_coordinates": torch.cat(coords, dim=0) if coords else torch.zeros(0, 3),
        "edge_index": torch.cat(edges, dim=0) if edges else torch.zeros(0, 2, dtype=torch.long),
        "edge_lengths": torch.cat(r, dim=0) if r else torch.zeros(0),
        "edge_vectors": torch.cat(u, dim=0) if u else torch.zeros(0, 3),
        "node_graph_index": torch.cat(gidx, dim=0) if gidx else torch.zeros(0, dtype=torch.long),
        "num_graphs": len(graphs),
    }
    if has_nf:
        batch["node_features"] = torch.cat(nf, dim=0)
    if has_an:
        batch["atomic_numbers"] = torch.cat(an, dim=0)
    if has_cell:
        batch["cell"] = torch.stack(cells, dim=0)
    if len(graphs) > 0 and all("y" in g for g in graphs) and graphs[0]["y"].numel() > 0:
        batch["y"] = torch.stack([g["y"] for g in graphs], dim=0)
    return batch


def _clamp_z(sample, max_num_elements):
    """Clamp atomic numbers into the embedding table (copy only if needed)."""
    z = sample["atomic_numbers"]
    if z.numel() and int(z.max()) >= max_num_elements:
        return {**sample, "atomic_numbers": z.clamp_max(max_num_elements - 1)}
    return sample


class GraphCollator:
    """One full graph per sample, plus stacked graph-level labels.

    Any sample tensor outside STRUCTURE_KEYS (and not starting with "_") is a
    graph-level label and is stacked: y [1] -> [G, 1], a scalar -> [G].
    """

    def __init__(self, cutoff: float = 6.0, max_num_elements: int = 128):
        self.cutoff = cutoff
        self.max_num_elements = max_num_elements

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        samples = [_clamp_z(s, self.max_num_elements) for s in samples]
        batch = collate_graphs([featurize(s, self.cutoff) for s in samples])
        for key, value in samples[0].items():
            if key in STRUCTURE_KEYS or key.startswith("_") or not torch.is_tensor(value):
                continue
            batch[key] = torch.stack([s[key] for s in samples], dim=0)
        return batch


class JEPACollator:
    """Two complementary k-hop ego views per sample, plus the full graph.

    Returns (view_a, view_b, full_target):
        view_a / view_b      batches of the ego subsets and their complements
        full_target["full"]  batch of the whole samples (the target encoder's input)
        full_target["index_a"/"index_b"]  each view's atoms as rows of `full`
    Crystals additionally get full_target["anchor_a"/"anchor_b"] ([G], rows of
    `full`): one reference atom per side for the target-atom positional query,
    since a centroid of wrapped periodic coordinates is not a geometric centre.
    The ego center anchors view_a; a random atom of view_b anchors view_b.
    """

    def __init__(self, cutoff: float = 6.0, ego_hops=(2, 3), ego_topology_cutoff: float = 1.8,
                 max_num_elements: int = 128):
        self.cutoff = cutoff
        self.ego_hops = ego_hops
        self.ego_topology_cutoff = ego_topology_cutoff
        self.max_num_elements = max_num_elements

    def __call__(self, samples: List[Dict[str, torch.Tensor]]):
        samples = [_clamp_z(s, self.max_num_elements) for s in samples]
        periodic = len(samples) > 0 and "cell" in samples[0]

        a_graphs, b_graphs, full_graphs = [], [], []
        index_a, index_b, anchor_a, anchor_b = [], [], [], []
        offset = 0
        for s in samples:
            n = s["node_coordinates"].size(0)
            ia, ib, center = ego_partition(s, self.ego_hops, self.ego_topology_cutoff)
            a_graphs.append(featurize(s, self.cutoff, ia))
            b_graphs.append(featurize(s, self.cutoff, ib))
            full_graphs.append(featurize(s, self.cutoff))
            index_a.append(ia + offset)
            index_b.append(ib + offset)
            if periodic:
                # a sample too small to split has both anchors on atom 0
                ab = int(ib[int(torch.randint(ib.numel(), (1,)))]) if n >= 2 else 0
                anchor_a.append(center + offset)
                anchor_b.append(ab + offset)
            offset += n

        full_target = {
            "full": collate_graphs(full_graphs),
            "index_a": torch.cat(index_a) if index_a else torch.zeros(0, dtype=torch.long),
            "index_b": torch.cat(index_b) if index_b else torch.zeros(0, dtype=torch.long),
        }
        if periodic:
            full_target["anchor_a"] = torch.tensor(anchor_a, dtype=torch.long)
            full_target["anchor_b"] = torch.tensor(anchor_b, dtype=torch.long)
        return collate_graphs(a_graphs), collate_graphs(b_graphs), full_target


def move_batch(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
