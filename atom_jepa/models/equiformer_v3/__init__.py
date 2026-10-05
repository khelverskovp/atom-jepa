"""Vendored EquiformerV3 component modules (from atomicarchitects/equiformer_v3,
experimental/models/equiformer_v3). The fairchem-coupled top-level model and the
energy/force/stress output heads are intentionally removed: we build our own
headless backbone in atom_jepa/models/eqv3_backbone.py. These components depend only on
torch, e3nn (so3.py) and torch_geometric (transformer_block.py, softmax.py).

EquiformerV3 is MIT-licensed (LICENSE in this folder); parts that originate in
fairchem are MIT-licensed too (LICENSE-fairchem).
"""
