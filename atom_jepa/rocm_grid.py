"""Exact coefficient-space spherical-grid product, with first-order autograd.

For each node and channel, split the input into coefficient vectors a and b.
The reference computes gate * P((T a) * (T b)), where T evaluates on a spherical
quadrature grid and P projects back. Precompute the fixed contraction

    coefficient[o, i, j] = sum_g P[o, g] T[g, i] T[g, j]

and evaluate all its terms directly. No quadrature coefficients are pruned.
The supported lmax=mmax=2 basis has (lmax+1)^2 = 9 coefficients. This restriction
comes from the tested model, not from a claim that all angular orders work.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

SUPPORTED_LMAX = 2
COEFFICIENT_COUNT = (SUPPORTED_LMAX + 1) ** 2
INPUT_BRANCHES = 2  # The two vectors multiplied on the spherical grid.
# Execution choices, separate from the mathematical dimensions. These were used
# in the bounded MI250X probes; they are not claimed optimal on every GPU/shape.
CHANNEL_TILE = 64
NUM_WARPS = 4


@triton.jit
def _forward(X, GATE, COEFF, OUT, UNGATED,
             CHANNELS: tl.constexpr, SPHERE: tl.constexpr, TILE: tl.constexpr):
    node = tl.program_id(0)
    output_component = tl.program_id(1)
    channel = tl.program_id(2) * TILE + tl.arange(0, TILE)
    valid = channel < CHANNELS
    input_width = 2 * CHANNELS
    input_base = node * SPHERE * input_width
    coefficient_base = output_component * SPHERE * SPHERE
    value = tl.full((TILE,), 0, tl.float32)
    for i in tl.static_range(SPHERE):
        a = tl.load(X + input_base + i * input_width + channel, valid, 0).to(tl.float32)
        for j in tl.static_range(SPHERE):
            b = tl.load(X + input_base + j * input_width + CHANNELS + channel, valid, 0).to(tl.float32)
            coupling = tl.load(COEFF + coefficient_base + i * SPHERE + j)
            value += coupling * a * b
    gate = tl.load(GATE + node * CHANNELS + channel, valid, 0).to(tl.float32)
    output_offset = (node * SPHERE + output_component) * CHANNELS + channel
    # Save the ungated value, not OUT / GATE: zero gates must differentiate safely.
    tl.store(UNGATED + output_offset, value, valid)
    tl.store(OUT + output_offset, value * gate, valid)


@triton.jit
def _backward_inputs(X, GATE, COEFF, GRAD_OUT, GRAD_X,
                     CHANNELS: tl.constexpr, SPHERE: tl.constexpr, TILE: tl.constexpr):
    node = tl.program_id(0)
    input_component = tl.program_id(1)
    channel = tl.program_id(2) * TILE + tl.arange(0, TILE)
    valid = channel < CHANNELS
    input_width = 2 * CHANNELS
    input_base = node * SPHERE * input_width
    grad_a = tl.full((TILE,), 0, tl.float32)
    grad_b = tl.full((TILE,), 0, tl.float32)
    # d/da_i: sum_(o,j) grad_o C[o,i,j] b_j; d/db_i uses C[o,j,i].
    for o in tl.static_range(SPHERE):
        grad = tl.load(GRAD_OUT + (node * SPHERE + o) * CHANNELS + channel, valid, 0).to(tl.float32)
        for j in tl.static_range(SPHERE):
            a = tl.load(X + input_base + j * input_width + channel, valid, 0).to(tl.float32)
            b = tl.load(X + input_base + j * input_width + CHANNELS + channel, valid, 0).to(tl.float32)
            coupling_a = tl.load(COEFF + o * SPHERE * SPHERE + input_component * SPHERE + j)
            coupling_b = tl.load(COEFF + o * SPHERE * SPHERE + j * SPHERE + input_component)
            grad_a += grad * coupling_a * b
            grad_b += grad * coupling_b * a
    gate = tl.load(GATE + node * CHANNELS + channel, valid, 0).to(tl.float32)
    offset = input_base + input_component * input_width + channel
    tl.store(GRAD_X + offset, grad_a * gate, valid)
    tl.store(GRAD_X + offset + CHANNELS, grad_b * gate, valid)


@triton.jit
def _backward_gate(GRAD_OUT, UNGATED, GRAD_GATE,
                   CHANNELS: tl.constexpr, SPHERE: tl.constexpr, TILE: tl.constexpr):
    node = tl.program_id(0)
    channel = tl.program_id(1) * TILE + tl.arange(0, TILE)
    valid = channel < CHANNELS
    value = tl.full((TILE,), 0, tl.float32)
    for o in tl.static_range(SPHERE):
        offset = (node * SPHERE + o) * CHANNELS + channel
        grad = tl.load(GRAD_OUT + offset, valid, 0).to(tl.float32)
        ungated = tl.load(UNGATED + offset, valid, 0)
        value += grad * ungated
    tl.store(GRAD_GATE + node * CHANNELS + channel, value, valid)


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, coefficient):
        # Layout: x [nodes, sphere, 2*channels], gate [nodes, 1, channels].
        x, gate = x.contiguous(), gate.contiguous()
        nodes, sphere, input_width = x.shape
        channels = input_width // INPUT_BRANCHES
        out = torch.empty((nodes, sphere, channels), device=x.device, dtype=x.dtype)
        ungated = torch.empty_like(out, dtype=torch.float32)
        if nodes:
            _forward[(nodes, sphere, triton.cdiv(channels, CHANNEL_TILE))](
                x, gate, coefficient, out, ungated, channels, sphere, CHANNEL_TILE, num_warps=NUM_WARPS)
        ctx.save_for_backward(x, gate, coefficient, ungated)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, gate, coefficient, ungated = ctx.saved_tensors
        grad = grad.contiguous()
        nodes, sphere, channels = grad.shape
        dx, dgate = torch.empty_like(x), torch.empty_like(gate)
        if nodes:
            _backward_inputs[(nodes, sphere, triton.cdiv(channels, CHANNEL_TILE))](
                x, gate, coefficient, grad, dx, channels, sphere, CHANNEL_TILE, num_warps=NUM_WARPS)
            _backward_gate[(nodes, triton.cdiv(channels, CHANNEL_TILE))](
                grad, ungated, dgate, channels, sphere, CHANNEL_TILE, num_warps=NUM_WARPS)
        return dx, dgate, None


class ROCmGridProduct(torch.nn.Module):
    def __init__(self, grid):
        super().__init__()
        to_grid = grid.to_grid_mat.detach().cpu().double()
        from_grid = grid.from_grid_mat.detach().cpu().double()
        if to_grid.shape[1] != COEFFICIENT_COUNT or from_grid.shape[0] != COEFFICIENT_COUNT:
            raise ValueError('only lmax=mmax=2 is supported')
        coefficient = torch.einsum('og,gi,gj->oij', from_grid, to_grid, to_grid).float().contiguous()
        # Derived from the reference grid; no trained state or checkpoint-key change.
        self.register_buffer('coeff', coefficient.to(grid.to_grid_mat.device), persistent=False)

    def forward(self, x, gate):
        if not x.is_cuda or not torch.version.hip:
            raise RuntimeError('ROCm GPU required')
        if x.ndim != 3 or x.shape[1] != COEFFICIENT_COUNT or x.shape[2] % INPUT_BRANCHES:
            raise ValueError('expected [nodes, 9, 2*channels]')
        if gate.numel() != x.shape[0] * x.shape[2] // INPUT_BRANCHES:
            raise ValueError('gate shape mismatch')
        if x.dtype not in (torch.float32, torch.bfloat16) or gate.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError('FP32/BF16 only')
        if self.coeff.dtype != torch.float32 or gate.device != x.device or self.coeff.device != x.device:
            raise ValueError('FP32 coefficients and same-device inputs required; use autocast, not module.bfloat16()')
        return _Product.apply(x, gate, self.coeff)


def enable_rocm_grid(body):
    """Attach derived operators; preserve weights and the usable reference path."""
    if body.lmax != SUPPORTED_LMAX or body.mmax != SUPPORTED_LMAX:
        raise ValueError('only lmax=mmax=2')
    body.set_grid_mlp_optimization(True)
    for block in body.blocks:
        pairs = [(block.ga.act, block.ga.act.so3_grid), (block.ffn.grid_mlp, block.ffn.so3_grid)]
        for layer, grid in pairs:
            dropout = getattr(layer, 'grid_drop', None)
            if not isinstance(dropout, torch.nn.Identity) and getattr(dropout, 'p', None) != 0:
                raise ValueError('grid dropout must be inactive')
            layer.gated_product = ROCmGridProduct(grid)
