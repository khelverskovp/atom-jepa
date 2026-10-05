"""CUDA contractions for the validated ADMET lmax=mmax=2 execution path.

Operators contain only derived constants. Their buffers are non-persistent so
optimized and ordinary models have identical checkpoint parameter/buffer keys.
"""
import math
import numpy as np
from cuequivariance.segmented_polynomials.subscripts import Subscripts
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet
import cuequivariance_ops_torch  # Register CUDA kernels; fail if unavailable.


def derived_operator(operator):
    if list(operator.parameters()):
        raise RuntimeError("ADMET execution operators must not add trainable parameters")
    for module in operator.modules():
        module._non_persistent_buffers_set.update(module._buffers)
    operator._admet_derived_operator = True
    return operator

class CompactGatherScaleRotate(torch.nn.Module):

    def __init__(self, rotation, channels, method='uniform_1d', *, device):
        super().__init__()
        S = (rotation.lmax + 1) ** 2
        out_l = [int(math.isqrt(i)) for i in rotation.wigner_index_to_m_array.argmax(1).tolist()]
        products = []
        for side in range(2):
            d = cue.SegmentedTensorProduct.from_subscripts(Subscripts(',u,u,u'))
            for _ in range(S * S):
                d.add_segment(0, ())
            for _ in range(S):
                d.add_segment(1, (channels,))
            for _ in range(2 * (rotation.lmax + 1)):
                d.add_segment(2, (channels,))
            for _ in range(2 * S):
                d.add_segment(3, (channels,))
            for out, degree in enumerate(out_l):
                for inp in range(degree ** 2, (degree + 1) ** 2):
                    d.add_path(out * S + inp, inp, 2 * degree + side, 2 * out + side, c=np.asarray(1.0))
            products.append(d)
        poly = cue.SegmentedPolynomial(inputs=[products[0].operands[i] for i in [0, 1, 1, 2]], outputs=[products[0].operands[3]], operations=[([0, 1, 3, 4], products[0]), ([0, 2, 3, 4], products[1])])
        self.f = cuet.SegmentedPolynomial(poly, method=method, output_dtype_map=[1]).to(device)

    def forward(self, x, scale, wigner, edges):
        out = self.f([wigner.flatten(1), x.flatten(1), x.flatten(1), scale.flatten(1)], input_indices={1: edges[0].contiguous(), 2: edges[1].contiguous()})[0]
        return out.view(scale.shape[0], x.shape[1], 2 * x.shape[2])

class WeightedRotateReduce(torch.nn.Module):

    def __init__(self, rotation, heads, channels, fp32_output=False, *, device):
        super().__init__()
        S = (rotation.lmax + 1) ** 2
        in_l = [int(math.isqrt(i)) for i in rotation.wigner_index_to_m_array.argmax(1).tolist()]
        d = cue.SegmentedTensorProduct.from_subscripts(Subscripts(',u,,u'))
        for _ in range(S * S):
            d.add_segment(0, ())
        for _ in range(S * heads):
            d.add_segment(1, (channels,))
        for _ in range(heads):
            d.add_segment(2, ())
        for _ in range(S * heads):
            d.add_segment(3, (channels,))
        for out in range(S):
            for inp, l in enumerate(in_l):
                if l != math.isqrt(out):
                    continue
                for h in range(heads):
                    d.add_path(out * S + inp, inp * heads + h, h, out * heads + h, c=np.asarray(1.0))
        self.f = cuet.SegmentedPolynomial(cue.SegmentedPolynomial(inputs=d.operands[:3], outputs=[d.operands[3]], operations=[([0, 1, 2, 3], d)]), method='uniform_1d', output_dtype_map=[0 if fp32_output else 1]).to(device)
        self.fp32_output = fp32_output
        self.S, self.C = (S, heads * channels)

    def forward(self, x, alpha, wigner, targets, nodes):
        wigner = wigner.float() if self.fp32_output else wigner
        out = self.f([wigner.flatten(1), x.flatten(1), alpha.flatten(1)], output_indices={0: targets.contiguous()}, output_shapes={0: nodes.flatten(1)})[0]
        return out.view(nodes.shape[0], self.S, self.C).to(x.dtype)

class InitialRotateReduce(torch.nn.Module):

    def __init__(self, channels=256, scale=16.0, fp32_output=False, sparse=False, *, device):
        super().__init__()
        d = cue.SegmentedTensorProduct.from_subscripts(Subscripts(',u,u'))
        for _ in range(81):
            d.add_segment(0, ())
        for _ in range(3):
            d.add_segment(1, (channels,))
        for _ in range(9):
            d.add_segment(2, (channels,))
        for o in range(9):
            for i in [math.isqrt(o)] if sparse else range(3):
                d.add_path(o * 9 + i, i, o, c=np.asarray(1.0 / scale))
        self.f = cuet.SegmentedPolynomial(cue.SegmentedPolynomial(inputs=d.operands[:2], outputs=[d.operands[2]], operations=[([0, 1, 2], d)]), method='uniform_1d', output_dtype_map=[0 if fp32_output else 1]).to(device)
        self.fp32_output = fp32_output

    def forward(self, w, x, targets, nodes):
        if torch.is_autocast_enabled('cuda'):
            dtype = torch.get_autocast_dtype('cuda')
            w, x = (w.to(dtype), x.to(dtype))
        if self.fp32_output:
            w = w.float()
        shape = x.new_empty((nodes, 9 * x.shape[-1]))
        return self.f([w.flatten(1), x.flatten(1)], output_indices={0: targets.contiguous()}, output_shapes={0: shape})[0].view(nodes, 9, x.shape[-1]).to(x.dtype)

class GatedGridProduct(torch.nn.Module):

    def __init__(self, grid, channels, *, device):
        super().__init__()
        T = grid.to_grid_mat.detach().cpu().double()
        P = grid.from_grid_mat.detach().cpu().double()
        C = torch.einsum('og,gi,gj->oij', P, T, T)
        S = T.shape[1]
        d = cue.SegmentedTensorProduct.from_subscripts(Subscripts('u,u,u,u'))
        for _ in range(2 * S):
            d.add_segment(0, (channels,))
            d.add_segment(1, (channels,))
        d.add_segment(2, (channels,))
        for _ in range(S):
            d.add_segment(3, (channels,))
        for o, i, j in (C.abs() > 1e-07).nonzero().tolist():
            d.add_path(2 * i, 2 * j + 1, 0, o, c=np.asarray(float(C[o, i, j])))
        self.f = cuet.SegmentedPolynomial(cue.SegmentedPolynomial(inputs=[d.operands[0], d.operands[2]], outputs=[d.operands[3]], operations=[([0, 0, 1, 2], d)]), method='uniform_1d', output_dtype_map=[0]).to(device)

    def forward(self, x, gate):
        return self.f([x.flatten(1), gate.flatten(1)])[0].view(x.shape[0], x.shape[1], x.shape[2] // 2)
