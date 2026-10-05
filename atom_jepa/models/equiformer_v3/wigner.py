import os
from functools import lru_cache
import torch


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))


@lru_cache(maxsize=128)
def _wigner_constants(l, device, dtype):
    """Cache only immutable constants, separately for each degree/device/dtype.

    Use the original FP64 J table for every dtype, avoiding a round trip through
    FP32. The bounded process-local cache is shared by model/EMA instances and
    never enters state_dict. Callers pass a tensor's concrete device (cuda:N).
    """
    # An inference-only first call must not leave inference tensors in the cache:
    # subsequent differentiable rotations may need to save J/frequencies.
    with torch.inference_mode(False), torch.no_grad():
        J = _Jd[l].to(dtype=dtype, device=device)
        inds = torch.arange(0, 2 * l + 1, device=device)
        reversed_inds = torch.arange(2 * l, -1, -1, device=device)
        frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    return J, inds, reversed_inds, frequencies


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    constants = _wigner_constants(l, alpha.device, alpha.dtype)
    J = constants[0]
    Xa = _z_rot_mat(alpha, l, constants)
    Xb = _z_rot_mat(beta, l, constants)
    Xc = _z_rot_mat(gamma, l, constants)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l, constants=None):
    if constants is None:
        constants = _wigner_constants(l, angle.device, angle.dtype)
    _, inds, reversed_inds, frequencies = constants
    M = angle.new_zeros((*angle.shape, 2 * l + 1, 2 * l + 1))
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M