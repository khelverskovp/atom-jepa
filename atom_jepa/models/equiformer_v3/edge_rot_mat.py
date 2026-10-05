import torch


"""
Per-edge SE(3) rotation-frame construction for EquiformerV3.

We build, for each edge, a rotation that aligns the edge's relative-position
vector with a canonical axis (the model then performs SO(2) attention in that
frame). Aligning the edge fixes 2 of the 3 rotational DOF; the remaining DOF is
the rotation ABOUT the edge axis, which is a gauge freedom: the SO(2)
convolution is SO(2)-equivariant about that axis, so the network output does not
depend on how the perpendicular frame is chosen.

The upstream implementation chose the perpendicular reference vector RANDOMLY and
asserted it was not parallel to the edge. Over a large number of edges (e.g.
encoding all of QM9 for the probe in one pass) the unlucky case where every
candidate random vector is ~parallel to some edge eventually occurs and the
assertion fails. We instead pick the perpendicular reference DETERMINISTICALLY as
the world axis least aligned with the edge, which is always well-conditioned
(cross-product magnitude >= ~0.44) and never degenerate -- so there is no random
draw, no assertion, and the result is bit-reproducible. Because the perpendicular
frame is a gauge freedom, this does not change the model's (equivariant) output;
it only removes the crash and the per-forward jitter from the random frame.

    For gradient methods, we do not backpropagate rotation if the y component of
    the unit relative-position vector is very close to `_ROTATION_MASK_THRESHOLD`.
"""
_ROTATION_MASK_THRESHOLD = 0.999999


def init_edge_rot_mat(edge_distance_vec, use_rotation_mask=False):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0 ** 2, dim=1))

    # Defensive: a (near-)zero-length edge would make norm_x NaN and poison the
    # whole batch. Clamp the distance for the division and give any degenerate
    # edge a fixed unit axis. Such edges (overlapping atoms) should already be
    # removed upstream by build_radius_graph's min-distance floor; this guard just
    # guarantees the frame never becomes NaN and avoids the upstream debug spam.
    degenerate = (edge_vec_0_distance < 1e-6).view(-1, 1)
    norm_x = edge_vec_0 / edge_vec_0_distance.clamp_min(1e-6).view(-1, 1)
    norm_x = torch.where(degenerate, norm_x.new_tensor([1.0, 0.0, 0.0]).expand_as(norm_x), norm_x)

    if use_rotation_mask:
        # For gradient methods: snap near-y edges exactly to +/- y so the frame
        # is stable and rotation is not backpropagated through there.
        yprod = norm_x @ norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod > _ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod < -_ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0, -1.0, 0.0])

    # Deterministic, always-well-conditioned perpendicular reference: use the
    # world x-axis unless the edge is too aligned with it, in which case use y.
    # |norm_x . ref| is then bounded away from 1, so the cross products below are
    # never degenerate (no random vector, no alignment assertion needed).
    e_x = norm_x.new_tensor([1.0, 0.0, 0.0]).expand_as(norm_x)
    e_y = norm_x.new_tensor([0.0, 1.0, 0.0]).expand_as(norm_x)
    dot_x = torch.abs(torch.sum(norm_x * e_x, dim=1, keepdim=True))
    ref = torch.where(dot_x < 0.9, e_x, e_y)

    norm_z = torch.cross(norm_x, ref, dim=1)
    norm_z = norm_z / torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True))

    # Construct the 3D rotation matrix (same axis assignment / signs as upstream)
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    if use_rotation_mask:
        return edge_rot_mat
    else:
        return edge_rot_mat.detach()