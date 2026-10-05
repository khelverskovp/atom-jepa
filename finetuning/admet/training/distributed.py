"""Distributed helpers for ADMET training, enabled by torchrun environment variables.

Training is sharded across ranks; rank 0 evaluates the full validation/test sets so each
molecule's conformers stay together. Batch size is per rank; the effective batch size
scales with world size. Single-process calls use inert defaults."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.distributed as dist


@dataclass
class DistState:
    """Rank, device, and world-size settings, with single-process defaults."""
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """Rank 0. Guards eval, checkpoint writes, wandb, and printing."""
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def is_main_process() -> bool:
    """Return whether RANK is zero, without requiring process-group initialization."""
    return _env_int("RANK", 0) == 0


def init_distributed(device_hint: str = "cuda", timeout_hours: float = 2.0) -> DistState:
    """Initialize a torchrun process group, or return single-process defaults. CUDA falls
    back to CPU when unavailable."""
    world_size = _env_int("WORLD_SIZE", 1)
    want_cuda = (device_hint == "cuda") and torch.cuda.is_available()

    if world_size <= 1:
        return DistState(False, 0, 0, 1, torch.device("cuda" if want_cuda else "cpu"))

    if not want_cuda:
        # nccl needs CUDA. Refuse loudly rather than silently falling back to a
        # gloo CPU run that would be far slower than the single-process path.
        raise RuntimeError(
            f"WORLD_SIZE={world_size} (torchrun) but CUDA is unavailable or "
            f"misc.device={device_hint!r}. Multi-GPU requires CUDA; run without "
            "torchrun for a single-process CPU/GPU job."
        )

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank % max(1, torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(hours=timeout_hours),
        )
        # Destroy the process group at exit, not between seeds; avoid an exit barrier.
        import atexit
        atexit.register(lambda: dist.destroy_process_group()
                        if dist.is_initialized() else None)
    state = DistState(True, rank, local_rank, world_size, torch.device("cuda", local_rank))
    _verify_collectives(state)
    return state


def _verify_collectives(state: DistState) -> None:
    """Check a known all-reduce result at startup to detect broken collective transport."""
    expected = float(sum(range(1, state.world_size + 1)))
    probe = torch.full((8,), float(state.rank + 1), device=state.device)
    dist.all_reduce(probe)
    if not torch.allclose(probe, torch.full_like(probe, expected)):
        raise RuntimeError(
            f"[rank {state.rank}] NCCL SANITY CHECK FAILED: all_reduce returned "
            f"{probe[0].item()} but every rank must see {expected}.\n"
            "The collective did not error -- it returned CORRUPT DATA, so any "
            "training run started now would silently produce garbage.\n"
            "On this cluster the known cause is broken GPU peer-to-peer "
            "transport; the fix is to export NCCL_P2P_DISABLE=1 before "
            "torchrun (the JEPA submit scripts already do). Verify a node with:\n"
            "    sbatch scripts/diagnose_nccl.sh"
        )


def barrier(state: DistState) -> None:
    if state.enabled and dist.is_initialized():
        dist.barrier()


def all_reduce_sum_(t: torch.Tensor, state: DistState) -> torch.Tensor:
    """Sum a tensor across ranks in place, for epoch totals and label counts."""
    if state.enabled:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean_(t: torch.Tensor, state: DistState) -> torch.Tensor:
    """In-place MEAN across ranks."""
    if state.enabled:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= state.world_size
    return t


def average_gradients_(params, state: DistState) -> None:
    """Average gradients of parameters outside the DDP wrapper, such as MTLLoss.log_sigma."""
    if not state.enabled:
        return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= state.world_size
    off = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[off:off + n].view_as(g))
        off += n


def broadcast_obj(obj: Any, state: DistState, src: int = 0) -> Any:
    """Broadcast a picklable object from src so ranks share evaluation and stopping
    decisions."""
    if not state.enabled:
        return obj
    box: List[Any] = [obj if state.rank == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for direct access and checkpoints without DDP prefixes."""
    return getattr(model, "module", model)


def wrap_ddp(model: torch.nn.Module, state: DistState) -> torch.nn.Module:
    """Wrap the current trainable parameters in DDP. Rebuild after changing requires_grad,
    including encoder unfreezing."""
    if not state.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP
    core = unwrap(model)
    # device_ids applies only to CUDA DDP.
    on_cuda = any(p.is_cuda for p in core.parameters())
    kwargs = ({"device_ids": [state.local_rank], "output_device": state.local_rank}
              if on_cuda else {})
    return DDP(core, find_unused_parameters=False, **kwargs)


def loader_workers(requested: int, state: DistState) -> int:
    """Divide requested workers across local ranks, keeping at least one per rank when
    requested. Zero remains zero."""
    if not state.enabled or requested <= 0:
        return requested
    local_ws = _env_int("LOCAL_WORLD_SIZE", state.world_size)
    return max(1, requested // max(1, local_ws))


@contextmanager
def main_process_first(state: DistState):
    """Run on rank 0 before other ranks, for caches that must be built before reading."""
    if state.enabled and not state.is_main:
        barrier(state)
        yield
    else:
        yield
        if state.enabled:
            barrier(state)
