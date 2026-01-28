# project/src/training/ddp_utils.py
"""DDP utilities for distributed training."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DDPInfo:
    rank: int
    local_rank: int
    world_size: int
    is_distributed: bool


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, None)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def get_ddp_info() -> DDPInfo:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    return DDPInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=world_size > 1,
    )


def is_main_process() -> bool:
    return get_ddp_info().rank == 0


def ddp_barrier() -> None:
    info = get_ddp_info()
    if info.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_ddp(backend: Optional[str] = None, init_method: Optional[str] = None) -> Tuple[torch.device, DDPInfo]:
    """
    Initialize torch.distributed if WORLD_SIZE > 1.

    Returns:
        (device, ddp_info)
    """
    info = get_ddp_info()

    # device selection
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
        device = torch.device(f"cuda:{info.local_rank}")
        backend = backend or "nccl"
    else:
        device = torch.device("cpu")
        backend = backend or "gloo"

    if not info.is_distributed:
        return device, info

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,  # typically env://
        )

    return device, info


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def pick_free_port() -> int:
    """Helper for single-node launch scripts."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port
