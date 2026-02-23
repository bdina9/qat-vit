import torch
import torch.distributed as dist

# torchrun will have already set the env vars; just initialise.
if not dist.is_initialized():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

print("rank =", dist.get_rank())
print("world size =", dist.get_world_size())
