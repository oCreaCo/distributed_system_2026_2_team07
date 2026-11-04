"""Pairwise NCCL latency/bandwidth microbenchmark between the 8 pods.

Uses torch.distributed's NCCL backend (same NCCL + socket transport that DT-FM's cupy
communicator uses) so the numbers reflect what training traffic actually sees under tc.
RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT come from the environment (set by test.py).
Rank 0 writes a JSON file: {"pairs": {"i-j": {"latency_ms": ..., "bandwidth_gbps": ...}}}.

Every rank calls dist.barrier() the same number of times (once before and once after each
pair); only the two pair ranks issue send/recv in between. Mismatched collectives deadlock NCCL.
"""
import argparse
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def ping_pong(rank, a, b, tensor, reps):
    """Round trips a->b->a; returns seconds per one-way trip. Called only by ranks a and b."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        if rank == a:
            dist.send(tensor, b)
            dist.recv(tensor, b)
        else:
            dist.recv(tensor, a)
            dist.send(tensor, a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-mb", type=int, default=64)
    ap.add_argument("--bw-reps", type=int, default=3)
    ap.add_argument("--lat-reps", type=int, default=20)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=3))
    print(f"rank {rank}: process group ready", flush=True)
    small = torch.zeros(1, dtype=torch.float32, device="cuda")
    big = torch.zeros(args.size_mb * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")

    results = {}
    for a in range(world):
        for b in range(a + 1, world):
            dist.barrier()
            if rank in (a, b):
                ping_pong(rank, a, b, small, 2)  # warm up the pair's NCCL channel
                lat = ping_pong(rank, a, b, small, args.lat_reps)
                bw_t = ping_pong(rank, a, b, big, args.bw_reps)
                if rank == a:
                    results[f"{a}-{b}"] = {
                        "latency_ms": lat * 1e3,
                        "bandwidth_gbps": args.size_mb * 8 / 1e3 / bw_t,  # one-way MB -> Gbit/s
                    }
                    print(f"pair {a}-{b}: latency {lat * 1e3:.2f} ms, "
                          f"bandwidth {results[f'{a}-{b}']['bandwidth_gbps']:.2f} Gbit/s", flush=True)
            dist.barrier()

    # gather every pair's result to rank 0
    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    if rank == 0:
        merged = {}
        for g in gathered:
            merged.update(g)
        with open(args.out, "w") as f:
            json.dump({"size_mb": args.size_mb, "pairs": merged}, f, indent=1)
        print(f"wrote {args.out} ({len(merged)} pairs)", flush=True)
    dist.barrier()
    # rank 0 hosts the TCP store; a normal destroy_process_group() on the other ranks hangs once
    # rank 0 has exited, so leave without tearing the group down.
    os._exit(0)


if __name__ == "__main__":
    main()
