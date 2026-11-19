"""
mlp_ranges.py in 2 worker processes, like sglang's TP workers:
  - workers are started with "spawn", so they must inherit LD_PRELOAD and
    NVTX_INJECTION64_PATH from the environment
  - after cudaProfilerStop each worker is killed with SIGKILL, as sglang does,
    so its files must already be written at cudaProfilerStop
  - the parent never uses the GPU and writes no files

Run:
    ACCELPROF_CUPTI_CAPTURE=profiler_api accelprof -v -d cupti -t range_time python mlp_ranges_mp.py
    python check_expected.py mlp_ranges_mp 2
"""

import multiprocessing as mp
import os
import signal
import time

import torch

import mlp_ranges

NUM_WORKERS = 2


def worker(rank, done):
    mlp_ranges.run(f"cuda:{rank % torch.cuda.device_count()}")
    done.set()
    time.sleep(600)     # wait to be killed


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    workers = []
    for rank in range(NUM_WORKERS):
        done = ctx.Event()
        p = ctx.Process(target=worker, args=(rank, done))
        p.start()
        workers.append((p, done))

    for p, done in workers:
        done.wait()
        os.kill(p.pid, signal.SIGKILL)
        p.join()
        print(f"killed worker {p.pid}")
