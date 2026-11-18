"""
Two worker processes running mlp_ranges, like sglang's TP workers:
  - workers are started with "spawn", so they must inherit LD_PRELOAD and
    NVTX_INJECTION64_PATH from the environment
  - after cudaProfilerStop each worker is killed with SIGKILL (no atexit),
    so the data must already be on disk from the flush at cudaProfilerStop
  - the parent never touches CUDA and should write no files

Run:
    ACCELPROF_CUPTI_CAPTURE=profiler_api accelprof -v -d cupti -t range_time python mlp_ranges_mp.py
    python check_expected.py --app mlp_ranges_mp --procs 2
"""

import argparse
import multiprocessing as mp
import os
import signal
import time

import mlp_ranges

NUM_WORKERS = 2


def worker(rank, args, stopped):
    import torch
    device = f"cuda:{rank % torch.cuda.device_count()}"
    print(f"[worker {rank}] pid {os.getpid()} on {device}, "
          f"NVTX_INJECTION64_PATH={os.environ.get('NVTX_INJECTION64_PATH')}", flush=True)
    mlp_ranges.run(device, args.warmup, args.prefill_iters, args.decode_iters)
    stopped.set()
    time.sleep(3600)    # wait to be killed


if __name__ == "__main__":
    args = mlp_ranges.add_args(argparse.ArgumentParser()).parse_args()
    ctx = mp.get_context("spawn")

    workers = []
    for rank in range(NUM_WORKERS):
        stopped = ctx.Event()
        p = ctx.Process(target=worker, args=(rank, args, stopped))
        p.start()
        workers.append((p, stopped))

    for p, stopped in workers:
        if not stopped.wait(timeout=600):
            print(f"worker {p.pid} did not finish")
        time.sleep(1)
        os.kill(p.pid, signal.SIGKILL)      # same as sglang's kill_process_tree
        p.join()
        print(f"killed worker {p.pid}")
    print("mlp_ranges_mp done")
