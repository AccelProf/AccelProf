"""
Simplest test for nv-cupti + range_time.

One iteration = one call of forward(). It opens these NVTX ranges:

    model.model                        one iteration
      model.model.layers.0.mlp.gate    logits = x @ w_gate    (router GEMM)
      model.model.layers.0.mlp.topk    topk(logits, 8)        (router top-k)
      model.model.layers.0.mlp.up      x = x @ w_up           (GEMM)
      model.model.layers.1.mlp.gate    ...
      ...

The range text has sglang's format, {'Module': name, 'Inputs': [[tokens, dim]]},
so range_time's default regexes work without extra settings.

Recorded (between cudaProfilerStart and Stop):
    4 prefill iterations (512 tokens), then 16 decode iterations (1 token).

Run:
    ACCELPROF_CUPTI_CAPTURE=profiler_api accelprof -v -d cupti -t range_time python mlp_ranges.py
    python check_expected.py mlp_ranges
"""

import contextlib

import torch
import torch.cuda.profiler

NUM_LAYERS = 2
HIDDEN = 2048
NUM_EXPERTS = 128
TOP_K = 8
PREFILL_TOKENS = 512
WARMUP_ITERS = 3
PREFILL_ITERS = 4
DECODE_ITERS = 16


@contextlib.contextmanager
def nvtx_range(name, x):
    torch.cuda.nvtx.range_push(str({"Module": name, "Inputs": [list(x.shape)]}))
    yield
    torch.cuda.nvtx.range_pop()


def forward(x, weights):
    with nvtx_range("model.model", x):
        for i, (w_gate, w_up) in enumerate(weights):
            layer = f"model.model.layers.{i}.mlp"
            with nvtx_range(f"{layer}.gate", x):
                logits = x @ w_gate
            with nvtx_range(f"{layer}.topk", logits):
                torch.topk(logits, TOP_K, dim=-1)
            with nvtx_range(f"{layer}.up", x):
                x = x @ w_up
    return x


def run(device="cuda:0"):
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.02

    weights = [(randn(HIDDEN, NUM_EXPERTS), randn(HIDDEN, HIDDEN)) for _ in range(NUM_LAYERS)]
    x_prefill = randn(PREFILL_TOKENS, HIDDEN)
    x_decode = randn(1, HIDDEN)

    for _ in range(WARMUP_ITERS):       # not recorded
        forward(x_prefill, weights)
        forward(x_decode, weights)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()         # cudaProfilerStart: recording starts
    for _ in range(PREFILL_ITERS):
        forward(x_prefill, weights)
    for _ in range(DECODE_ITERS):
        forward(x_decode, weights)
    torch.cuda.synchronize()            # GPU must finish before Stop
    torch.cuda.profiler.stop()          # cudaProfilerStop: recording stops, files are written


if __name__ == "__main__":
    run()
    print("mlp_ranges done")
