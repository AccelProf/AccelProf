"""
Small MLP to test the nv-cupti backend and the range_time tool.

Every module pushes an NVTX range in the same text format as sglang's
--enable-layerwise-nvtx-marker, so the default range_time regexes work:
    {'Module': 'model.model.layers.0.mlp.gate', 'Inputs': [[512, 2048]]}

Module tree (4 layers, Qwen3-30B-A3B-like router shape):
    model                         Model
    model.model                   Body            <- one iteration
    model.model.layers.N          Layer           (residual add)
    model.model.layers.N.mlp      MLP
        .gate   Linear 2048->128                  (layer 1: on a side stream)
        .topk   softmax + topk(8)
        .up     Linear 2048->1024
        .act    SiLU
        .down   Linear 1024->2048
        .scale  one Triton kernel, layer 0 only   (launched with cuLaunchKernel)

Per measured iteration, outside the model:
    y + 1           one kernel outside any range
    "copy_out"      one DtoH memcpy inside a plain NVTX range

Run:
    ACCELPROF_CUPTI_CAPTURE=profiler_api accelprof -v -d cupti -t range_time python mlp_ranges.py
    python check_expected.py --app mlp_ranges --procs 1
"""

import argparse

import torch
import torch.cuda.nvtx
import torch.cuda.profiler
import torch.nn as nn

HIDDEN = 2048
NUM_EXPERTS = 128
TOP_K = 8
INTER = 1024
NUM_LAYERS = 4
SIDE_STREAM_LAYER = 1
DTYPE = torch.bfloat16

try:
    import triton
    import triton.language as tl

    @triton.jit
    def scale_kernel(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(x_ptr + offs, x * 2, mask=mask)

    def scale(x):
        n = x.numel()
        scale_kernel[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)
        return x

    HAS_TRITON = True
except ImportError:
    def scale(x):
        return x * 2

    HAS_TRITON = False


class NvtxHooks:
    """One NVTX range per module forward, text format of sglang's PytHooks."""

    def register(self, model, prefix="model"):
        for name, module in model.named_modules(prefix=prefix):
            module.register_forward_pre_hook(self._make_pre_hook(name))
            module.register_forward_hook(self._post_hook)

    @staticmethod
    def _make_pre_hook(name):
        def pre_hook(module, inputs):
            marker = {"Module": name}
            shapes = [list(x.shape) for x in inputs if torch.is_tensor(x)]
            if shapes:
                marker["Inputs"] = shapes
            torch.cuda.nvtx.range_push(str(marker))
        return pre_hook

    @staticmethod
    def _post_hook(module, inputs, output):
        torch.cuda.nvtx.range_pop()


class TopK(nn.Module):
    def forward(self, logits):
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return torch.topk(probs, TOP_K, dim=-1)


class Scale(nn.Module):
    def forward(self, x):
        return scale(x)


class MLP(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.topk = TopK()
        self.up = nn.Linear(HIDDEN, INTER, bias=False)
        self.act = nn.SiLU()
        self.down = nn.Linear(INTER, HIDDEN, bias=False)
        self.scale = Scale() if layer_id == 0 else None
        self.side_stream = None     # set in build_model() for SIDE_STREAM_LAYER

    def forward(self, x):
        if self.side_stream is not None:
            main_stream = torch.cuda.current_stream()
            self.side_stream.wait_stream(main_stream)
            with torch.cuda.stream(self.side_stream):
                logits = self.gate(x)
            main_stream.wait_stream(self.side_stream)
        else:
            logits = self.gate(x)
        self.topk(logits)
        h = self.down(self.act(self.up(x)))
        if self.scale is not None:
            h = self.scale(h)
        return h


class Layer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.mlp = MLP(layer_id)

    def forward(self, x):
        return x + self.mlp(x)


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(i) for i in range(NUM_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()

    def forward(self, x):
        return self.model(x)


def build_model(device):
    model = Model().to(device=device, dtype=DTYPE)
    model.model.layers[SIDE_STREAM_LAYER].mlp.side_stream = torch.cuda.Stream(device=device)
    NvtxHooks().register(model)
    return model


def run_iteration(model, x):
    y = model(x)
    z = y + 1                               # outside any range
    torch.cuda.nvtx.range_push("copy_out")
    z[0].cpu()                              # one DtoH memcpy
    torch.cuda.nvtx.range_pop()


def run(device, warmup, prefill_iters, decode_iters, prefill_tokens=512):
    torch.cuda.set_device(device)
    model = build_model(device)
    x_prefill = torch.randn(prefill_tokens, HIDDEN, device=device, dtype=DTYPE)
    x_decode = torch.randn(1, HIDDEN, device=device, dtype=DTYPE)

    with torch.no_grad():
        # warmup: cuBLAS handles, Triton compile, allocator
        for _ in range(warmup):
            run_iteration(model, x_prefill)
            run_iteration(model, x_decode)
        torch.cuda.synchronize()

        torch.cuda.profiler.start()         # cudaProfilerStart
        for _ in range(prefill_iters):
            run_iteration(model, x_prefill)
        for _ in range(decode_iters):
            run_iteration(model, x_decode)
        torch.cuda.synchronize()            # all GPU work done before Stop
        torch.cuda.profiler.stop()          # cudaProfilerStop


def add_args(parser):
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prefill-iters", type=int, default=4)
    parser.add_argument("--decode-iters", type=int, default=16)
    return parser


if __name__ == "__main__":
    args = add_args(argparse.ArgumentParser()).parse_args()
    print(f"triton: {HAS_TRITON}")
    run("cuda:0", args.warmup, args.prefill_iters, args.decode_iters)
    print("mlp_ranges done")
