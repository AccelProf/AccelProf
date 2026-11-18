"""
Check the range_time output of mlp_ranges.py / mlp_ranges_mp.py against what
the scripts are known to do. Use the same --prefill-iters / --decode-iters
as the test run.

    python check_expected.py --app mlp_ranges --procs 1
    python check_expected.py --app mlp_ranges_mp --procs 2
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import mlp_ranges

MODULE_RE = re.compile(r"'Module': '([^']+)'")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def module_of(name):
    m = MODULE_RE.search(name)
    return m.group(1) if m else name


def check_process(prefix, args):
    num_iters = args.prefill_iters + args.decode_iters
    ranges = read_csv(prefix + ".ranges.csv")
    activities = read_csv(prefix + ".activities.csv")
    iterations = read_csv(prefix + ".iterations.csv")
    groups = read_csv(prefix + ".groups.csv")
    module = {r["range_id"]: module_of(r["name"]) for r in ranges}
    results = []

    def check(what, ok, detail=""):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {what} {detail}")

    # iterations
    tokens = [int(r["num_tokens"]) for r in iterations]
    expected_tokens = [512] * args.prefill_iters + [1] * args.decode_iters
    check("iterations", tokens == expected_tokens, f"({len(tokens)} found, {num_iters} expected)")

    # ranges: model, model.model, 7 per layer, mlp.scale in layer 0, copy_out
    per_iter = 2 + 7 * mlp_ranges.NUM_LAYERS + 1 + 1
    check("range count", len(ranges) == per_iter * num_iters, f"({len(ranges)} found, {per_iter * num_iters} expected)")
    check("all ranges closed", all(int(r["cpu_end_ns"]) > 0 for r in ranges))

    # activities
    no_launch = [a for a in activities if a["has_launch"] != "1"]
    outside = [a for a in activities if a["has_launch"] == "1" and a["range_id"] == "0"]
    memcpy = [a for a in activities if a["short_name"] == "[memcpy DtoH]"]
    check("every activity has a launch record", not no_launch, f"({len(no_launch)} without)")
    check("one kernel outside ranges per iteration", len(outside) == num_iters, f"({len(outside)} found)")
    check("one DtoH memcpy per iteration, in copy_out",
          len(memcpy) == num_iters and all(module[a["range_id"]] == "copy_out" for a in memcpy),
          f"({len(memcpy)} found)")

    # per module
    by_suffix = defaultdict(list)
    for r in ranges:
        by_suffix[module[r["range_id"]].split(".")[-1]].append(int(r["total_activities"]))
    for suffix in ("gate", "topk", "up", "down"):
        least = min(by_suffix[suffix], default=0)
        check(f"every mlp.{suffix} has GPU work", least >= 1, f"(min {least})")
    check("mlp.act is exactly 1 kernel", set(by_suffix["act"]) == {1}, f"({sorted(set(by_suffix['act']))})")
    check("mlp.scale is exactly 1 kernel", set(by_suffix["scale"]) == {1}, f"({sorted(set(by_suffix['scale']))})")
    scale_names = {a["short_name"] for a in activities if module.get(a["range_id"], "").endswith("mlp.scale")}
    print(f"  [INFO] mlp.scale kernel: {sorted(scale_names)} (Triton if it says scale_kernel)")

    # side stream: gate of layer SIDE_STREAM_LAYER runs on another stream than layer 0
    def gate_streams(layer):
        name = f"model.model.layers.{layer}.mlp.gate"
        return {a["stream_id"] for a in activities if module.get(a["range_id"]) == name}
    s0, s1 = gate_streams(0), gate_streams(mlp_ranges.SIDE_STREAM_LAYER)
    check("side-stream gate attributed on its own stream", bool(s1) and not (s0 & s1),
          f"(layer0 {sorted(s0)}, layer1 {sorted(s1)})")

    # groups: gate + topk per layer per iteration
    expected_groups = num_iters * mlp_ranges.NUM_LAYERS * 2
    check("groups.csv rows", len(groups) == expected_groups, f"({len(groups)} found, {expected_groups} expected)")

    # timing summary for reference
    times = defaultdict(list)
    for g in groups:
        times[(g["component"], int(g["num_tokens"]))].append(float(g["kern_us"]))
    for (component, tokens), values in sorted(times.items()):
        print(f"  [INFO] {component:5s} tokens={tokens:4d} n={len(values):3d} mean={sum(values) / len(values):8.3f} us")
    return all(results)


if __name__ == "__main__":
    parser = mlp_ranges.add_args(argparse.ArgumentParser())
    parser.add_argument("--dir", default=".")
    parser.add_argument("--app", default="mlp_ranges")
    parser.add_argument("--procs", type=int, default=1)
    args = parser.parse_args()

    prefixes = sorted(p[: -len(".ranges.csv")]
                      for p in glob.glob(os.path.join(args.dir, f"{args.app}.*.range_time.ranges.csv")))
    print(f"found output of {len(prefixes)} process(es), expected {args.procs}")
    ok = len(prefixes) == args.procs
    for prefix in prefixes:
        print(os.path.basename(prefix))
        ok = check_process(prefix, args) and ok
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
