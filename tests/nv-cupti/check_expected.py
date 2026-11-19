"""
Read range_time's output of mlp_ranges.py and print one line per range,
to compare with the test code (and with nsys). Also checks a few counts.

    python check_expected.py mlp_ranges          # single process
    python check_expected.py mlp_ranges_mp 2     # two worker processes
"""

import csv
import glob
import re
import sys
from collections import defaultdict

import mlp_ranges as test


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def module_and_tokens(text):
    # "{'Module': 'model.model.layers.1.mlp.gate', 'Inputs': [[1, 2048]]}" -> ("model.model.layers.N.mlp.gate", 1)
    m = re.search(r"'Module': '([^']+)', 'Inputs': \[\[(\d+)", text)
    return re.sub(r"layers\.\d+", "layers.N", m.group(1)), int(m.group(2))


def check(ok, what):
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    return ok


def check_process(prefix):
    ranges = read(prefix + ".ranges.csv")
    kernels = read(prefix + ".activities.csv")
    iters = test.PREFILL_ITERS + test.DECODE_ITERS
    ranges_per_iter = 1 + 3 * test.NUM_LAYERS      # model.model + (gate, topk, up) per layer

    ok = check(len(ranges) == iters * ranges_per_iter,
               f"{len(ranges)} ranges, expected {iters} iterations x {ranges_per_iter}")
    ok &= check(all(int(r["total_activities"]) > 0 for r in ranges), "every range ran GPU work")
    ok &= check(all(k["has_launch"] == "1" and k["range_id"] != "0" for k in kernels),
                "every kernel was launched inside a range")

    # one line per (range name, tokens)
    names_in_range = defaultdict(set)
    for k in kernels:
        names_in_range[k["range_id"]].add(k["short_name"])
    times = defaultdict(list)
    names = defaultdict(set)
    for r in ranges:
        key = module_and_tokens(r["name"])
        times[key].append(float(r["total_kern_us"]))
        names[key] |= names_in_range[r["range_id"]]

    print(f"  {'range':32s} {'tokens':>6s} {'count':>5s} {'GPU us':>9s}  kernels")
    for (module, tokens), values in sorted(times.items()):
        kernel_names = ", ".join(sorted(names[(module, tokens)])) or "(only in child ranges)"
        print(f"  {module:32s} {tokens:6d} {len(values):5d} {sum(values) / len(values):9.3f}  {kernel_names}")
    return ok


if __name__ == "__main__":
    app = sys.argv[1] if len(sys.argv) > 1 else "mlp_ranges"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    prefixes = sorted(p[: -len(".ranges.csv")] for p in glob.glob(f"{app}.*.range_time.ranges.csv"))
    ok = check(len(prefixes) == procs, f"{len(prefixes)} process(es) wrote files, expected {procs}")
    for prefix in prefixes:
        print(prefix)
        ok &= check_process(prefix)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
