#!/usr/bin/env python3
"""
Extract key performance metrics from an Nsight Compute (.ncu-rep) report.

Metrics extracted per kernel invocation:
  - kernel_name      : demangled kernel name
  - mangled_name     : mangled (symbol) name
  - cycles           : elapsed cycles (gpc__cycles_elapsed.max)
  - global_ld_instr  : global memory load requests  (...)
  - global_st_instr  : global memory store requests (...)
  - shared_ld_instr  : shared memory load instructions  (...)
  - shared_st_instr  : shared memory store instructions (...)
  - local_ld_instr   : local memory load requests  (...)
  - local_st_instr   : local memory store requests (...)
  - l1_to_l2_ld_sectors  : L1->L2 read sectors  (...)
  - l1_to_l2_st_sectors  : L1->L2 write sectors (...)
  - l2_to_dram_ld_sectors: L2->global DRAM read sectors  (...)
  - l2_to_dram_st_sectors: L2->global DRAM write sectors (...)

Each sector is 32 bytes.  Multiply sector counts by 32 to get bytes.

Usage
-----
  python extract_ncu_metrics.py <report.ncu-rep> [options]

Options
-------
  --csv          Write CSV output instead of a formatted table
  --bytes        Convert sector counts to bytes (x32) in the output
  -o FILE        Write output to FILE instead of stdout

Example
-------
  python extract_ncu_metrics.py origin_100.ncu-rep
  python extract_ncu_metrics.py origin_100.ncu-rep --compare opt_100.ncu-rep --id-only
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ncu_report bootstrap
# ---------------------------------------------------------------------------
NCU_PYTHON_PATH = Path("/usr/local/cuda-13.0/nsight-compute-2025.3.0/extras/python")

if NCU_PYTHON_PATH.exists() and str(NCU_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(NCU_PYTHON_PATH))

try:
    import ncu_report
except ImportError as exc:
    sys.exit(
        f"ERROR: Cannot import ncu_report. "
        f"Ensure Nsight Compute is installed and the path is correct.\n{exc}"
    )

# ---------------------------------------------------------------------------
# Metric name constants
# ---------------------------------------------------------------------------
M_GRID_X        = "launch__grid_dim_x"
M_GRID_Y        = "launch__grid_dim_y"
M_GRID_Z        = "launch__grid_dim_z"
M_BLOCK_X       = "launch__block_dim_x"
M_BLOCK_Y       = "launch__block_dim_y"
M_BLOCK_Z       = "launch__block_dim_z"

# 使用 gpc__cycles_elapsed.max 获取真实的 kernel 执行周期长度
M_CYCLES        = "gpc__cycles_elapsed.max" 
M_GLOBAL_LD     = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
M_GLOBAL_ST     = "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
M_SHARED_LD     = "smsp__sass_inst_executed_op_shared_ld.sum"
M_SHARED_ST     = "smsp__sass_inst_executed_op_shared_st.sum"
M_LOCAL_LD      = "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum"
M_LOCAL_ST      = "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum"
M_L1_L2_LD      = "lts__t_sectors_srcunit_tex_op_read.sum"
M_L1_L2_ST      = "lts__t_sectors_srcunit_tex_op_write.sum"
M_L2_DRAM_LD    = "dram__sectors_read.sum"
M_L2_DRAM_ST    = "dram__sectors_write.sum"

SECTOR_BYTES = 32  # bytes per sector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric_value(action: "ncu_report.IAction", name: str):
    """Return the numeric value of a metric, or None if unavailable."""
    m = action.metric_by_name(name)
    if m is None:
        return None
    k = m.kind()
    if k in (ncu_report.IMetric.ValueKind_UINT64, ncu_report.IMetric.ValueKind_UINT32):
        return m.as_uint64()
    if k in (ncu_report.IMetric.ValueKind_FLOAT, ncu_report.IMetric.ValueKind_DOUBLE):
        return m.as_double()
    return None


def extract_actions(report_path: str | Path):
    """Yield one dict of metrics per kernel invocation in the report."""
    ctx = ncu_report.load_report(str(report_path))

    launch_id = 0
    for range_idx in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(range_idx)
        for action_idx in range(rng.num_actions()):
            action = rng.action_by_idx(action_idx)

            gx = _metric_value(action, M_GRID_X)
            gy = _metric_value(action, M_GRID_Y)
            gz = _metric_value(action, M_GRID_Z)
            bx = _metric_value(action, M_BLOCK_X)
            by = _metric_value(action, M_BLOCK_Y)
            bz = _metric_value(action, M_BLOCK_Z)

            yield {
                "launch_id":            launch_id,
                "kernel_name":          action.name(ncu_report.IAction.NameBase_DEMANGLED),
                "mangled_name":         action.name(ncu_report.IAction.NameBase_MANGLED),
                "grid_size":            f"({gx},{gy},{gz})",
                "block_size":           f"({bx},{by},{bz})",
                "cycles":               _metric_value(action, M_CYCLES),
                "global_ld_instr":      _metric_value(action, M_GLOBAL_LD),
                "global_st_instr":      _metric_value(action, M_GLOBAL_ST),
                "shared_ld_instr":      _metric_value(action, M_SHARED_LD),
                "shared_st_instr":      _metric_value(action, M_SHARED_ST),
                "local_ld_instr":       _metric_value(action, M_LOCAL_LD),
                "local_st_instr":       _metric_value(action, M_LOCAL_ST),
                "l1_to_l2_ld_sectors":  _metric_value(action, M_L1_L2_LD),
                "l1_to_l2_st_sectors":  _metric_value(action, M_L1_L2_ST),
                "l2_to_dram_ld_sectors": _metric_value(action, M_L2_DRAM_LD),
                "l2_to_dram_st_sectors": _metric_value(action, M_L2_DRAM_ST),
            }
            launch_id += 1


# ---------------------------------------------------------------------------
# Grouping (Single Report)
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "cycles", "global_ld_instr", "global_st_instr",
    "shared_ld_instr", "shared_st_instr",
    "local_ld_instr", "local_st_instr",
    "l1_to_l2_ld_sectors", "l1_to_l2_st_sectors",
    "l2_to_dram_ld_sectors", "l2_to_dram_st_sectors",
]

GROUP_KEYS = ("kernel_name", "mangled_name", "grid_size", "block_size")

MEM_INSTR_AVG_COLS = [
    "global_ld_instr_avg", "global_st_instr_avg",
    "shared_ld_instr_avg", "shared_st_instr_avg",
    "local_ld_instr_avg",  "local_st_instr_avg",
]

GROUPED_COLUMNS = [
    "kernel_name", "mangled_name", "grid_size", "block_size", "invocations",
    "cycles_avg", "total_mem_instr_avg",
    "global_ld_instr_avg", "global_st_instr_avg",
    "shared_ld_instr_avg", "shared_st_instr_avg",
    "local_ld_instr_avg", "local_st_instr_avg",
    "l1_to_l2_ld_sectors_avg", "l1_to_l2_st_sectors_avg",
    "l2_to_dram_ld_sectors_avg", "l2_to_dram_st_sectors_avg",
]

GROUPED_SECTOR_FIELDS = {
    "l1_to_l2_ld_sectors_avg", "l1_to_l2_st_sectors_avg",
    "l2_to_dram_ld_sectors_avg", "l2_to_dram_st_sectors_avg",
}

def group_by_kernel(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in GROUP_KEYS)
        buckets[key].append(row)

    order = {tuple(r[k] for k in GROUP_KEYS): i for i, r in enumerate(rows)}
    grouped = []
    for key, group in sorted(buckets.items(), key=lambda kv: order[kv[0]]):
        agg: dict = dict(zip(GROUP_KEYS, key))
        agg["invocations"] = len(group)
        for col in NUMERIC_COLS:
            vals = [r[col] for r in group if r.get(col) is not None]
            agg[f"{col}_avg"] = (sum(vals) / len(vals)) if vals else None
        total_vals = [sum(r[c] for c in MEM_INSTR_COLS if r.get(c) is not None) for r in group]
        agg["total_mem_instr_avg"] = sum(total_vals) / len(total_vals) if total_vals else None
        grouped.append(agg)

    return grouped


# ---------------------------------------------------------------------------
# Comparison Logic & Columns
# ---------------------------------------------------------------------------
MEM_INSTR_COLS = [
    "global_ld_instr", "global_st_instr",
    "shared_ld_instr", "shared_st_instr",
    "local_ld_instr",  "local_st_instr",
]
L1_L2_COLS   = ["l1_to_l2_ld_sectors",   "l1_to_l2_st_sectors"]
L2_DRAM_COLS = ["l2_to_dram_ld_sectors"] 

COMPARE_METRIC_COLS = [
    "cycles_speedup",
    "mem_instr_reduce_pct",
    "l1_to_l2_traffic_reduce_pct",
    "l2_to_dram_rd_traffic_reduce_pct",
]

def _reduce_pct(orig_total, opt_total):
    if orig_total is None or opt_total is None or orig_total == 0:
        return None
    return (orig_total - opt_total) / orig_total * 100.0

def _sum_cols(row: dict, cols: list[str]):
    vals = [row.get(c) for c in cols if row.get(c) is not None]
    return sum(vals) if vals else None


# --- Strict Comparison ---
MATCH_KEYS = ("launch_id", "kernel_name", "mangled_name", "grid_size", "block_size")
COMPARE_GROUP_KEYS = ("kernel_name", "mangled_name", "grid_size", "block_size")

COMPARE_COLUMNS = [
    "launch_id", "kernel_name", "mangled_name", "grid_size", "block_size",
    *COMPARE_METRIC_COLS
]
COMPARE_GROUPED_COLUMNS = [
    "kernel_name", "mangled_name", "grid_size", "block_size", "invocations",
    "cycles_speedup_avg", "mem_instr_reduce_pct_avg",
    "l1_to_l2_traffic_reduce_pct_avg", "l2_to_dram_rd_traffic_reduce_pct_avg",
]

def compare_reports(orig_rows: list[dict], opt_rows: list[dict]):
    opt_index = {tuple(r[k] for k in MATCH_KEYS): r for r in opt_rows}
    matched, unmatched_orig = [], []

    for orig in orig_rows:
        key = tuple(orig[k] for k in MATCH_KEYS)
        opt = opt_index.pop(key, None)
        if opt is None:
            unmatched_orig.append(orig)
            continue
        o_cycles, p_cycles = orig.get("cycles"), opt.get("cycles")
        cmp = {k: orig[k] for k in MATCH_KEYS}
        cmp["cycles_speedup"] = o_cycles / p_cycles if o_cycles is not None and p_cycles and p_cycles != 0 else None
        cmp["mem_instr_reduce_pct"] = _reduce_pct(_sum_cols(orig, MEM_INSTR_COLS), _sum_cols(opt,  MEM_INSTR_COLS))
        cmp["l1_to_l2_traffic_reduce_pct"] = _reduce_pct(_sum_cols(orig, L1_L2_COLS), _sum_cols(opt,  L1_L2_COLS))
        cmp["l2_to_dram_rd_traffic_reduce_pct"] = _reduce_pct(_sum_cols(orig, L2_DRAM_COLS), _sum_cols(opt,  L2_DRAM_COLS))
        matched.append(cmp)

    return matched, unmatched_orig, list(opt_index.values())

def group_compare(matched: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in matched: buckets[tuple(row[k] for k in COMPARE_GROUP_KEYS)].append(row)

    order = {tuple(r[k] for k in COMPARE_GROUP_KEYS): i for i, r in enumerate(matched)}
    grouped = []
    for key, group in sorted(buckets.items(), key=lambda kv: order[kv[0]]):
        agg = dict(zip(COMPARE_GROUP_KEYS, key))
        agg["invocations"] = len(group)
        for col in COMPARE_METRIC_COLS:
            vals = [r[col] for r in group if r.get(col) is not None]
            agg[f"{col}_avg"] = (sum(vals) / len(vals)) if vals else None
        grouped.append(agg)
    return grouped


# --- Relaxed Comparison ---
RELAXED_COMPARE_COLUMNS = [
    "orig_launch_id", "opt_launch_id", "kernel_name",
    "orig_grid_size", "orig_block_size", "opt_grid_size", "opt_block_size",
    *COMPARE_METRIC_COLS
]
RELAXED_COMPARE_GROUPED_COLUMNS = [
    "kernel_name", "invocations",
    "cycles_speedup_avg", "mem_instr_reduce_pct_avg",
    "l1_to_l2_traffic_reduce_pct_avg", "l2_to_dram_rd_traffic_reduce_pct_avg",
]

def _base_kernel_name(name: str) -> str:
    for ch in ('<', '('):
        idx = name.find(ch)
        if idx != -1: name = name[:idx]
    name = name.strip()
    parts = name.split()
    return parts[-1] if parts else name

def _make_relaxed_cmp_row(orig: dict, opt: dict) -> dict:
    o_cycles, p_cycles = orig.get("cycles"), opt.get("cycles")
    return {
        "orig_launch_id":  orig["launch_id"],
        "opt_launch_id":   opt["launch_id"],
        "kernel_name":     orig["kernel_name"],
        "orig_grid_size":  orig.get("grid_size"),
        "orig_block_size": orig.get("block_size"),
        "opt_grid_size":   opt.get("grid_size"),
        "opt_block_size":  opt.get("block_size"),
        "cycles_speedup": (o_cycles / p_cycles if o_cycles is not None and p_cycles and p_cycles != 0 else None),
        "mem_instr_reduce_pct": _reduce_pct(_sum_cols(orig, MEM_INSTR_COLS), _sum_cols(opt, MEM_INSTR_COLS)),
        "l1_to_l2_traffic_reduce_pct": _reduce_pct(_sum_cols(orig, L1_L2_COLS), _sum_cols(opt, L1_L2_COLS)),
        "l2_to_dram_rd_traffic_reduce_pct": _reduce_pct(_sum_cols(orig, L2_DRAM_COLS), _sum_cols(opt, L2_DRAM_COLS)),
    }

def compare_reports_relaxed(orig_rows: list[dict], opt_rows: list[dict]):
    from collections import defaultdict
    orig_buckets, opt_buckets = defaultdict(list), defaultdict(list)
    for r in orig_rows: orig_buckets[_base_kernel_name(r["kernel_name"])].append(r)
    for r in opt_rows:  opt_buckets[_base_kernel_name(r["kernel_name"])].append(r)

    matched, unmatched_orig, unmatched_opt = [], [], []
    seen = set()
    for name in list(orig_buckets.keys()):
        seen.add(name)
        o_list, p_list = orig_buckets[name], opt_buckets.get(name, [])
        for orig, opt in zip(o_list, p_list):
            matched.append(_make_relaxed_cmp_row(orig, opt))
        unmatched_orig.extend(o_list[len(p_list):])
        unmatched_opt.extend(p_list[len(o_list):])

    for name, p_list in opt_buckets.items():
        if name not in seen: unmatched_opt.extend(p_list)
    return matched, unmatched_orig, unmatched_opt

def group_compare_relaxed(matched: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in matched: buckets[_base_kernel_name(row["kernel_name"])].append(row)

    order = {_base_kernel_name(r["kernel_name"]): i for i, r in enumerate(matched)}
    grouped = []
    for name, group in sorted(buckets.items(), key=lambda kv: order[kv[0]]):
        agg: dict = {"kernel_name": name, "invocations": len(group)}
        for col in COMPARE_METRIC_COLS:
            vals = [r[col] for r in group if r.get(col) is not None]
            agg[f"{col}_avg"] = (sum(vals) / len(vals)) if vals else None
        grouped.append(agg)
    return grouped


# --- ID-Only Comparison ---
ID_COMPARE_COLUMNS = [
    "launch_id", "orig_kernel_name", "opt_kernel_name",
    "orig_grid_size", "orig_block_size", "opt_grid_size", "opt_block_size",
    *COMPARE_METRIC_COLS
]
ID_COMPARE_GROUPED_COLUMNS = [
    "orig_kernel_name", "opt_kernel_name", "invocations",
    "cycles_speedup_avg", "mem_instr_reduce_pct_avg",
    "l1_to_l2_traffic_reduce_pct_avg", "l2_to_dram_rd_traffic_reduce_pct_avg",
]

def compare_reports_id_only(orig_rows: list[dict], opt_rows: list[dict]):
    """Match exclusively by absolute launch_id, allowing names and sizes to differ."""
    opt_index = {r["launch_id"]: r for r in opt_rows}
    matched, unmatched_orig = [], []

    for orig in orig_rows:
        lid = orig["launch_id"]
        opt = opt_index.pop(lid, None)
        if opt is None:
            unmatched_orig.append(orig)
            continue
        
        o_cycles, p_cycles = orig.get("cycles"), opt.get("cycles")
        cmp = {
            "launch_id": lid,
            "orig_kernel_name": orig["kernel_name"],
            "opt_kernel_name": opt["kernel_name"],
            "orig_grid_size": orig.get("grid_size"),
            "orig_block_size": orig.get("block_size"),
            "opt_grid_size": opt.get("grid_size"),
            "opt_block_size": opt.get("block_size"),
            "cycles_speedup": (o_cycles / p_cycles if o_cycles is not None and p_cycles and p_cycles != 0 else None),
            "mem_instr_reduce_pct": _reduce_pct(_sum_cols(orig, MEM_INSTR_COLS), _sum_cols(opt, MEM_INSTR_COLS)),
            "l1_to_l2_traffic_reduce_pct": _reduce_pct(_sum_cols(orig, L1_L2_COLS), _sum_cols(opt, L1_L2_COLS)),
            "l2_to_dram_rd_traffic_reduce_pct": _reduce_pct(_sum_cols(orig, L2_DRAM_COLS), _sum_cols(opt, L2_DRAM_COLS)),
        }
        matched.append(cmp)

    return matched, unmatched_orig, list(opt_index.values())

def group_compare_id_only(matched: list[dict]) -> list[dict]:
    """Group id-matched rows by the unique pair of (orig_kernel_name, opt_kernel_name)."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in matched:
        key = (row["orig_kernel_name"], row["opt_kernel_name"])
        buckets[key].append(row)

    seen = set()
    order = []
    for r in matched:
        k = (r["orig_kernel_name"], r["opt_kernel_name"])
        if k not in seen:
            seen.add(k)
            order.append(k)

    grouped = []
    for key in order:
        group = buckets[key]
        agg = {
            "orig_kernel_name": key[0],
            "opt_kernel_name": key[1],
            "invocations": len(group)
        }
        for col in COMPARE_METRIC_COLS:
            vals = [r[col] for r in group if r.get(col) is not None]
            agg[f"{col}_avg"] = (sum(vals) / len(vals)) if vals else None
        grouped.append(agg)
    return grouped


# ---------------------------------------------------------------------------
# I/O Writers
# ---------------------------------------------------------------------------
def _fmt(val) -> str:
    if val is None: return ""
    if isinstance(val, float): return f"{val:.2f}"
    return str(val)

def _write_csv_rows(rows: list[dict], cols: list[str], out):
    writer = csv.DictWriter(out, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

def _write_table_rows(rows: list[dict], cols: list[str], out):
    widths = {c: max(len(c), *(len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}
    sep = "  "
    out.write(sep.join(c.ljust(widths[c]) for c in cols) + "\n")
    out.write(sep.join("-" * widths[c] for c in cols) + "\n")
    for row in rows:
        out.write(sep.join(_fmt(row.get(c, "")).ljust(widths[c]) for c in cols) + "\n")

# -- Generic Report Writes --
SECTOR_FIELDS = {"l1_to_l2_ld_sectors", "l1_to_l2_st_sectors", "l2_to_dram_ld_sectors", "l2_to_dram_st_sectors"}
COLUMNS = ["launch_id", "kernel_name", "mangled_name", "grid_size", "block_size", "cycles",
           "global_ld_instr", "global_st_instr", "shared_ld_instr", "shared_st_instr",
           "local_ld_instr",  "local_st_instr", "l1_to_l2_ld_sectors",  "l1_to_l2_st_sectors",
           "l2_to_dram_ld_sectors", "l2_to_dram_st_sectors"]

def _apply_bytes(row: dict, to_bytes: bool, is_grouped: bool = False) -> dict:
    if not to_bytes: return row
    out = dict(row)
    fields = GROUPED_SECTOR_FIELDS if is_grouped else SECTOR_FIELDS
    replace_str = "_sectors_avg" if is_grouped else "_sectors"
    target_str = "_bytes_avg" if is_grouped else "_bytes"
    for f in fields:
        val = out.pop(f, None)
        out[f.replace(replace_str, target_str)] = None if val is None else (val * SECTOR_BYTES if is_grouped else int(val) * SECTOR_BYTES)
    return out

def _rename_cols(columns: list[str], to_bytes: bool, is_grouped: bool = False) -> list[str]:
    if not to_bytes: return columns
    fields = GROUPED_SECTOR_FIELDS if is_grouped else SECTOR_FIELDS
    replace_str = "_sectors_avg" if is_grouped else "_sectors"
    target_str = "_bytes_avg" if is_grouped else "_bytes"
    return [c.replace(replace_str, target_str) if c in fields else c for c in columns]

def write_generic(rows, cols, out, is_csv, to_bytes, is_grouped=False):
    cols = _rename_cols(cols, to_bytes, is_grouped)
    processed = [_apply_bytes(r, to_bytes, is_grouped) for r in rows]
    if is_csv: _write_csv_rows(processed, cols, out)
    else: _write_table_rows(processed, cols, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract performance metrics from an Nsight Compute report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("report", help="Path to origin .ncu-rep report file")
    p.add_argument("--compare", metavar="OPT_REPORT",
                   help="Compare against a second .ncu-rep report.")
    
    match_group = p.add_mutually_exclusive_group()
    match_group.add_argument("--relaxed", action="store_true",
                             help="(compare mode) Match launches by base kernel name only, ignoring mangled name, grid, and block size.")
    match_group.add_argument("--id-only", action="store_true",
                             help="(compare mode) Match strictly by launch_id index. Useful if kernel names/sizes changed but execution order is identical.")
    
    p.add_argument("--csv",   action="store_true", help="Output as CSV")
    p.add_argument("--bytes", action="store_true", help="Convert sector counts to bytes (x32)")
    p.add_argument("--group", action="store_true", help="Group results and average metrics.")
    p.add_argument("-o", "--output", default="-", help="Output file path (default: stdout)")
    return p.parse_args()


def main():
    args = parse_args()

    report_path = Path(args.report)
    if not report_path.exists(): sys.exit(f"ERROR: Report file not found: {report_path}")

    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")

    try:
        if args.compare:
            opt_path = Path(args.compare)
            if not opt_path.exists(): sys.exit(f"ERROR: Optimized report not found: {opt_path}")

            orig_rows = list(extract_actions(report_path))
            opt_rows  = list(extract_actions(opt_path))

            if args.id_only:
                matched, unmatched_orig, unmatched_opt = compare_reports_id_only(orig_rows, opt_rows)
            elif args.relaxed:
                matched, unmatched_orig, unmatched_opt = compare_reports_relaxed(orig_rows, opt_rows)
            else:
                matched, unmatched_orig, unmatched_opt = compare_reports(orig_rows, opt_rows)

            if not matched: sys.exit("No matching launches found between the two reports.")

            if args.id_only:
                if args.group:
                    grp = group_compare_id_only(matched)
                    if args.csv: _write_csv_rows(grp, ID_COMPARE_GROUPED_COLUMNS, out)
                    else: _write_table_rows(grp, ID_COMPARE_GROUPED_COLUMNS, out)
                else:
                    if args.csv: _write_csv_rows(matched, ID_COMPARE_COLUMNS, out)
                    else: _write_table_rows(matched, ID_COMPARE_COLUMNS, out)
                    
            elif args.relaxed:
                if args.group:
                    grp = group_compare_relaxed(matched)
                    if args.csv: _write_csv_rows(grp, RELAXED_COMPARE_GROUPED_COLUMNS, out)
                    else: _write_table_rows(grp, RELAXED_COMPARE_GROUPED_COLUMNS, out)
                else:
                    if args.csv: _write_csv_rows(matched, RELAXED_COMPARE_COLUMNS, out)
                    else: _write_table_rows(matched, RELAXED_COMPARE_COLUMNS, out)
                    
            else:
                if args.group:
                    grp = group_compare(matched)
                    if args.csv: _write_csv_rows(grp, COMPARE_GROUPED_COLUMNS, out)
                    else: _write_table_rows(grp, COMPARE_GROUPED_COLUMNS, out)
                else:
                    if args.csv: _write_csv_rows(matched, COMPARE_COLUMNS, out)
                    else: _write_table_rows(matched, COMPARE_COLUMNS, out)

            print(f"Matched: {len(matched)}  |  Only in origin: {len(unmatched_orig)}  |  Only in optimized: {len(unmatched_opt)}", file=sys.stderr)

        else:
            rows = list(extract_actions(report_path))
            if not rows: sys.exit("No kernel actions found in the report.")

            if args.group:
                grouped = group_by_kernel(rows)
                write_generic(grouped, GROUPED_COLUMNS, out, args.csv, args.bytes, is_grouped=True)
            else:
                write_generic(rows, COLUMNS, out, args.csv, args.bytes, is_grouped=False)

            if args.output != "-": print(f"Written {len(rows)} kernel(s) to {args.output}", file=sys.stderr)

    finally:
        if out is not sys.stdout: out.close()


if __name__ == "__main__":
    main()