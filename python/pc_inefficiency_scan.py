#!/usr/bin/env python3
"""
Scan a directory of pc_dependency_analysis JSON outputs for all known
inefficiency patterns (mirrors the detection logic in kernel_level_traffic_v2.html).

Detected patterns
-----------------
  ATOMIC_HIGH    Atomic node: > 50% inter-block accesses  → severe serialization
  ATOMIC_MEDIUM  Atomic node: > 15% inter-block or > 50% inter-warp
  ATOMIC_LOW     Atomic node: >  3% inter-block or > 20% inter-warp
  RAR            Purely intra-thread Read-After-Read (redundant load)
  RAW            Purely intra-thread Read-After-Write (redundant load)
  WAW            Purely intra-thread Write-After-Write (redundant store)
  GRID_REUSE     > 60% intra-grid reuse  (inter-block data reuse opportunity)
  BLOCK_REUSE    > 60% intra-(thread+warp+block) reuse on global memory

Usage
-----
  python pc_inefficiency_scan.py <directory> [options]

Options
-------
  --min-severity LEVEL   Only report at or above: LOW | MEDIUM | HIGH (default: LOW)
                         Severity mapping:
                           HIGH   → ATOMIC_HIGH
                           MEDIUM → ATOMIC_MEDIUM, WAW
                           LOW    → ATOMIC_LOW, RAR, RAW, GRID_REUSE, BLOCK_REUSE
  --min-weight N         Skip nodes with fewer than N total accesses (default: 100)
  --min-traffic SIZE     Only report kernels whose global cold-miss traffic
                         (read + write) is at least SIZE, e.g. 1MB, 500KB, 20MB
                         (default: 0, i.e. show all)
  --json                 Machine-readable JSON output
  -v / --verbose         Show per-node details for every flagged file
  --no-color             Disable ANSI color in text output
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Pattern definitions and severity ordering
# ---------------------------------------------------------------------------
PATTERN_SEVERITY: Dict[str, str] = {
    "ATOMIC_HIGH":   "HIGH",
    "WAW":           "MEDIUM",
    "ATOMIC_MEDIUM": "MEDIUM",
    "RAR":           "LOW",
    "RAW":           "LOW",
    "ATOMIC_LOW":    "LOW",
    "GRID_REUSE":    "LOW",
    "BLOCK_REUSE":   "LOW",
}

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

PATTERN_LABEL = {
    "ATOMIC_HIGH":   "Atomic Contention (HIGH)",
    "ATOMIC_MEDIUM": "Atomic Contention (MEDIUM)",
    "ATOMIC_LOW":    "Atomic Contention (LOW)",
    "RAR":           "Redundant Read-After-Read",
    "RAW":           "Redundant Read-After-Write",
    "WAW":           "Redundant Write-After-Write",
    "GRID_REUSE":    "High Intra-Grid Reuse",
    "BLOCK_REUSE":   "High Intra-Block/Warp/Thread Reuse",
}

# Detection thresholds (same values as HTML tool)
ATOMIC_HIGH_CROSS_PCT   = 50.0
ATOMIC_MEDIUM_CROSS_PCT = 15.0
ATOMIC_MEDIUM_WARP_PCT  = 50.0
ATOMIC_LOW_CROSS_PCT    =  3.0
ATOMIC_LOW_WARP_PCT     = 20.0
GRID_REUSE_PCT          = 60.0
BLOCK_REUSE_PCT         = 60.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    pattern:   str          # one of PATTERN_SEVERITY keys
    severity:  str          # "LOW" | "MEDIUM" | "HIGH"
    pc_hex:    str
    flags:     str
    total_accesses: int
    metrics:   Dict[str, float]   # pattern-specific percentages / values
    detail:    str


@dataclass
class ColdMissTraffic:
    global_read:  float = 0.0
    global_write: float = 0.0
    shared_read:  float = 0.0
    shared_write: float = 0.0

    @property
    def total(self) -> float:
        return self.global_read + self.global_write + self.shared_read + self.shared_write


@dataclass
class FileReport:
    path:         Path
    kernel_name:  str
    kernel_id:    Optional[int]
    device_id:    Optional[int]
    cold_miss:    ColdMissTraffic = field(default_factory=ColdMissTraffic)
    findings:     List[Finding]   = field(default_factory=list)

    @property
    def max_severity(self) -> str:
        if not self.findings:
            return "NONE"
        return max((f.severity for f in self.findings), key=lambda s: SEVERITY_ORDER[s])

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_file(path: Path, min_weight: int) -> FileReport:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    kernel      = data.get("kernel") or {}
    kernel_name = kernel.get("kernel_name", path.stem)

    node_map: Dict[int, dict] = {n["pc"]: n for n in (data.get("nodes") or [])}
    edges = data.get("edges") or []

    # Group edges by current_pc
    edges_by_pc: Dict[int, List[dict]] = {}
    for e in edges:
        pc = e["current_pc"]
        edges_by_pc.setdefault(pc, []).append(e)

    # Flag map for ancient node lookup
    flag_map: Dict[int, str] = {pc: (n.get("flags") or "") for pc, n in node_map.items()}

    # ── Cold-miss traffic (mirrors calculateGlobalStats in the HTML tool) ──
    cold_miss = ColdMissTraffic()
    for e in edges:
        if not e.get("cold_miss"):
            continue
        count = sum((e.get("dist") or {}).values())
        size  = e.get("current_access_size") or 4
        # normalise to sector-equivalent bytes (same formula as the HTML tool)
        traffic = size * count if size < 4 else (size * count) / (-((-size) // 4))
        flags = flag_map.get(e["current_pc"], "")
        if "SHARED" in flags:
            if "WRITE" in flags: cold_miss.shared_write += traffic
            else:                cold_miss.shared_read  += traffic
        else:
            if "WRITE" in flags: cold_miss.global_write += traffic
            else:                cold_miss.global_read  += traffic

    report = FileReport(
        path=path,
        kernel_name=kernel_name,
        kernel_id=kernel.get("kernel_id"),
        device_id=kernel.get("device_id"),
        cold_miss=cold_miss,
    )

    for pc, node in node_map.items():
        deps = edges_by_pc.get(pc, [])
        if not deps:
            continue

        flags     = node.get("flags") or ""
        is_atomic = "ATOMIC" in flags
        is_shared = "SHARED" in flags
        is_read   = "READ"   in flags
        is_write  = "WRITE"  in flags
        pc_hex    = node.get("pc_hex") or hex(pc)

        total_weight = 0
        sum_thread = sum_warp = sum_block = sum_grid = 0
        all_intra_thread  = True
        all_ancient_write = True
        all_ancient_read  = True

        for e in deps:
            dist  = e.get("dist") or {}
            w     = sum(dist.values())
            total_weight += w

            if not e.get("cold_miss"):
                sum_thread += dist.get("intra_thread", 0)
                sum_warp   += dist.get("intra_warp",   0)
                sum_block  += dist.get("intra_block",  0)
                sum_grid   += dist.get("intra_grid",   0)

            # For redundancy: must be ONLY intra_thread (no cold, no cross-warp)
            only_thread = (
                not e.get("cold_miss")
                and dist.get("intra_thread", 0) > 0
                and not dist.get("intra_warp",  0)
                and not dist.get("intra_block", 0)
                and not dist.get("intra_grid",  0)
            )
            if not only_thread:
                all_intra_thread = False

            anc_flags = flag_map.get(e.get("ancient_pc"), "")
            if not anc_flags or "WRITE" not in anc_flags:
                all_ancient_write = False
            if not anc_flags or "READ"  not in anc_flags:
                all_ancient_read  = False

        if total_weight < min_weight:
            continue

        cross_pct = (sum_block + sum_grid) / total_weight * 100
        warp_pct  =  sum_warp              / total_weight * 100
        grid_pct  =  sum_grid              / total_weight * 100
        local_pct = (sum_thread + sum_warp + sum_block) / total_weight * 100

        findings = report.findings

        # ── Atomic contention ──────────────────────────────────────────────
        if is_atomic:
            if cross_pct > ATOMIC_HIGH_CROSS_PCT:
                findings.append(Finding(
                    pattern="ATOMIC_HIGH", severity="HIGH",
                    pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                    metrics={"inter_block_pct": cross_pct, "inter_warp_pct": warp_pct},
                    detail=f"{cross_pct:.1f}% accesses from other blocks — severe "
                           "serialization risk. Consider block-level reduction first.",
                ))
            elif cross_pct > ATOMIC_MEDIUM_CROSS_PCT or warp_pct > ATOMIC_MEDIUM_WARP_PCT:
                findings.append(Finding(
                    pattern="ATOMIC_MEDIUM", severity="MEDIUM",
                    pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                    metrics={"inter_block_pct": cross_pct, "inter_warp_pct": warp_pct},
                    detail=f"{cross_pct:.1f}% inter-block, {warp_pct:.1f}% inter-warp — "
                           "partial privatization may help.",
                ))
            elif cross_pct > ATOMIC_LOW_CROSS_PCT or warp_pct > ATOMIC_LOW_WARP_PCT:
                findings.append(Finding(
                    pattern="ATOMIC_LOW", severity="LOW",
                    pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                    metrics={"inter_block_pct": cross_pct, "inter_warp_pct": warp_pct},
                    detail=f"Low atomic contention ({cross_pct:.1f}% inter-block).",
                ))

        # ── Redundancy patterns (purely intra-thread) ──────────────────────
        if all_intra_thread:
            if   is_read  and all_ancient_read:
                findings.append(Finding(
                    pattern="RAR", severity="LOW",
                    pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                    metrics={"intra_thread_pct": 100.0},
                    detail="Purely intra-thread Read-After-Read — redundant load.",
                ))
            elif is_read  and all_ancient_write:
                findings.append(Finding(
                    pattern="RAW", severity="LOW",
                    pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                    metrics={"intra_thread_pct": 100.0},
                    detail="Purely intra-thread Read-After-Write — may be eliminable "
                           "via register forwarding.",
                ))
            elif is_write and all_ancient_write:
                findings.append(Finding(
                    pattern="WAW", severity="MEDIUM",
                    pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                    metrics={"intra_thread_pct": 100.0},
                    detail="Purely intra-thread Write-After-Write — earlier write is "
                           "dead; consider removing redundant store.",
                ))

        # ── High intra-grid reuse ──────────────────────────────────────────
        if grid_pct > GRID_REUSE_PCT:
            findings.append(Finding(
                pattern="GRID_REUSE", severity="LOW",
                pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                metrics={"grid_reuse_pct": grid_pct},
                detail=f"{grid_pct:.1f}% reuse across blocks — data is shared "
                       "between CTAs; shared/L2 caching may help.",
            ))

        # ── High intra-block reuse (global memory only) ────────────────────
        if local_pct > BLOCK_REUSE_PCT and not is_shared:
            findings.append(Finding(
                pattern="BLOCK_REUSE", severity="LOW",
                pc_hex=pc_hex, flags=flags, total_accesses=total_weight,
                metrics={"local_reuse_pct": local_pct},
                detail=f"{local_pct:.1f}% reuse within thread/warp/block — "
                       "consider register reuse or shared memory buffering.",
            ))

    # Sort: severity desc → inter_block desc (or first metric)
    report.findings.sort(key=lambda f: (
        -SEVERITY_ORDER[f.severity],
        -next(iter(f.metrics.values()), 0),
    ))
    return report


# ---------------------------------------------------------------------------
# Size parsing helper
# ---------------------------------------------------------------------------
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

def parse_size(s: str) -> float:
    """Parse a human-readable size string like '20MB' or '500KB' into bytes."""
    s = s.strip()
    for suffix, mult in sorted(_SIZE_UNITS.items(), key=lambda x: -len(x[0])):
        if s.upper().endswith(suffix):
            return float(s[: -len(suffix)]) * mult
    return float(s)  # bare number = bytes


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
SEVERITY_COLOR = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[94m"}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def _c(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{RESET}" if use_color else text


def _sev(severity: str, use_color: bool) -> str:
    label = f"[{severity:<6}]"
    return _c(label, SEVERITY_COLOR.get(severity, "") + BOLD, use_color)


def print_text_report(reports: List[FileReport], min_severity: str,
                      verbose: bool, use_color: bool,
                      min_traffic: float = 0.0) -> None:
    from collections import Counter, defaultdict

    min_level = SEVERITY_ORDER[min_severity]
    flagged = [
        r for r in reports
        if r.has_findings
        and SEVERITY_ORDER.get(r.max_severity, -1) >= min_level
        and (r.cold_miss.global_read + r.cold_miss.global_write) >= min_traffic
    ]

    total = len(reports)
    header = f"PC Inefficiency Scan — {total} file(s) analyzed"
    print(f"\n{'='*70}")
    print(_c(header, BOLD, use_color))
    print(f"{'='*70}")

    if not flagged:
        print(f"  No findings with severity >= {min_severity}.\n")
        return

    # Group by max severity for the summary line
    by_sev: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in flagged:
        by_sev[r.max_severity] += 1
    summary_parts = [f"{v} {k}" for k, v in by_sev.items() if v]
    print(f"  Flagged: {len(flagged)} / {total}  ({', '.join(summary_parts)})\n")

    # Group reports by kernel name
    groups: Dict[str, List[FileReport]] = defaultdict(list)
    for r in flagged:
        groups[r.kernel_name].append(r)

    # Sort groups by worst severity then total global cold-miss traffic (desc)
    def _group_key(kv):
        name, grp = kv
        max_sev = max(SEVERITY_ORDER[r.max_severity] for r in grp)
        total_traffic = sum(r.cold_miss.global_read + r.cold_miss.global_write for r in grp)
        return (-max_sev, -total_traffic)

    for kernel_name, grp in sorted(groups.items(), key=_group_key):
        # Aggregate cold-miss traffic across all invocations
        agg_cm = ColdMissTraffic(
            global_read=sum(r.cold_miss.global_read  for r in grp),
            global_write=sum(r.cold_miss.global_write for r in grp),
            shared_read=sum(r.cold_miss.shared_read  for r in grp),
            shared_write=sum(r.cold_miss.shared_write for r in grp),
        )
        all_visible = [
            f for r in grp
            for f in r.findings
            if SEVERITY_ORDER[f.severity] >= min_level
        ]
        grp_max_sev = max((r.max_severity for r in grp), key=lambda s: SEVERITY_ORDER[s])

        print(_c(f"  Kernel: {kernel_name}", BOLD, use_color))
        print(f"    Invocations : {len(grp)}  |  Max severity: "
              + _c(grp_max_sev, SEVERITY_COLOR.get(grp_max_sev, "") + BOLD, use_color))
        print(f"    Cold Miss Traffic (aggregated):")
        print(f"      Global  read  {_fmt_bytes(agg_cm.global_read):<12}  write {_fmt_bytes(agg_cm.global_write)}")
        print(f"      Shared  read  {_fmt_bytes(agg_cm.shared_read):<12}  write {_fmt_bytes(agg_cm.shared_write)}")
        print(f"      Total   {_fmt_bytes(agg_cm.total)}")
        print(f"    Findings: {len(all_visible)}")

        counts = Counter(f.pattern for f in all_visible)
        pattern_summary = "  ".join(
            f"{PATTERN_LABEL[p]} ×{n}" for p, n in sorted(
                counts.items(), key=lambda kv: -SEVERITY_ORDER[PATTERN_SEVERITY[kv[0]]]
            )
        )
        print(f"    {_c(pattern_summary, DIM, use_color)}")

        # Per-invocation breakdown
        for r in grp:
            visible = [f for f in r.findings if SEVERITY_ORDER[f.severity] >= min_level]
            if not visible:
                continue
            cm = r.cold_miss
            inv_label = r.path.name
            if r.kernel_id is not None:
                inv_label += f"  (id={r.kernel_id}, dev={r.device_id})"
            print(f"\n    [{_c(inv_label, DIM, use_color)}]")
            print(f"      Global cold-miss  read {_fmt_bytes(cm.global_read)}  "
                  f"write {_fmt_bytes(cm.global_write)}")

            if verbose:
                for finding in visible:
                    sev_str = _sev(finding.severity, use_color)
                    label   = PATTERN_LABEL[finding.pattern]
                    metrics = "  ".join(f"{k}: {v:.1f}%" for k, v in finding.metrics.items())
                    print(f"\n        {sev_str}  {label}")
                    print(f"          PC      : {finding.pc_hex}  [{finding.flags}]")
                    print(f"          Accesses: {finding.total_accesses:,}")
                    print(f"          Metrics : {metrics}")
                    print(f"          Note    : {finding.detail}")
        print()

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------
def print_json_report(reports: List[FileReport], min_severity: str,
                      min_traffic: float = 0.0) -> None:
    from collections import defaultdict

    min_level = SEVERITY_ORDER[min_severity]

    # Group by kernel name
    groups: Dict[str, List[FileReport]] = defaultdict(list)
    for r in reports:
        if (r.cold_miss.global_read + r.cold_miss.global_write) < min_traffic:
            continue
        visible = [f for f in r.findings if SEVERITY_ORDER[f.severity] >= min_level]
        if not visible:
            groups[r.kernel_name]  # ensure key exists even if empty after filter
            groups[r.kernel_name].append(r)
        else:
            groups[r.kernel_name].append(r)

    out = []
    for kernel_name, grp in sorted(groups.items()):
        invocations = []
        for r in grp:
            visible = [f for f in r.findings if SEVERITY_ORDER[f.severity] >= min_level]
            if not visible:
                continue
            cm = r.cold_miss
            invocations.append({
                "file":      str(r.path),
                "kernel_id": r.kernel_id,
                "device_id": r.device_id,
                "max_severity": r.max_severity,
                "cold_miss_traffic": {
                    "global_read_bytes":  round(cm.global_read),
                    "global_write_bytes": round(cm.global_write),
                    "shared_read_bytes":  round(cm.shared_read),
                    "shared_write_bytes": round(cm.shared_write),
                    "total_bytes":        round(cm.total),
                },
                "findings": [
                    {
                        "pattern":        f.pattern,
                        "severity":       f.severity,
                        "label":          PATTERN_LABEL[f.pattern],
                        "pc_hex":         f.pc_hex,
                        "flags":          f.flags,
                        "total_accesses": f.total_accesses,
                        "metrics":        {k: round(v, 2) for k, v in f.metrics.items()},
                        "detail":         f.detail,
                    }
                    for f in visible
                ],
            })
        if not invocations:
            continue

        agg_gr  = sum(r.cold_miss.global_read  for r in grp)
        agg_gw  = sum(r.cold_miss.global_write for r in grp)
        agg_sr  = sum(r.cold_miss.shared_read  for r in grp)
        agg_sw  = sum(r.cold_miss.shared_write for r in grp)
        grp_max = max((r.max_severity for r in grp), key=lambda s: SEVERITY_ORDER[s])
        out.append({
            "kernel_name": kernel_name,
            "invocation_count": len(invocations),
            "max_severity": grp_max,
            "aggregated_cold_miss_traffic": {
                "global_read_bytes":  round(agg_gr),
                "global_write_bytes": round(agg_gw),
                "shared_read_bytes":  round(agg_sr),
                "shared_write_bytes": round(agg_sw),
                "total_bytes":        round(agg_gr + agg_gw + agg_sr + agg_sw),
            },
            "invocations": invocations,
        })
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan pc_dependency_analysis JSON outputs for all inefficiency patterns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("directory", type=Path,
                        help="Directory containing pc_dependency_analysis JSON files.")
    parser.add_argument("--min-severity", choices=["LOW", "MEDIUM", "HIGH"], default="LOW",
                        help="Only report findings at or above this severity (default: LOW).")
    parser.add_argument("--min-weight", type=int, default=100, metavar="N",
                        help="Ignore nodes with fewer than N total accesses (default: 100).")
    parser.add_argument("--min-traffic", default="0", metavar="SIZE",
                        help="Only report kernels with global cold-miss traffic (read+write) "
                             ">= SIZE, e.g. 20MB, 500KB (default: 0).")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON output.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-node details for every flagged file.")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color codes.")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory.", file=sys.stderr)
        return 2

    json_files = sorted(args.directory.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in '{args.directory}'.", file=sys.stderr)
        return 0

    reports: List[FileReport] = []
    for path in json_files:
        try:
            reports.append(analyze_file(path, args.min_weight))
        except Exception as exc:
            print(f"  Warning — skipped {path.name}: {exc}", file=sys.stderr)

    try:
        min_traffic = parse_size(args.min_traffic)
    except ValueError:
        print(f"Error: invalid --min-traffic value '{args.min_traffic}'.", file=sys.stderr)
        return 2

    use_color = not args.no_color and sys.stdout.isatty()

    if args.json:
        print_json_report(reports, args.min_severity, min_traffic)
    else:
        print_text_report(reports, args.min_severity, args.verbose, use_color, min_traffic)

    # Exit 1 if any HIGH severity finding present (useful in CI)
    has_high = any(
        r.max_severity == "HIGH" for r in reports
        if r.has_findings
        and (r.cold_miss.global_read + r.cold_miss.global_write) >= min_traffic
    )
    return 1 if has_high else 0


if __name__ == "__main__":
    sys.exit(main())
