#!/usr/bin/env python3
"""
Parse and cluster AccelProf-generated block_distribution/kernel_*.csv.

CSV layout (as produced by sanalyzer/src/tools/block_divergence_analysis.cpp):
  blockidx,blockidy,blockidz,<pc_0>,<pc_1>,...,<pc_n>,read_count,write_count

The number of <pc_i> columns is variable. Each PC column name looks like
0x0000000000000000 (16-hex-digit, zero-padded).
Each row corresponds to one block. Each PC column value is the executed count
for that memory instruction in that block (accumulated by popcount(active_mask)).

Clustering options:
  - exact: group rows with exactly identical feature vectors (excluding block id)
  - threshold: group by cosine-similarity graph connectivity (no preset k)
  - kmeans: spherical k-means with cosine similarity (requires --k)

All methods naturally support sparse features and variable PC column counts.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


FeatureVec = Dict[str, float]  # sparse vector: key is PC column name (or special feature), value is count


@dataclass(frozen=True)
class BlockRow:
    src_file: str
    blockidx: int
    blockidy: int
    blockidz: int
    read_count: int
    write_count: int
    features: FeatureVec  # already includes PC features; read/write inclusion is configurable


def _l2_norm(vec: FeatureVec) -> float:
    s = 0.0
    for v in vec.values():
        s += v * v
    return math.sqrt(s)


def _normalize(vec: FeatureVec) -> FeatureVec:
    n = _l2_norm(vec)
    if n == 0.0:
        return {}
    inv = 1.0 / n
    return {k: v * inv for k, v in vec.items() if v != 0.0}


def _dot(a: FeatureVec, b: FeatureVec) -> float:
    # Sparse dot-product: iterate over the smaller dict
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, av in a.items():
        bv = b.get(k)
        if bv is not None:
            s += av * bv
    return s


def read_block_distribution_csv(
    csv_path: Path,
    *,
    include_read_write: bool = True,
    read_key: str = "__read_count__",
    write_key: str = "__write_count__",
) -> List[BlockRow]:
    """
    Read one kernel_*.csv and return sparse features for each block.

    - PC columns: header[3:-2]
    - read_count/write_count: last two columns
    """
    rows: List[BlockRow] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return rows

        header = [h.strip() for h in header if h is not None]
        if len(header) < 5:
            raise ValueError(f"Unexpected CSV header column count ({len(header)}): {csv_path}")
        if header[0:3] != ["blockidx", "blockidy", "blockidz"]:
            raise ValueError(f"CSV header first 3 columns mismatch: {header[0:3]} in {csv_path}")
        if header[-2:] != ["read_count", "write_count"]:
            raise ValueError(f"CSV header last 2 columns mismatch: {header[-2:]} in {csv_path}")

        pc_cols = header[3:-2]

        for r in reader:
            if not r:
                continue
            if len(r) == 1 and r[0].strip() == "":
                continue
            if len(r) != len(header):
                # Tolerate: some tools may append an extra empty trailing column
                r = [x.strip() for x in r]
                if r and r[-1] == "" and len(r) == len(header) + 1:
                    r = r[:-1]
                if len(r) != len(header):
                    raise ValueError(
                        f"Row column count mismatch: header={len(header)}, row={len(r)}; file={csv_path}"
                    )

            blockidx = int(r[0])
            blockidy = int(r[1])
            blockidz = int(r[2])
            read_count = int(r[-2])
            write_count = int(r[-1])

            feat: FeatureVec = {}
            # PC features
            for pc_key, v in zip(pc_cols, r[3:-2]):
                iv = int(v)
                if iv != 0:
                    feat[pc_key] = float(iv)
            # Optional: include aggregated read/write counts to help distinguish RW patterns
            if include_read_write:
                if read_count != 0:
                    feat[read_key] = float(read_count)
                if write_count != 0:
                    feat[write_key] = float(write_count)

            rows.append(
                BlockRow(
                    src_file=str(csv_path),
                    blockidx=blockidx,
                    blockidy=blockidy,
                    blockidz=blockidz,
                    read_count=read_count,
                    write_count=write_count,
                    features=feat,
                )
            )

    return rows


@dataclass
class ClusterResult:
    assignments: List[int]              # cluster id for each sample
    similarities: List[float]           # cosine similarity (dot product) to the assigned centroid
    centroids: List[FeatureVec]         # L2-normalized centroids


def _canonical_int_signature(vec: FeatureVec) -> Tuple[Tuple[str, int], ...]:
    """
    Convert a "count-like" feature vector into a hashable signature for exact clustering.
    Note: in this pipeline values are integer counts (stored as float), so we cast to int().
    """
    items: List[Tuple[str, int]] = []
    for k, v in vec.items():
        iv = int(v)
        if iv != 0:
            items.append((k, iv))
    items.sort(key=lambda kv: kv[0])
    return tuple(items)


def cluster_exact(vectors: Sequence[FeatureVec]) -> ClusterResult:
    """
    Automatic clustering: group samples whose row feature vectors (excluding block id) are exactly identical.
    - No need to specify k
    - Great for checking whether blocks have any (strict) behavioral differences
    """
    if not vectors:
        raise ValueError("Empty vector set")

    normed = [_normalize(v) for v in vectors]

    sig2cid: Dict[Tuple[Tuple[str, int], ...], int] = {}
    assignments: List[int] = [-1] * len(vectors)
    for i, v in enumerate(vectors):
        sig = _canonical_int_signature(v)
        cid = sig2cid.get(sig)
        if cid is None:
            cid = len(sig2cid)
            sig2cid[sig] = cid
        assignments[i] = cid

    k = len(sig2cid)
    # centroid: average normalized vectors, then renormalize (spherical centroid)
    sums: List[FeatureVec] = [dict() for _ in range(k)]
    counts = [0] * k
    for v, a in zip(normed, assignments):
        counts[a] += 1
        s = sums[a]
        for key, val in v.items():
            s[key] = s.get(key, 0.0) + val
    centroids: List[FeatureVec] = []
    for j in range(k):
        avg = {key: val / counts[j] for key, val in sums[j].items()} if counts[j] else {}
        centroids.append(_normalize(avg))

    similarities = [_dot(v, centroids[a]) if centroids[a] else 0.0 for v, a in zip(normed, assignments)]
    return ClusterResult(assignments=assignments, similarities=similarities, centroids=centroids)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def cluster_threshold_cosine(
    vectors: Sequence[FeatureVec],
    *,
    sim_threshold: float = 0.999,
) -> ClusterResult:
    """
    Automatic clustering (no k): connect samples with cosine similarity >= threshold, then
    use connected components as clusters.
    - More robust to "nearly identical" block behavior (allows small count differences)
    - O(n^2); suitable for n up to a few thousand
    """
    if not vectors:
        raise ValueError("Empty vector set")
    if not (0.0 <= sim_threshold <= 1.0):
        raise ValueError("sim_threshold must be in [0, 1]")

    normed = [_normalize(v) for v in vectors]
    n = len(normed)
    uf = _UnionFind(n)
    for i in range(n):
        vi = normed[i]
        for j in range(i + 1, n):
            if _dot(vi, normed[j]) >= sim_threshold:
                uf.union(i, j)

    root2cid: Dict[int, int] = {}
    assignments: List[int] = [-1] * n
    for i in range(n):
        r = uf.find(i)
        cid = root2cid.get(r)
        if cid is None:
            cid = len(root2cid)
            root2cid[r] = cid
        assignments[i] = cid

    k = len(root2cid)
    sums: List[FeatureVec] = [dict() for _ in range(k)]
    counts = [0] * k
    for v, a in zip(normed, assignments):
        counts[a] += 1
        s = sums[a]
        for key, val in v.items():
            s[key] = s.get(key, 0.0) + val
    centroids: List[FeatureVec] = []
    for j in range(k):
        avg = {key: val / counts[j] for key, val in sums[j].items()} if counts[j] else {}
        centroids.append(_normalize(avg))

    similarities = [_dot(v, centroids[a]) if centroids[a] else 0.0 for v, a in zip(normed, assignments)]
    return ClusterResult(assignments=assignments, similarities=similarities, centroids=centroids)


def spherical_kmeans(
    vectors: Sequence[FeatureVec],
    k: int,
    *,
    max_iter: int = 50,
    seed: int = 0,
) -> ClusterResult:
    """
    Sparse spherical k-means:
      - Input vectors are L2-normalized
      - Similarity uses dot-product (equivalent to cosine)
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if not vectors:
        raise ValueError("Empty vector set")

    rnd = random.Random(seed)
    normed = [_normalize(v) for v in vectors]

    # Initialize centroids by sampling k distinct examples (best effort)
    idxs = list(range(len(normed)))
    rnd.shuffle(idxs)
    chosen = idxs[: min(k, len(idxs))]
    centroids: List[FeatureVec] = [dict(normed[i]) for i in chosen]
    while len(centroids) < k:
        centroids.append(dict(normed[rnd.randrange(len(normed))]))

    assignments = [-1] * len(normed)
    similarities = [0.0] * len(normed)

    for _it in range(max_iter):
        changed = 0

        # Assign
        for i, v in enumerate(normed):
            best_j = 0
            best_sim = -1.0
            for j, c in enumerate(centroids):
                sim = _dot(v, c)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if assignments[i] != best_j:
                changed += 1
                assignments[i] = best_j
            similarities[i] = best_sim

        if changed == 0:
            break

        # Update centroids: sum within cluster then renormalize
        sums: List[FeatureVec] = [dict() for _ in range(k)]
        counts = [0] * k
        for v, a in zip(normed, assignments):
            counts[a] += 1
            s = sums[a]
            for key, val in v.items():
                s[key] = s.get(key, 0.0) + val

        for j in range(k):
            if counts[j] == 0:
                # Empty cluster: reset to a random sample
                centroids[j] = dict(normed[rnd.randrange(len(normed))])
                continue
            # Average then renormalize (spherical)
            avg = {key: val / counts[j] for key, val in sums[j].items()}
            centroids[j] = _normalize(avg)

    return ClusterResult(assignments=assignments, similarities=similarities, centroids=centroids)


def centroid_top_features(centroid: FeatureVec, top_n: int) -> List[Tuple[str, float]]:
    items = sorted(centroid.items(), key=lambda kv: kv[1], reverse=True)
    return items[:top_n]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse AccelProf block_distribution/kernel_*.csv and cluster blocks (variable PC columns supported)."
    )
    ap.add_argument("csv_files", nargs="+", help="One or more kernel_*.csv paths")
    ap.add_argument(
        "--method",
        type=str,
        default="exact",
        choices=["exact", "threshold", "kmeans"],
        help="Clustering method: exact(default, group identical rows), threshold(cosine threshold, no preset k), kmeans(requires --k)",
    )
    ap.add_argument("--k", type=int, default=None, help="Number of clusters (required only for method=kmeans)")
    ap.add_argument(
        "--sim-threshold",
        type=float,
        default=0.999,
        help="For method=threshold: cosine similarity threshold (default 0.999)",
    )
    ap.add_argument("--max-iter", type=int, default=50, help="Max iterations (default 50)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    ap.add_argument(
        "--no-read-write",
        action="store_true",
        help="Do not include read_count/write_count as clustering features (PC dimensions only)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="clusters.csv",
        help="Output result CSV path (default clusters.csv)",
    )
    ap.add_argument(
        "--details",
        action="store_true",
        help="Print detailed per-cluster information (centroid top features and block(0,0,0) membership).",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=8,
        help="When --details is enabled: print top-N centroid features per cluster (default 8)",
    )
    args = ap.parse_args(argv)

    all_rows: List[BlockRow] = []
    for p in args.csv_files:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(str(path))
        all_rows.extend(
            read_block_distribution_csv(
                path,
                include_read_write=not args.no_read_write,
            )
        )

    if not all_rows:
        print("No block rows were read.")
        return 0

    vectors = [r.features for r in all_rows]
    if args.method == "kmeans":
        if args.k is None:
            raise ValueError("method=kmeans requires an explicit --k")
        res = spherical_kmeans(vectors, args.k, max_iter=args.max_iter, seed=args.seed)
    elif args.method == "threshold":
        res = cluster_threshold_cosine(vectors, sim_threshold=args.sim_threshold)
    else:
        res = cluster_exact(vectors)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "src_file",
                "blockidx",
                "blockidy",
                "blockidz",
                "cluster_id",
                "cosine_similarity",
                "read_count",
                "write_count",
            ]
        )
        for r, cid, sim in zip(all_rows, res.assignments, res.similarities):
            w.writerow([r.src_file, r.blockidx, r.blockidy, r.blockidz, cid, f"{sim:.6f}", r.read_count, r.write_count])

    # Summary
    k_out = len(res.centroids)
    sizes = [0] * k_out
    for a in res.assignments:
        sizes[a] += 1

    total_n = len(all_rows)
    # Minimal summary output by default (as requested): cluster_id, size, ratio
    print(f"Total samples: {total_n}; method={args.method}; cluster_count={k_out}; out={out_path}")
    for j in range(k_out):
        ratio = (sizes[j] / total_n) if total_n else 0.0
        print(f"[cluster {j}] size={sizes[j]} ratio={ratio:.2%}")

    # Always keep: which cluster does block (0,0,0) belong to? (print once per input file if multiple)
    block0_hits = []
    for r, cid, sim in zip(all_rows, res.assignments, res.similarities):
        if r.blockidx == 0 and r.blockidy == 0 and r.blockidz == 0:
            block0_hits.append((r.src_file, cid, sim, r.read_count, r.write_count))
    if block0_hits:
        print("\nblock(0,0,0) cluster belongs to cluster:")
        for src, cid, sim, rc, wc in block0_hits:
            ratio = (sizes[cid] / total_n) if total_n else 0.0
            print(
                f"  file={src}  cluster_id={cid}  cluster_ratio={ratio:.2%}  "
                f"cosine_similarity={sim:.6f}  read_count={rc}  write_count={wc}"
            )
    else:
        print("\nblock(0,0,0) was not found in the input data.")

    if args.details:
        if args.method == "kmeans":
            empty = [i for i, s in enumerate(sizes) if s == 0]
            if empty:
                print(f"\nNote: empty clusters {empty} (usually means k is too large; consider reducing --k).")

        for j in range(k_out):
            print(f"\n[cluster {j}] centroid top-{args.top}:")
            for kf, kv in centroid_top_features(res.centroids[j], args.top):
                print(f"  {kf}: {kv:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


