"""Re-extract the stone density-gradient (core/periphery HU ratio) for the
labeled cohort and join it onto the existing matched-stones table by stone_idx.

Why a dedicated runner: TotalSegmentator spawns multiprocessing workers that
re-import the entry module, so this MUST run as a real module via `python -m`
(never `-c`/stdin). Output is written incrementally and is resumable.

Run:  python -m drstone.gradient_extract [--limit N]
Writes: data/drstone_gradient.csv  (one row per (canonical_mrn, stone_idx))
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd

from drstone import config as C
from drstone.stone_features import extract_all_stones

OUT = os.path.join(C.OUTPUT_DIR, "drstone_gradient.csv")


def cohort_patients() -> pd.DataFrame:
    """Curated patients that contribute a labeled stone to the model cohort."""
    link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"),
                       dtype={"canonical_mrn": str})
    ms = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv"),
                     dtype={"canonical_mrn": str})
    coh = set(ms[ms.match_quality.isin(["rank_mass", "single_comp"])].canonical_mrn)
    cur = (link[link.curated_dir.notna() & link.canonical_mrn.isin(coh)]
           .drop_duplicates("canonical_mrn").reset_index(drop=True))
    return cur


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    warnings.filterwarnings("ignore")
    limit = int(_arg("--limit")) if "--limit" in sys.argv else None
    nshards = int(_arg("--nshards", 1))
    shard = int(_arg("--shard", 0))
    suffix = _arg("--suffix", "")           # distinct output file for a tail helper
    reverse = "--reverse" in sys.argv        # iterate the shard list back-to-front
    out_path = OUT if nshards == 1 else OUT.replace(".csv", f".part{shard}{suffix}.csv")

    pts = cohort_patients()
    if nshards > 1:
        pts = pts[pts.index % nshards == shard].reset_index(drop=True)
    if reverse:
        pts = pts.iloc[::-1].reset_index(drop=True)
    if limit:
        pts = pts.head(limit)

    import glob as _glob

    def load_done():
        """All patients already written to ANY part file for this shard, so a
        forward + reverse pair (and the original run) skip each other's work."""
        d = set()
        for src in _glob.glob(OUT.replace(".csv", f".part{shard}*.csv")) + [out_path]:
            if os.path.exists(src):
                try:
                    d |= set(pd.read_csv(src, dtype={"canonical_mrn": str}).canonical_mrn)
                except Exception:
                    pass
        return d

    done = load_done()
    if done:
        print(f"resuming: {len(done)} patients already extracted", flush=True)
    print(f"cohort: {len(pts)} patients ({len(pts) - len(done & set(pts.canonical_mrn))} to do)",
          flush=True)
    t0 = time.time()
    for i, r in pts.iterrows():
        mrn = r["canonical_mrn"]
        if mrn in load_done():          # refresh each iter: skip what the sibling did
            continue
        t = time.time()
        try:
            stones = extract_all_stones(r["curated_dir"], light=True)
        except Exception as e:
            print(f"[{i + 1}/{len(pts)}] {mrn} FAILED: {e}", flush=True)
            stones = []
        rows = [{
            "canonical_mrn": mrn, "stone_idx": j,
            "hu_core_p50": s.get("hu_core_p50"), "hu_rim_p50": s.get("hu_rim_p50"),
            "hu_core_over_rim": s.get("hu_core_over_rim"),
            "grad_measurable": s.get("grad_measurable"),
            "volume_mm3_recheck": s.get("volume_mm3"),
        } for j, s in enumerate(stones)]
        if rows:
            pd.DataFrame(rows).to_csv(out_path, mode="a",
                                      header=not os.path.exists(out_path), index=False)
        el = time.time() - t0
        print(f"[{i + 1}/{len(pts)}] {mrn}: {len(stones)} stones in {time.time() - t:.0f}s "
              f"(elapsed {el/60:.1f}m)", flush=True)
    print(f"done -> {out_path}  ({(time.time() - t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
