"""Multi-stone matching + MIL labels (Dr Stone, lift N).

Each patient may have several analyzed stones (COMP: mass + composition) and
several segmented stones (rich features). We match them per patient by rank:
largest segmented volume <-> largest analyzed mass (mass ~= volume * density),
which is robust to absolute-density differences across compositions.

Match quality is recorded so modeling can weight/filter:
  rank_mass      : #seg == #comp and all masses present (clean stone-level label)
  rank_partial   : counts/masses incomplete; top-min matched
  patient_level  : no usable mass -> assign the patient's dominant composition
                   (MIL-style; weaker label)

Output: drstone_data/drstone_matched_stones.csv (stone-level features + label +
match_quality + patient bag id), lifting the labeled set from 131 -> ~all stones.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.data_ingest import build_crossover_map, load_composition

warnings.filterwarnings("ignore")

LABEL_COLS = ["mass_mg", "dominant_parent", "ua_any", "mixed", "n_components",
              "modeling_class"]


def main():
    xls = pd.ExcelFile(C.WORKBOOK, engine="openpyxl")
    xmap = build_crossover_map(xls)
    comp = load_composition(xls, xmap)
    pcols = [c for c in comp.columns if c.startswith("p_")]
    seg = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_all_stones.csv"),
                      dtype={"canonical_mrn": str})
    label_cols = LABEL_COLS + pcols

    rows = []
    for mrn, segs in seg.groupby("canonical_mrn"):
        cs = comp[comp["canonical_mrn"] == mrn]
        if cs.empty:
            continue
        cs = cs.sort_values("mass_mg", ascending=False, na_position="last").reset_index(drop=True)
        # Keep only the top-N_comp largest segmented objects (the real stones);
        # the rest are calcific flecks/phleboliths and are dropped, not labeled.
        segs = (segs.sort_values("volume_mm3", ascending=False)
                    .head(len(cs)).reset_index(drop=True))
        has_mass = cs["mass_mg"].notna()
        n = min(len(segs), len(cs))         # == len(segs) now
        for i in range(n):
            srow = segs.iloc[i].to_dict()
            srow["patient_bag"] = mrn
            srow["n_seg"] = len(segs); srow["n_comp"] = len(cs)
            crow = cs.iloc[i]
            if len(cs) == 1:
                q = "single_comp"
            elif has_mass.iloc[:n].all() and len(segs) == len(cs):
                q = "rank_mass"            # equal counts, all masses present
            elif has_mass.iloc[i]:
                q = "rank_partial"
            else:
                q = "patient_level"        # this stone's mass missing -> weaker
            for c in label_cols:
                srow[c] = crow[c]
            srow["match_quality"] = q
            rows.append(srow)

    out = pd.DataFrame(rows)
    out["y_ua"] = (out["dominant_parent"] == "UA").astype(int)
    p = os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv")
    out.to_csv(p, index=False)
    print(f"matched stones: {len(out)} from {out['patient_bag'].nunique()} patients")
    print("match quality:", dict(out["match_quality"].value_counts()))
    print("clean (rank_mass/single_comp):",
          int(out["match_quality"].isin(["rank_mass", "single_comp"]).sum()))
    print("dominant composition:", dict(out["dominant_parent"].value_counts()))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
