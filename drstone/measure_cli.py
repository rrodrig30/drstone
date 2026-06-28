"""Measure stone HU from a DICOM series and print JSON (stdout = JSON only).

Run in an isolated subprocess from the web layer so TotalSegmentator/nnUNet
worker spawning never touches the async server process:
    python -m drstone.measure_cli "<dicom_series_dir>"
"""

from __future__ import annotations

import contextlib
import json
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    real_stdout = sys.stdout
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    list_all = "--all" in sys.argv
    if not args:
        print(json.dumps({"error": "no path"}), file=real_stdout)
        return
    path = args[0]
    try:
        # Send all segmentation/TS noise to stderr; keep stdout for JSON only.
        with contextlib.redirect_stdout(sys.stderr):
            if list_all:
                from drstone.stone_segmentation import list_stones
                res = {"stones": list_stones(path)}
            else:
                from drstone.stone_segmentation import segment_stone
                res = segment_stone(path)
    except Exception as e:
        res = {"stones": [], "found": False, "error": str(e)}
    print(json.dumps(res), file=real_stdout)


if __name__ == "__main__":
    main()
