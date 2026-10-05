"""Collect validation-grid JSON files into a ranked CSV and report missing cells. Rerun the
winner for final testing."""

import argparse
import csv
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", required=True, help="Directory of per-cell *.json files.")
    ap.add_argument("--expected", type=int, default=None,
                    help="Expected cell count; warns if fewer are present.")
    ap.add_argument("--out-csv", default=None, help="Defaults to <grid-dir>/grid_summary.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.grid_dir, "*.json")))
    paths = [p for p in paths if not p.endswith("grid_summary.json")]
    if not paths:
        raise SystemExit(f"ERROR: no cell JSONs in {args.grid_dir}. "
                         "Has any array element finished? Check its job log.")

    rows = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        row = {"cell_id": d.get("cell_id", os.path.basename(p)[:-5]),
               **{k: v for k, v in (d.get("cell") or {}).items()},
               "val_macro": d.get("val_macro"),
               "best_epoch": d.get("best_epoch")}
        rows.append(row)

    rows.sort(key=lambda r: (r["val_macro"] is None, r["val_macro"]))
    cols = list(rows[0].keys())
    for r in rows:
        for c in cols:
            r.setdefault(c, "")

    w = {c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols}
    print(f"\n==================== grid: {args.grid_dir} "
          f"({len(rows)} cell(s), best first) ====================")
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        vals = [f"{r[c]:.4f}" if c == "val_macro" and isinstance(r[c], float) else f"{r[c]}"
                for c in cols]
        print("  " + "  ".join(v.ljust(w[c]) for v, c in zip(vals, cols)))

    if args.expected is not None and len(rows) < args.expected:
        print(f"\nWARNING: {len(rows)}/{args.expected} cells present -- "
              f"{args.expected - len(rows)} still running or failed. The ranking "
              "above is provisional.")

    best = rows[0]
    print(f"\nbest val_macro = {best['val_macro']:.4f}  ({best['cell_id']})")
    print("  -> a CONFIGURATION CHOICE, not a result. Re-run it through the "
          "multi-seed test evaluation to get a number worth reporting.")

    out_csv = args.out_csv or os.path.join(args.grid_dir, "grid_summary.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
