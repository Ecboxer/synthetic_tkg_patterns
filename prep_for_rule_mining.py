import argparse, os
from pathlib import Path
import pandas as pd

def read_edge_file(path: Path):
    cols = ["head","rel","tail","t","wt","pattern"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str, engine="python")
    return df[["head","rel","tail"]].copy()

def convert_run(run_dir: Path, out_dir: Path, safe_prefix: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train","valid","test"]:
        p = run_dir / f"{split}.txt"
        if not p.exists(): 
            continue
        df = read_edge_file(p)
        if safe_prefix:
            df["head"] = "e" + df["head"].astype(str)
            df["rel"]  = "r" + df["rel"].astype(str)
            df["tail"] = "e" + df["tail"].astype(str)
        df.to_csv(out_dir / f"{split}.txt", sep=" ", header=False, index=False)

    # copy through pattern2id.txt if present (for later analysis)
    p2 = run_dir / "pattern2id.txt"
    if p2.exists():
        (out_dir / "pattern2id.txt").write_bytes(p2.read_bytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dirs", nargs="+", help="Export dirs containing run_* folders")
    ap.add_argument("--out-root", default=None, help="Root output (default: <export_dir>_prepared)")
    ap.add_argument("--safe-prefix", action="store_true", help="Prefix entities/relations with letters")
    args = ap.parse_args()

    for ed in args.export_dirs:
        ed = Path(ed).resolve()
        out_root = Path(args.out_root or (str(ed) + "_prepared")).resolve()
        runs = sorted([p for p in ed.glob("run_*") if p.is_dir()])
        if not runs:
            print(f"[prep] No runs in {ed}")
            continue
        for r in runs:
            convert_run(r, out_root / r.name, args.safe_prefix)
            print(f"[prep] Wrote {out_root / r.name}")
        # convenience symlink
        link = out_root / "dataset"
        target = out_root / "run_0"
        try:
            if link.exists() or link.is_symlink(): link.unlink()
            if target.exists(): link.symlink_to(target, target_is_directory=True)
        except Exception as e:
            print(f"[prep] (optional) symlink failed: {e}")

if __name__ == "__main__":
    main()

'''
python prep_for_rule_mining.py \
  data_static_small_20251028 data_static_medium_20251028 data_static_large_20251028 \
  data_static_sequential_small_20251028 data_static_sequential_medium_20251028 data_static_sequential_large_20251028 \
  --safe-prefix
'''