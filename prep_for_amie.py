#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def convert_one(src_path: Path, dst_path: Path):
    # expects prepared train/valid/test with cols: head rel tail (first 3 cols); ignores extras
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with src_path.open("r") as fh:
        for line in fh:
            if not line.strip(): continue
            parts = line.strip().split()  # works for tabs or spaces
            if len(parts) < 3: continue
            h, r, t = parts[0], parts[1], parts[2]
            # Ensure tokens begin with a letter (AMIE-friendly)
            def norm(x, pfx):
                # if already alpha-start, keep; else prefix (strip leading underscores just in case)
                x = x.strip()
                if not x: return pfx+"0"
                if x[0].isalpha(): return x
                return f"{pfx}{x}"
            h = norm(h, "e")
            r = norm(r, "r")
            t = norm(t, "e")
            out.append(f"{h}\t{r}\t{t}\n")
    with dst_path.open("w") as g:
        g.writelines(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_roots", nargs="+", help="*_prepared roots with run_*")
    ap.add_argument("--out-suffix", default="_amie", help="subdir name to write cleaned triples")
    args = ap.parse_args()

    for root in args.prepared_roots:
        root = Path(root).resolve()
        for run_dir in sorted([p for p in root.glob("run_*") if p.is_dir()]):
            src_train = run_dir/"train.txt"
            if not src_train.exists():
                print(f"[amie-prep] missing {src_train}, skipping")
                continue
            out_dir = run_dir/args.out_suffix
            out_dir.mkdir(exist_ok=True)
            convert_one(src_train, out_dir/"train.tsv")
            # Optional: also convert valid/test if you plan to use them
            if (run_dir/"valid.txt").exists():
                convert_one(run_dir/"valid.txt", out_dir/"valid.tsv")
            if (run_dir/"test.txt").exists():
                convert_one(run_dir/"test.txt", out_dir/"test.tsv")
            print(f"[amie-prep] wrote {out_dir/'train.tsv'}")

if __name__ == "__main__":
    main()


'''
python prep_for_amie.py data_static_*_20251028_prepared data_static_sequential_*_20251028_prepared
'''
