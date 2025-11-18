import argparse, shutil, subprocess, sys
from pathlib import Path

def run_and_capture(cmd, cwd=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def detect_supported_flags(jar_path, java_heap):
    cmd = ["java", f"-Xmx{java_heap}", "-jar", str(jar_path), "-help"]
    rc, out, err = run_and_capture(cmd)
    help_text = (out or "") + ("\n" + err if err else "")
    have = {
        "mins": "-mins" in help_text,
        "minhc": "-minhc" in help_text,
        "minpca": "-minpca" in help_text,
        "maxad": "-maxad" in help_text,
        "maxadc": "-maxadc" in help_text,
        "const": "-const" in help_text,
        "oute": "-oute" in help_text,
        "datalog": "-datalog" in help_text,
    }
    return have, help_text

def build_cmd(jar, java_heap, train, args, supported):
    cmd = ["java", f"-Xmx{java_heap}", "-jar", str(jar), train]

    # Optional parser mode (-datalog) if supported/asked for
    if args.datalog and supported.get("datalog"):
        cmd.append("-datalog")

    # Thresholds
    if supported.get("mins"):   cmd += ["-mins", str(args.mins)]
    if supported.get("minhc"):  cmd += ["-minhc", str(args.minhc)]
    if supported.get("minpca"): cmd += ["-minpca", str(args.minpca)]

    # Search space (fully CLI-controlled)
    if args.maxad is not None and supported.get("maxad"):
        cmd += ["-maxad", str(args.maxad)]
    if args.const and supported.get("const"):
        cmd.append("-const")
        if args.maxadc is not None and supported.get("maxadc"):
            cmd += ["-maxadc", str(args.maxadc)]

    if args.oute and supported.get("oute"):
        cmd.append("-oute")

    return cmd

def _normalize_token(tok: str, prefix="x"):
    """AMIE prefers identifiers starting with a letter. Prefix if not."""
    if not tok:
        return tok
    c0 = tok[0]
    if not (("A" <= c0 <= "Z") or ("a" <= c0 <= "z")):
        return prefix + tok
    return tok

def _extract_triples(lines_iter, safe_prefix=True, prefix="x"):
    for ln in lines_iter:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()  # split on any whitespace
        if len(parts) < 3:
            continue
        h, r, t = parts[0], parts[1], parts[2]
        if safe_prefix:
            h = _normalize_token(h, prefix)
            r = _normalize_token(r, prefix)
            t = _normalize_token(t, prefix)
        yield (h, r, t)

def build_unified_amie_input(run_dir: Path, safe_prefix=True, sep="\t") -> Path:
    """
    Read run_{k}/train.txt, valid.txt, test.txt (our synthetic 6-col TSV is fine),
    write deduped (h{sep}r{sep}t) to run_{k}/_amie/all.tsv.
    """
    srcs = [run_dir/"train.txt", run_dir/"valid.txt", run_dir/"test.txt"]
    missing = [p.name for p in srcs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {run_dir}: {missing}")

    out_dir = run_dir / "_amie"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all.tsv"

    seen = set()
    with out_path.open("w") as fout:
        for src in srcs:
            with src.open("r") as fin:
                for h, r, t in _extract_triples(fin, safe_prefix=safe_prefix, prefix="x"):
                    key = (h, r, t)
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(f"{h}{sep}{r}{sep}{t}\n")
    print(f"[amie] unified triples -> {out_path} ({len(seen)} unique)")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_roots", nargs="+", help="*_prepared roots with run_*")
    ap.add_argument("--jar", default="./AMIE-3.5.1/amie3.5.1.jar")
    ap.add_argument("--java-heap", default="8G")

    # Mining thresholds
    ap.add_argument("--mins", type=int, default=10)
    ap.add_argument("--minhc", type=float, default=0.1)
    ap.add_argument("--minpca", type=float, default=0.1)

    # Search-space (CLI-controlled)
    ap.add_argument("--maxad", type=int, default=3, help="max atoms in rule/body")
    ap.add_argument("--const", action="store_true", help="enable constants in rules")
    ap.add_argument("--maxadc", type=int, default=1, help="max constants per rule (needs --const)")

    # Parser mode
    ap.add_argument("--datalog", action="store_true", help="force -datalog when supported")

    # Output / control
    ap.add_argument("--oute", action="store_true")
    ap.add_argument("--only-runs", nargs="*", default=None)
    ap.add_argument("--log-help", action="store_true")
    ap.add_argument("--clobber", action="store_true", help="delete amie/ outdir before running")

    # Input building options
    ap.add_argument("--no-safe-prefix", dest="safe_prefix", action="store_false",
                    help="do not prefix numeric/non-letter identifiers")
    ap.set_defaults(safe_prefix=True)
    ap.add_argument("--sep", choices=["tab", "space"], default="tab",
                    help="separator for AMIE input file")

    args = ap.parse_args()

    jar = Path(args.jar).resolve()
    if not jar.exists():
        sys.exit(f"AMIE jar not found: {jar}")

    supported, help_text = detect_supported_flags(jar, args.java_heap)
    print("[amie] Detected support:", supported)
    if args.log_help:
        (Path(".") / "amie_help.txt").write_text(help_text)

    sep_char = "\t" if args.sep == "tab" else " "

    for root in args.prepared_roots:
        root = Path(root).resolve()
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        if args.only_runs:
            runs = [p for p in runs if p.name in set(args.only_runs)]
        if not runs:
            print(f"[amie] no runs in {root}")
            continue

        for run_dir in runs:
            # 1) Build unified (train+valid+test) input for AMIE
            try:
                unified_path = build_unified_amie_input(run_dir, safe_prefix=args.safe_prefix, sep=sep_char)
            except Exception as e:
                print(f"[amie] SKIP {run_dir.name}: {e}")
                continue

            # 2) Prepare output area
            out_dir = run_dir / "amie"
            if args.clobber and out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_rules  = out_dir / "amie_rules.tsv"
            out_stdout = out_dir / "stdout.txt"
            out_stderr = out_dir / "stderr.txt"

            # 3) Build and run AMIE command
            cmd = build_cmd(jar, args.java_heap, unified_path.as_posix(), args, supported)
            print("[amie]", " ".join(cmd))

            rc, out, err = run_and_capture(cmd)
            # Overwrite on every run
            out_stdout.write_text(out or "")
            out_stderr.write_text(err or "")

            if rc != 0:
                print(f"[amie] ERROR (exit={rc}). See:\n  {out_stdout}\n  {out_stderr}")
                if err:
                    e_lines = err.strip().splitlines()
                    print("[amie:stderr head]", "\n".join(e_lines[:15]))
                    if len(e_lines) > 15:
                        print("... (see file for full stderr)")
                sys.exit(1)
            else:
                # Mirror stdout (typical rule table) to canonical file
                out_rules.write_text(out or "")
                print(f"[amie] OK -> {out_rules}")

if __name__ == "__main__":
    main()

'''
Example:
python run_amie_all.py data_static_small_20251028_prepared data_static_sequential_small_20251028_prepared \
  --java-heap 8G --mins 5 --minhc 0.05 --minpca 0.05 --maxad 3 --const --maxadc 1 \
  --datalog --oute --log-help --clobber --sep tab
'''
