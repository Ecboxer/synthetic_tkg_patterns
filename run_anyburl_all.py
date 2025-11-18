import argparse, shutil, subprocess, sys, traceback
from pathlib import Path

LEARN_TEMPLATE = """\
PATH_TRAINING = {train}
PATH_OUTPUT   = {out_prefix}
SAFE_PREFIX_MODE = {safe_prefix}
SNAPSHOTS_AT = {snapshots}
WORKER_THREADS = {threads}
TIME = {time_sec}
MAX_LENGTH_CYCLIC = {len_cyclic}
MAX_LENGTH_ACYCLIC = {len_acyclic}
MAX_LENGTH_GROUNDED_CYCLIC = {len_grounded_cyclic}
THRESHOLD_CORRECT_PREDICTIONS = {thresh_corr}
THRESHOLD_CONFIDENCE = {thresh_conf}
"""

def _extract_triples(lines_iter):
    for ln in lines_iter:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:  # tolerate 6-col input by taking first 3
            continue
        yield parts[0], parts[1], parts[2]

def build_unified_input(run_dir: Path) -> Path:
    srcs = [run_dir/"train.txt", run_dir/"valid.txt", run_dir/"test.txt"]
    missing = [p.name for p in srcs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {run_dir}: {missing}")

    out_dir = run_dir / "_anyburl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all.txt"

    seen = set()
    with out_path.open("w") as fout:
        for src in srcs:
            with src.open("r") as fin:
                for h, r, t in _extract_triples(fin):
                    key = (h, r, t)
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(f"{h} {r} {t}\n")
    if not seen:
        raise RuntimeError(f"No triples extracted for {run_dir} (check inputs).")
    print(f"[anyburl] unified triples -> {out_path} ({len(seen)} unique)")
    return out_path

def run_java(cmd, stdout_path: Path, stderr_path: Path, cwd=None, timeout=None):
    print("[anyburl]", " ".join(cmd))
    with stdout_path.open("w") as out_f, stderr_path.open("w") as err_f:
        try:
            res = subprocess.run(cmd, cwd=cwd, stdout=out_f, stderr=err_f, timeout=timeout)
            return res.returncode
        except subprocess.TimeoutExpired:
            err_f.write("\n[TIMEOUT]\n")
            return -9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_roots", nargs="+")
    ap.add_argument("--jar", default="./AnyBURL/AnyBURL-23-1.jar")
    ap.add_argument("--java-heap", default="12G")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time-sec", type=int, default=300)
    ap.add_argument("--snapshots", default="10,50,100,200")
    ap.add_argument("--only-runs", nargs="*", default=None)
    ap.add_argument("--clobber", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=None)

    # identifiers & rule lengths
    ap.add_argument("--safe-prefix", action="store_true", default=True)
    ap.add_argument("--len-cyclic", type=int, default=3)
    ap.add_argument("--len-acyclic", type=int, default=3)
    ap.add_argument("--len-grounded-cyclic", type=int, default=3)

    # thresholds
    ap.add_argument("--thresh-correct", type=int, default=2)
    ap.add_argument("--thresh-conf", type=float, default=1e-4)

    args = ap.parse_args()

    jar = Path(args.jar).resolve()
    if not jar.exists():
        print(f"AnyBURL jar not found: {jar}", file=sys.stderr)
        sys.exit(2)

    failures, successes = [], []

    for root in args.prepared_roots:
        root = Path(root).resolve()
        runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
        if args.only_runs:
            runs = [p for p in runs if p.name in set(args.only_runs)]
        if not runs:
            print(f"[anyburl] no runs in {root}")
            continue

        for run_dir in runs:
            try:
                unified = build_unified_input(run_dir)

                out_dir   = run_dir / "anyburl"
                rules_dir = out_dir / "rules"
                logs_dir  = out_dir / "logs"
                if args.clobber and out_dir.exists():
                    shutil.rmtree(out_dir)
                rules_dir.mkdir(parents=True, exist_ok=True)
                logs_dir.mkdir(parents=True, exist_ok=True)

                # IMPORTANT: file *prefix*, not directory
                out_prefix = (rules_dir / "rules").as_posix()

                cfg = out_dir / "config-learn.properties"
                cfg.write_text(LEARN_TEMPLATE.format(
                    train=unified.as_posix(),
                    out_prefix=out_prefix,
                    snapshots=args.snapshots,
                    threads=args.threads,
                    time_sec=args.time_sec,
                    safe_prefix=str(args.safe_prefix).lower(),
                    len_cyclic=args.len_cyclic,
                    len_acyclic=args.len_acyclic,
                    len_grounded_cyclic=args.len_grounded_cyclic,
                    thresh_corr=args.thresh_correct,
                    thresh_conf=args.thresh_conf,
                ))

                stdout_log = logs_dir / "stdout.txt"
                stderr_log = logs_dir / "stderr.txt"
                cmd = ["java", f"-Xmx{args.java_heap}", "-cp", str(jar),
                       "de.unima.ki.anyburl.Learn", str(cfg)]

                rc = run_java(cmd, stdout_log, stderr_log, timeout=args.timeout_sec)

                # Did we produce any rule snapshots?
                produced = list(rules_dir.glob("rules-*"))
                ok = (rc == 0) or any(p.stat().st_size > 0 for p in produced)

                (out_dir / ("SUCCESS.txt" if ok else "FAILED.txt")).write_text(
                    f"exit_code={rc}\nproduced_snapshots={[p.name for p in produced]}\n"
                )

                if ok:
                    successes.append((root.name, run_dir.name))
                else:
                    failures.append((root.name, run_dir.name, "no rules produced"))

            except Exception as e:
                (run_dir / "anyburl").mkdir(parents=True, exist_ok=True)
                (run_dir / "anyburl" / "FAILED.txt").write_text(
                    f"exception: {e}\n{traceback.format_exc()}"
                )
                failures.append((root.name, run_dir.name, f"exception: {e}"))

    print("\n=== AnyBURL summary ===")
    print(f"  OK: {len(successes)}")
    for rroot, rrun in successes:
        print(f"    - {rroot}/{rrun}")
    print(f"  FAIL: {len(failures)}")
    for rroot, rrun, why in failures:
        print(f"    - {rroot}/{rrun}  ({why})")

    # Always exit 0 so your shell pipeline completes all batches.
    sys.exit(0)

if __name__ == "__main__":
    main()
