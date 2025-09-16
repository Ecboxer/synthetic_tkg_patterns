import argparse
import json
import os
import shutil
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

# Bayesian optimizer
try:
    import optuna
    HAS_OPTUNA = True
    print('Using Optuna')
except Exception:
    # Require Optuna
    HAS_OPTUNA = False
    raise EnvironmentError('Could not import Optuna')

# Import your generator
# NOTE: This imports `config.py` at module import time (because run.py imports it),
# which is fine as long as config.py exists. We will NOT use `configs` here; we call run(config, run_id).
from run import run as generate_one


# Metric keys used everywhere
_METRIC_KEYS = [
    "n_ents",
    "n_rels",
    "n_tws",
    "n_quads",
    "n_unique_triples",
    "avg_degree",
    "deg_p25",
    "deg_p75",
]

# Handle empty edgelist-producing hyperparameter settings
PENALTY_SCORE = 1e9

def _empty_metrics():
    return {
        "n_ents": 0, "n_rels": 0, "n_tws": 0, "n_quads": 0,
        "n_unique_triples": 0, "avg_degree": 0.0, "deg_p25": 0.0, "deg_p75": 0.0,
    }

# Data loading & metric helpers
def _read_edgelist_any(path: str) -> pd.DataFrame:
    """
    Robust loader for both reference TKG and your synthetic files.
    - Assumes tab-separated with >= 5 columns possible.
    - Returns standardized columns: head, rel, tail, t (ints).
    - Ignores extra columns (pattern/weights/whatever).
    """
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, engine="python")
    if df.shape[1] < 4:
        raise ValueError(f"File {path} must have at least 4 tab-separated columns.")

    # Keep first 4 columns: head, rel, tail, t
    df = df.iloc[:, :4].copy()
    df.columns = ["head", "rel", "tail", "t"]

    # Coerce to int
    for c in ["head", "rel", "tail", "t"]:
        df[c] = pd.to_numeric(df[c], errors="raise", downcast="integer")

    return df


def load_tkg_dir(edgelist_dir: str) -> pd.DataFrame:
    """
    Merge train/valid/test edgelists found in a directory.
    Files must be named: train.txt, valid.txt, test.txt
    """
    parts = []
    for name in ["train.txt", "valid.txt", "test.txt"]:
        p = os.path.join(edgelist_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
        parts.append(_read_edgelist_any(p))
    df = pd.concat(parts, ignore_index=True)
    # Normalize types
    for c in ["head", "rel", "tail", "t"]:
        df[c] = df[c].astype(np.int64)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Metrics:
    - n_ents: #unique entities appearing as head/tail across all rows
    - n_rels: #unique relations
    - n_tws:  #unique timestamps
    - n_quads: #rows
    - n_unique_triples: #unique (h,r,t) ignoring time
    - avg_degree: average entity degree (count appearances in head or tail across all rows)
    - deg_p25, deg_p75: 25th/75th percentile of entity total degrees
    """
    # Unique entities (head ∪ tail)
    ents = pd.unique(pd.concat([df["head"], df["tail"]], axis=0))
    n_ents = len(ents)
    n_rels = df["rel"].nunique()
    n_tws = df["t"].nunique()
    n_quads = int(df.shape[0])
    n_unique_triples = df[["head", "rel", "tail"]].drop_duplicates().shape[0]

    # Degree per entity: count appearances as head or tail
    # (Counts rows, not weights; matches reference TKG convention where last col is -1.)
    deg = pd.Series(0, index=pd.Index(ents, name="entity"), dtype=np.int64)
    # head contributions
    head_counts = df["head"].value_counts()
    deg = deg.add(head_counts, fill_value=0)
    # tail contributions
    tail_counts = df["tail"].value_counts()
    deg = deg.add(tail_counts, fill_value=0)
    # fill NaNs (if any) then cast to int
    deg = deg.fillna(0).astype(np.int64)

    avg_degree = float(deg.mean()) if n_ents > 0 else 0.0
    deg_p25 = float(np.percentile(deg.values, 25)) if n_ents > 0 else 0.0
    deg_p75 = float(np.percentile(deg.values, 75)) if n_ents > 0 else 0.0

    return {
        "n_ents": int(n_ents),
        "n_rels": int(n_rels),
        "n_tws": int(n_tws),
        "n_quads": int(n_quads),
        "n_unique_triples": int(n_unique_triples),
        "avg_degree": avg_degree,
        "deg_p25": deg_p25,
        "deg_p75": deg_p75,
    }


def normalized_error(syn: float, real: float, denom_floor: float = 1.0) -> float:
    """Relative absolute error with small floor for stability."""
    denom = max(abs(real), denom_floor)
    return abs(syn - real) / denom


def score_metrics(syn: dict, real: dict, weights: dict = None) -> float:
    """
    Weighted sum of normalized errors across the specified metrics.
    """
    if weights is None:
        weights = {
            "n_ents": 1.0,
            "n_rels": 1.0,
            "n_tws": 1.0,
            "n_quads": 1.0,
            "n_unique_triples": 1.0,
            "avg_degree": 1.0,
            "deg_p25": 1.0,
            "deg_p75": 1.0,
        }
    total = 0.0
    for k, w in weights.items():
        total += w * normalized_error(syn[k], real[k])
    return float(total)


def _rng_from_seed(seed):
    import numpy as np
    # Accept either an int/None or an existing RandomState
    return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)


# Config proposal helpers
def binom_force_lambda(n: int, p: float):
    """
    Build a config-compatible sampler: lambda seed: scipy.stats.binom.rvs(n, p, size=1, random_state=seed)[0]
    We define it here to avoid repeating lambdas in the trial loop.
    """
    import scipy.stats as st
    return lambda seed: st.binom.rvs(n=n, p=p, size=1, random_state=seed)[0]


def make_config_from_trial(
    base_config: dict,
    trial_params: dict,
    export_dir: str,
    trial_id: int
) -> dict:
    """
    Create a runnable config by cloning a base template and applying trial parameters.
    Ensures run_dir uniqueness and seeds.
    """
    cfg = deepcopy(base_config)

    # Core quantities
    cfg['n_ents'] = int(trial_params['n_ents'])
    cfg['n_rels'] = int(trial_params['n_rels'])
    cfg['n_tws']  = int(trial_params['n_tws'])

    # Pattern counts
    cfg['n_1_hop'] = int(trial_params['n_1_hop'])
    cfg['n_2_hop'] = int(trial_params['n_2_hop'])
    cfg['n_3_hop'] = int(trial_params['n_3_hop'])

    # Booleans
    cfg['require_unique_triples'] = bool(trial_params['require_unique_triples'])
    cfg['prohibit_selfconnections'] = bool(trial_params['prohibit_selfconnections'])
    cfg['prohibit_new_consequence_relations'] = bool(trial_params['prohibit_new_consequence_relations'])
    cfg['require_sequential_rule'] = bool(trial_params['require_sequential_rule'])

    # Pass resolved samplers (or None for uniform)
    cfg['pat_distr_ents'] = trial_params.get('_pat_distr_ents_fn', None)
    cfg['pat_distr_rels'] = trial_params.get('_pat_distr_rels_fn', None)

    # Random wiring: either use a fixed density or a Poisson sampler for integer edges per entity
    if trial_params.get('use_density_dist', False):
        lam = float(trial_params['rnd_lambda'])
        import scipy.stats as st
        cfg['rnd_avg_density_distr'] = (lambda lam:
            (lambda seed=None: int(st.poisson(mu=lam).rvs(1, random_state=seed)[0]))
        )(lam)
        cfg['rnd_avg_density'] = 0.0   # ignored when *_distr is set
    else:
        cfg['rnd_avg_density'] = float(trial_params['rnd_avg_density'])
        cfg['rnd_avg_density_distr'] = None

    # Skip-consequence prob (applies when a pattern would fire)
    cfg['p_skip_consequence'] = float(trial_params['p_skip_consequence'])

    # Forcing counts:
    # If *_distr is selected, we override p_force & n_force with a distribution sampler.
    use_force_distr = bool(trial_params.get('use_force_distr', True))
    if use_force_distr:
        n = int(trial_params['force_binom_n'])
        p = float(trial_params['force_binom_p'])
        sampler = binom_force_lambda(n=n, p=p)
        cfg['n_hops2n_force_distr'] = {1: sampler, 2: sampler, 3: sampler}
        # n_hops2p_force/n_hops2n_force are ignored when *_distr is present
        cfg['n_hops2p_force'] = {1: 0.0, 2: 0.0, 3: 0.0}
        cfg['n_hops2n_force'] = {1: 0, 2: 0, 3: 0}
    else:
        # Deterministic attempts and per-attempt probability
        cfg['n_hops2p_force'] = {
            1: float(trial_params['p_force']),
            2: float(trial_params['p_force']),
            3: float(trial_params['p_force']),
        }
        nf = int(trial_params['n_force'])
        cfg['n_hops2n_force'] = {1: nf, 2: nf, 3: nf}
        cfg['n_hops2n_force_distr'] = {1: None, 2: None, 3: None}

    # Make this run produce exactly one dataset into a unique folder
    cfg['n_runs'] = 1
    cfg['n_jobs'] = 1
    cfg['debug']  = False

    trial_export = os.path.join(export_dir, f"trial_{trial_id}")
    cfg['export_dir'] = trial_export
    # Seed per trial for reproducibility & variety
    cfg['seed'] = int(trial_params.get('seed_base', 0) + trial_id)

    return cfg


# Distribution helpers for pat_distr_*
def _empirical_probs_entities(df: pd.DataFrame) -> np.ndarray:
    counts = pd.concat([df["head"], df["tail"]], ignore_index=True).value_counts()
    p = counts.values.astype(float)
    p = p / p.sum()
    return p


def _empirical_probs_relations(df: pd.DataFrame) -> np.ndarray:
    counts = df["rel"].value_counts()
    p = counts.values.astype(float)
    p = p / p.sum()
    return p


def make_empirical_weight_sampler(probs: np.ndarray):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    def sampler(n: int, seed=None):
        rng = _rng_from_seed(seed)
        idx = rng.choice(len(probs), size=int(n), replace=True, p=probs)
        w = probs[idx] + 1e-12 * rng.rand(int(n))  # tiny jitter to reduce ties
        return w
    return sampler


def make_gamma_weight_sampler(shape: float, scale: float):
    def sampler(n: int, seed=None):
        import scipy.stats as st
        return st.gamma.rvs(shape, loc=0.0, scale=scale, size=int(n), random_state=seed)
    return sampler


# Distribution + error helpers
def compute_distributions(df: pd.DataFrame) -> dict:
    """
    Lightweight distributional summaries:
      - relation frequencies (counts per relation id)
      - timestamp frequencies (counts per t)
      - entity degree histogram (degree -> #entities), where degree counts
        appearances as head or tail across all rows.
    Returned as JSON-serializable dicts with int keys converted to str.
    """
    # Relation frequency
    rel_freq = df['rel'].value_counts(sort=False).to_dict()
    rel_freq = {int(k): int(v) for k, v in rel_freq.items()}

    # Timestamp frequency
    time_freq = df['t'].value_counts(sort=False).to_dict()
    time_freq = {int(k): int(v) for k, v in time_freq.items()}

    # Entity degree histogram
    ents = pd.unique(pd.concat([df["head"], df["tail"]], axis=0))
    deg = pd.Series(0, index=pd.Index(ents, name="entity"), dtype=np.int64)
    deg = deg.add(df["head"].value_counts(), fill_value=0)
    deg = deg.add(df["tail"].value_counts(), fill_value=0).fillna(0).astype(np.int64)
    # degree -> count
    deg_hist = pd.Series(deg.values).value_counts(sort=False).to_dict()
    deg_hist = {int(k): int(v) for k, v in deg_hist.items()}

    return {
        "relation_frequency": rel_freq,
        "timestamp_frequency": time_freq,
        "entity_degree_histogram": deg_hist,
    }


def metric_error_breakdown(syn: dict, real: dict, denom_floor: float = 1.0) -> dict:
    """
    For each metric k in _METRIC_KEYS:
      - relerr_k  : normalized_error(syn[k], real[k])
      - absdiff_k : syn[k] - real[k]
    """
    out = {}
    for k in _METRIC_KEYS:
        re = normalized_error(syn[k], real[k], denom_floor=denom_floor)
        ad = float(syn[k] - real[k])
        out[f"relerr_{k}"] = float(re)
        out[f"absdiff_{k}"] = ad
    return out


def extract_synthetic_distributions_from_dir(run_dir: str) -> dict:
    run0 = os.path.join(run_dir, "run_0")
    if not os.path.isdir(run0):
        run0 = run_dir
    df = load_tkg_dir(run0)
    return compute_distributions(df)


# Write helper
def _write_trial_artifacts(
    export_root: str,
    trial_id: int,
    cfg: dict,
    params: dict,
    syn_metrics: dict,
    score: float,
    *,
    real_metrics: dict = None,
    syn_distributions: dict = None,
    real_distributions: dict = None,
) -> None:
    """Write per-trial artifacts and append a summary TSV row with both synthetic and real stats."""
    # Drop callables / internal keys from params so JSON is safe
    safe_params = {k: v for k, v in params.items()
                   if not callable(v) and not k.endswith("_fn")}

    out_dir = cfg["export_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Always write syn metrics
    with open(os.path.join(out_dir, "syn_metrics.json"), "w") as f:
        json.dump(syn_metrics, f, indent=2)

    # Write real/target metrics alongside
    if real_metrics is not None:
        with open(os.path.join(out_dir, "real_metrics.json"), "w") as f:
            json.dump(real_metrics, f, indent=2)

    # Write distributions for syn + real
    if syn_distributions is not None:
        with open(os.path.join(out_dir, "syn_distributions.json"), "w") as f:
            json.dump(syn_distributions, f, indent=2)
    if real_distributions is not None:
        with open(os.path.join(out_dir, "real_distributions.json"), "w") as f:
            json.dump(real_distributions, f, indent=2)

    # Per-metric errors
    errors = metric_error_breakdown(syn_metrics, real_metrics) if real_metrics else {}
    if errors:
        with open(os.path.join(out_dir, "metric_errors.json"), "w") as f:
            json.dump(errors, f, indent=2)

    # Keep trial params + score
    with open(os.path.join(out_dir, "trial_params.json"), "w") as f:
        json.dump(safe_params, f, indent=2)
    with open(os.path.join(out_dir, "score.txt"), "w") as f:
        f.write(f"{score:.6f}\n")

    # Build summary row (keep your original columns, add real_* and error columns)
    summary_tsv = os.path.join(export_root, "summary.tsv")
    row = {
        "trial": trial_id,
        "score": score,
        # Synthetic TKG
        **{k: syn_metrics[k] for k in _METRIC_KEYS},
        # Benchmark TKG
        **({f"real_{k}": real_metrics[k] for k in _METRIC_KEYS} if real_metrics else {}),
        # Errors
        **errors,
        # Params (unchanged)
        "pat_distr_ents": safe_params.get("_pat_distr_ents_choice"),
        "pat_distr_rels": safe_params.get("_pat_distr_rels_choice"),
        "n_ents": safe_params.get("n_ents"),
        "n_rels": safe_params.get("n_rels"),
        "n_tws": safe_params.get("n_tws"),
        "n_1_hop": safe_params.get("n_1_hop"),
        "n_2_hop": safe_params.get("n_2_hop"),
        "n_3_hop": safe_params.get("n_3_hop"),
        "rnd_avg_density": safe_params.get("rnd_avg_density"),
        "p_skip_consequence": safe_params.get("p_skip_consequence"),
        "use_force_distr": safe_params.get("use_force_distr"),
        "p_force": safe_params.get("p_force"),
        "n_force": safe_params.get("n_force"),
        "force_binom_n": safe_params.get("force_binom_n"),
        "force_binom_p": safe_params.get("force_binom_p"),
    }

    # Append summary.tsv
    header_needed = not os.path.exists(summary_tsv)
    with open(summary_tsv, "a") as f:
        if header_needed:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(row[k]) for k in row.keys()) + "\n")


# Helper to handle best trial
def materialize_best_trial(export_root: str, best_trial_id: int) -> str:
    """
    Copy the directory of the best trial to <export_root>/best_trial.
    Returns the destination path.
    """
    src = os.path.join(export_root, f"trial_{best_trial_id}")
    dst = os.path.join(export_root, "best_trial")

    if not os.path.isdir(src):
        raise FileNotFoundError(f"Best trial directory not found: {src}")

    # Replace any previous best_trial directory
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Copy full tree
    shutil.copytree(src, dst)

    # Tiny breadcrumb
    with open(os.path.join(dst, "BEST_FROM.txt"), "w") as f:
        f.write(f"Copied from: {os.path.abspath(src)}\n")

    return dst


# Objective wrapper
def prepare_target_metrics(ref_tkg_dir: str) -> dict:
    target_df = load_tkg_dir(ref_tkg_dir)
    return compute_metrics(target_df)


def extract_synthetic_metrics_from_dir(run_dir: str) -> dict:
    """
    Given the 'export_dir' used by run(), this function reads run_0/(train|valid|test).txt
    and computes metrics on the merged edgelist.
    """
    run0 = os.path.join(run_dir, "run_0")
    if not os.path.isdir(run0):
        # Fallback: sometimes export_dir itself is the run_* directory if caller set it so
        run0 = run_dir
    df = load_tkg_dir(run0)
    return compute_metrics(df)


def objective_factory(
    base_config: dict, ref_tkg_dir: str, export_dir: str, metric_weights: dict,
):
    target = prepare_target_metrics(ref_tkg_dir)

    # Precompute empirical samplers
    ref_tkg_df = load_tkg_dir(ref_tkg_dir)
    _emp_ent_probs = _empirical_probs_entities(ref_tkg_df)
    _emp_rel_probs = _empirical_probs_relations(ref_tkg_df)
    _emp_ent_sampler = make_empirical_weight_sampler(_emp_ent_probs)
    _emp_rel_sampler = make_empirical_weight_sampler(_emp_rel_probs)

    # Compute target distributions and sizes at the study root
    target_distributions = compute_distributions(ref_tkg_df)
    REF_COUNTS = {
        "n_ents": int(target["n_ents"]),
        "n_rels": int(target["n_rels"]),
        "n_tws":  int(target["n_tws"]),
    }

    # Persistent target artifacts
    root_target_metrics = os.path.join(export_dir, "target_metrics.json")
    root_target_dists   = os.path.join(export_dir, "target_distributions.json")
    if not os.path.exists(root_target_metrics):
        with open(root_target_metrics, "w") as f:
            json.dump(target, f, indent=2)
    if not os.path.exists(root_target_dists):
        with open(root_target_dists, "w") as f:
            json.dump(target_distributions, f, indent=2)

    # Define choices & mapping for pat_distr_*
    gamma_options = {
        "gamma(1,2)": (1.0, 2.0),
        "gamma(2,2)": (2.0, 2.0),
        "gamma(3,1)": (3.0, 1.0),
        "gamma(0.5,1.0)": (0.5, 1.0),
    }
    distr_choices = ["uniform", "empirical"] + list(gamma_options.keys())

    def _choice_to_sampler(choice: str, which: str):
        if choice == "uniform":
            return None
        if choice == "empirical":
            return _emp_ent_sampler if which == "ents" else _emp_rel_sampler
        if choice in gamma_options:
            sh, sc = gamma_options[choice]
            return make_gamma_weight_sampler(sh, sc)
        raise ValueError(f"Unknown distribution choice: {choice}")

    def objective_optuna(trial):
        # Suggest params
        params = {}
        params['seed_base'] = 0

        # Entities/relations/timestamps (based on reference TKG)
        params['n_ents'] = REF_COUNTS['n_ents']
        params['n_rels'] = REF_COUNTS['n_rels']
        params['n_tws']  = REF_COUNTS['n_tws']

        # Choose pat_distr_* options
        e_choice = trial.suggest_categorical("pat_distr_ents", distr_choices)
        r_choice = trial.suggest_categorical("pat_distr_rels", distr_choices)
        params['_pat_distr_ents_fn'] = _choice_to_sampler(e_choice, "ents")
        params['_pat_distr_rels_fn'] = _choice_to_sampler(r_choice, "rels")
        # Keep human-readable choices for logging
        params['_pat_distr_ents_choice'] = e_choice
        params['_pat_distr_rels_choice'] = r_choice

        # Pattern counts
        params['n_1_hop'] = trial.suggest_int("n_1_hop", 1, 200)
        params['n_2_hop'] = trial.suggest_int("n_2_hop", 0, 200)
        params['n_3_hop'] = trial.suggest_int("n_3_hop", 0, 200)

        # Booleans pattern hyperparameters
        params['require_unique_triples'] = trial.suggest_categorical(
            "require_unique_triples", [False, True]
        )
        params['prohibit_selfconnections'] = trial.suggest_categorical(
            "prohibit_selfconnections", [False, True]
        )
        params['prohibit_new_consequence_relations'] = trial.suggest_categorical(
            "prohibit_new_consequence_relations", [False, True]
        )
        params['require_sequential_rule'] = trial.suggest_categorical(
            "require_sequential_rule", [False, True]
        )

        # Random wiring mode (disabled)
        params['use_density_dist'] = False
        params['rnd_avg_density'] = 0

        # Skip-consequence prob (disabled)
        params['p_skip_consequence'] = 0

        # Forcing mechanism (effectively a binomial distribution)
        params['use_force_distr'] = False
        params['p_force'] = trial.suggest_float("p_force", 0.0, 1.0, step=0.05)
        params['n_force'] = trial.suggest_int("n_force", 1, 5)

        # Build config for this trial
        cfg = make_config_from_trial(base_config, params, export_dir, trial.number)
        os.makedirs(cfg['export_dir'], exist_ok=True)

        # Generate one dataset
        try:
            generate_one(cfg, run_id=0)
            syn_metrics = extract_synthetic_metrics_from_dir(cfg['export_dir'])

            syn_distributions = extract_synthetic_distributions_from_dir(cfg['export_dir'])
            score = score_metrics(syn_metrics, target, metric_weights)
        except Exception as e:
            # Penalize this region of the space but keep the study running
            syn_metrics = _empty_metrics()
            score = PENALTY_SCORE

            # Persist failure info so you can debug later
            try:
                with open(os.path.join(cfg['export_dir'], "FAILED.txt"), "w") as f:
                    f.write(f"{type(e).__name__}: {e}\n")
            except Exception:
                pass

            _write_trial_artifacts(
                export_dir, trial.number, cfg, params, syn_metrics, score,
                real_metrics=target,
                syn_distributions=None,
                real_distributions=target_distributions,
            )
            trial.set_user_attr("failed", True)
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            print(f"[trial {trial.number:03d}] generation failed -> penalty {score} "
                f"({type(e).__name__}: {e})", file=sys.stderr)
            return score

        # Persist artifacts immediately and print a live line
        _write_trial_artifacts(
            export_dir, trial.number, cfg, params, syn_metrics, score,
            real_metrics=target,
            syn_distributions=syn_distributions,
            real_distributions=target_distributions,
        )
        print(f"[trial {trial.number:03d}] score={score:.4f} | E/R dists={params.get('_pat_distr_ents_choice')}/{params.get('_pat_distr_rels_choice')} | "
              f"n_hops=({params['n_1_hop']},{params['n_2_hop']},{params['n_3_hop']}) | n_ents={params['n_ents']} n_rels={params['n_rels']} n_tws={params['n_tws']}",
              flush=True)

        # Dashboard logging
        trial.set_user_attr("syn_metrics", syn_metrics)
        trial.set_user_attr("config_fragment", {
            k: cfg[k] for k in [
                'n_ents','n_rels','n_tws','n_1_hop','n_2_hop','n_3_hop',
                'require_unique_triples','prohibit_selfconnections',
                'prohibit_new_consequence_relations','require_sequential_rule',
                'rnd_avg_density','p_skip_consequence'
            ] if k in cfg
        } | {
            'pat_distr_ents': params.get('_pat_distr_ents_choice'),
            'pat_distr_rels': params.get('_pat_distr_rels_choice'),
        })
        return score

    return objective_optuna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_tkg_dir", required=True, help="Path to reference TKG folder with train/valid/test.txt")
    ap.add_argument("--budget", type=int, default=20, help="Number of datasets (trials) to generate")
    ap.add_argument("--export_root", default="opt_runs", help="Where to write trial results")
    ap.add_argument("--base_config", default=None,
                    help="Optional path to a JSON file with base config overrides (merged into your default).")
    ap.add_argument("--weights", default=None,
                    help="JSON dict of metric weights, e.g. '{\"n_ents\":2,\"avg_degree\":3}'")
    args = ap.parse_args()

    os.makedirs(args.export_root, exist_ok=True)

    # Base config template; Note: most search spaces defined in objective_optuna
    base_config = {
        'export_dir': args.export_root,
        'seed': 0,
        'debug': False,
        'split': (.8, .1, .1),
        'fast_force_consequence': True,
        'n_runs': 1,
        'n_jobs': 1,
        'n_ents': 50,
        'n_rels': 20,
        'n_tws': 50,
        'pat_distr_ents': None,
        'pat_distr_rels': None,
        'require_unique_triples': False,
        'prohibit_selfconnections': False,
        'prohibit_new_consequence_relations': True,
        'require_sequential_rule': False,
        'n_3_hop': 10,
        # Defined here
        'time_lag_3_hop': [
            (1, lambda seed: __import__("scipy").stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: __import__("scipy").stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: __import__("scipy").stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        'n_2_hop': 10,
        # Defined here
        'time_lag_2_hop': [
            (1, lambda seed: __import__("scipy").stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: __import__("scipy").stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        'n_1_hop': 10,
        # Defined here
        'time_lag_1_hop': [
            (1, lambda seed: __import__("scipy").stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        'max_retries': 3,
        'rnd_avg_density': 0,
        'rnd_avg_density_distr': None,
        'p_skip_consequence': 0.0,
        'n_hops2p_force': {1: 1.0, 2: 1.0, 3: 1.0},
        'n_hops2n_force': {1: 1, 2: 1, 3: 1},
        'n_hops2n_force_distr': {1: None, 2: None, 3: None},
    }

    # Optional JSON override for base
    if args.base_config:
        with open(args.base_config, "r") as fh:
            override = json.load(fh)
        base_config.update(override)

    # Metric weights
    metric_weights = None
    if args.weights:
        metric_weights = json.loads(args.weights)

    objective_optuna = objective_factory(
        base_config=base_config,
        ref_tkg_dir=args.ref_tkg_dir,
        export_dir=args.export_root,
        metric_weights=metric_weights or None,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_optuna, n_trials=args.budget, show_progress_bar=True)

    print("\n=== Best trial ===")
    print("Number:", study.best_trial.number)
    print("Score :", study.best_value)
    print("Params:", json.dumps(study.best_trial.params, indent=2))
    print("Syn metrics:", json.dumps(study.best_trial.user_attrs.get("syn_metrics", {}), indent=2))

    # Copy best trial’s outputs into export_root/best_trial
    try:
        best_dir = materialize_best_trial(args.export_root, study.best_trial.number)
        print(f"\nBest trial copied to: {best_dir}")
    except Exception as e:
        print(f"\n[WARN] Failed to copy best trial: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Run commands:
python3 optimize_tkg.py \
    --ref_tkg_dir /nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/ICEWS14 \
    --budget 100 \
    --export_root opt_runs_icews14_20250916

python3 optimize_tkg.py \
    --ref_tkg_dir /nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/ICEWS18 \
    --budget 100 \
    --export_root opt_runs_icews18_20250916

python3 optimize_tkg.py \
    --ref_tkg_dir /nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/YAGO \
    --budget 100 \
    --export_root opt_runs_yago_20250916

python3 optimize_tkg.py \
    --ref_tkg_dir /nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/WIKI \
    --budget 100 \
    --export_root opt_runs_wiki_20250916

Use --weights to emphasize certain metrics, e.g.:
--weights '{"n_ents":2,"n_rels":2,"n_quads":3,"avg_degree":3,"deg_p25":1,"deg_p75":1,"n_tws":1,"n_unique_triples":1}'

"""