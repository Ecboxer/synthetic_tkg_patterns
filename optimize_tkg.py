import argparse
import ast
import json
import os
import shutil
import sys

from collections import defaultdict
from copy import deepcopy
from typing import Callable, List, Tuple

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

GAMMA_OPTIONS = {
    "gamma(1,2)": (1.0, 2.0),
    "gamma(2,2)": (2.0, 2.0),
    "gamma(3,1)": (3.0, 1.0),
    "gamma(0.5,1.0)": (0.5, 1.0),
}

COLS_PUBLIC = ["head","rel","tail","t","wt","pattern"]
COLS_META = COLS_PUBLIC + ["is_antecedent","is_consequence"]

LagSpec = List[Tuple[int, Callable]]


def _normalize_pattern_col(series: pd.Series) -> list:
    out = []
    for cell in series:
        if isinstance(cell, list):
            out.append(cell)
        elif pd.isna(cell) or cell == "" or cell == "[]":
            out.append([])
        else:
            try:
                v = ast.literal_eval(str(cell))
                out.append(list(v) if isinstance(v, (list, tuple)) else [str(v)])
            except Exception:
                out.append([str(cell)])
    return out

def _read_split_any(run_dir: str, split: str) -> pd.DataFrame:
    """ Read edgelist files.
    Prefer <split>.meta.tsv (headered, includes flags). Fallback to <split>.txt (no flags).
    Returns a DataFrame with at least: head, rel, tail, t, wt, pattern, is_antecedent, is_consequence
    """
    p_meta = os.path.join(run_dir, f"{split}.meta.tsv")
    p_pub  = os.path.join(run_dir, f"{split}.txt")

    if os.path.exists(p_meta):
        df = pd.read_csv(p_meta, sep="\t") #, engine="python")
        # Ensure dtypes
        for c in ["head","rel","tail","t","is_antecedent","is_consequence"]:
            df[c] = pd.to_numeric(df[c], errors="raise", downcast="integer")
        # pattern: allow lists/strings
        if "pattern" in df.columns:
            df["pattern"] = _normalize_pattern_col(df["pattern"])
        else:
            df["pattern"] = [[]]*len(df)
        # wt
        if "wt" not in df.columns:
            df["wt"] = 1
        return df[COLS_META]

    # Fallback: public TSV (no header, no flags)
    df = pd.read_csv(p_pub, sep="\t", header=None, names=COLS_PUBLIC) #, engine="python")
    for c in ["head","rel","tail","t"]:
        df[c] = pd.to_numeric(df[c], errors="raise", downcast="integer")
    if "pattern" in df.columns:
        df["pattern"] = _normalize_pattern_col(df["pattern"])
    else:
        df["pattern"] = [[]]*len(df)
    df["wt"] = df.get("wt", 1)
    # synthesize flags: we don't know; assume not antecedent, not consequence
    df["is_antecedent"] = 0
    df["is_consequence"] = 0
    return df[COLS_META]

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
    df = pd.read_csv(path, sep="\t", header=None, dtype=str) #, engine="python")
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


def _load_json_arg(val: str):
    """If val is empty -> None. If it is a path to a file -> json.load(file). Otherwise -> json.loads(val)."""
    val = (val or "").strip()
    if not val:
        return None
    if os.path.exists(val):
        with open(val, "r") as f:
            return json.load(f)
    return json.loads(val)


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
    cfg['prevent_quad_collisions'] = bool(trial_params['prevent_quad_collisions'])
    cfg['prevent_triple_collisions'] = bool(trial_params['prevent_triple_collisions'])

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

def rehydrate_config_for_labeled_pass(
    base_cfg: dict,
    best_params: dict,
    ref_tkg_dir: str,
    export_root: str,
    trial_id: int,
) -> dict:
    """
    Build a clean, runnable config for the labeled rerun:
      - Recreate the same trial config (sizes, counts, seeds) via make_config_from_trial.
      - Re-map pat_distr_* categorical choices back to real callables.
      - Restore callable distributions from base_cfg where needed.
      - Force skip_labeling=False and put outputs under <export_root>/best_trial_labeled.
    """
    # Start from the SAME construction path the trial used
    cfg = make_config_from_trial(
        base_config=base_cfg,
        trial_params=deepcopy(best_params),
        export_dir=export_root,
        trial_id=trial_id,
    )

    # Recreate samplers exactly as in the trial
    choice_to_sampler = build_choice_to_sampler(ref_tkg_dir)
    e_choice = best_params.get("pat_distr_ents", "uniform")
    r_choice = best_params.get("pat_distr_rels", "uniform")
    cfg["pat_distr_ents"] = choice_to_sampler(e_choice, "ents")
    cfg["pat_distr_rels"] = choice_to_sampler(r_choice, "rels")

    # Ensure forced-instantiation distribution callables exist if trials used them
    nhd = cfg.get("n_hops2n_force_distr", None)
    if not (isinstance(nhd, dict) and all((v is None or callable(v)) for v in nhd.values())):
        cfg["n_hops2n_force_distr"] = deepcopy(base_cfg.get("n_hops2n_force_distr", {1: None, 2: None, 3: None}))

    # Random wiring sampler: keep what the trial used; guard against accidental JSONification
    if cfg.get("rnd_avg_density_distr") is not None and not callable(cfg["rnd_avg_density_distr"]):
        cfg["rnd_avg_density_distr"] = None  # trials in this code path used density, not a distribution callable

    # Time lag specs: keep original base callables/tuples (trials never mutate these)
    for key in ("time_lag_1_hop", "time_lag_2_hop", "time_lag_3_hop"):
        cfg[key] = deepcopy(base_cfg.get(key, []))

    # Final labeled pass settings
    cfg["skip_labeling"] = False
    cfg["export_dir"] = os.path.join(export_root, "best_trial_labeled")
    os.makedirs(cfg["export_dir"], exist_ok=True)

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


def make_zipf_weight_sampler(s: float = 1.3, q: float = 1.0):
    import numpy as np
    def sampler(n: int, seed=None):
        rng = _rng_from_seed(seed)
        ranks = np.arange(1, int(n) + 1, dtype=float)
        w = 1.0 / np.power(ranks + q, s)
        # tiny jitter for ties
        w = w / w.sum()
        # Return weights, not a categorical sample
        # shuffle to avoid perfect correlation with id order
        rng.shuffle(w)
        return w
    return sampler


def make_pareto_weight_sampler(alpha: float = 1.4, xm: float = 1.0, cap: float = None):
    import numpy as np
    def sampler(n: int, seed=None):
        rng = _rng_from_seed(seed)
        w = rng.pareto(alpha, size=int(n)) + 1.0
        w = w * xm
        if cap is not None:
            w = np.minimum(w, cap)
        return w
    return sampler


def make_lognorm_weight_sampler(mu: float = 0.0, sigma: float = 1.2):
    def sampler(n: int, seed=None):
        import numpy as np
        rng = _rng_from_seed(seed)
        return rng.lognormal(mean=mu, sigma=sigma, size=int(n))
    return sampler


def make_negbin_weight_sampler(r: float = 1.5, p: float = 0.3):
    def sampler(n: int, seed=None):
        import numpy as np
        rng = _rng_from_seed(seed)
        # add 1 to avoid zeros, then as float weights
        return rng.negative_binomial(r, p, size=int(n)).astype(float) + 1.0
    return sampler


def build_choice_to_sampler(ref_tkg_dir: str):
    """
    Returns a function choice_to_sampler(choice, which) that maps a string choice
    ('uniform' | 'empirical' | one of GAMMA_OPTIONS) to a callable sampler or None.
    """
    ref_df = load_tkg_dir(ref_tkg_dir)
    emp_ent = make_empirical_weight_sampler(_empirical_probs_entities(ref_df))
    emp_rel = make_empirical_weight_sampler(_empirical_probs_relations(ref_df))

    # preset menu -> callable factory
    MENU = {
        # existing
        **{name: (lambda sh=sc[0], sca=sc[1]: make_gamma_weight_sampler(sh, sca))
           for name, sc in GAMMA_OPTIONS.items()},
        # new: zipf
        "zipf(s=1.2,q=1.0)": (lambda: make_zipf_weight_sampler(1.2, 1.0)),
        "zipf(s=1.4,q=1.0)": (lambda: make_zipf_weight_sampler(1.4, 1.0)),
        "zipf(s=1.7,q=0.5)": (lambda: make_zipf_weight_sampler(1.7, 0.5)),
        # new: pareto
        "pareto(a=1.3,xm=1.0)": (lambda: make_pareto_weight_sampler(1.3, 1.0)),
        "pareto(a=1.6,xm=1.0)": (lambda: make_pareto_weight_sampler(1.6, 1.0)),
        # new: lognormal
        "lognorm(mu=0,s=1.0)": (lambda: make_lognorm_weight_sampler(0.0, 1.0)),
        "lognorm(mu=0,s=1.4)": (lambda: make_lognorm_weight_sampler(0.0, 1.4)),
        # new: negbin
        "negbin(r=1.2,p=0.35)": (lambda: make_negbin_weight_sampler(1.2, 0.35)),
        "negbin(r=2.0,p=0.25)": (lambda: make_negbin_weight_sampler(2.0, 0.25)),
    }

    def choice_to_sampler(choice: str, which: str):
        if choice == "uniform":
            return None
        if choice == "empirical":
            return emp_ent if which == "ents" else emp_rel
        if choice in MENU:
            return MENU[choice]()
        if choice in GAMMA_OPTIONS:
            sh, sc = GAMMA_OPTIONS[choice]
            return make_gamma_weight_sampler(sh, sc)
        raise ValueError(f"Unknown distribution choice: {choice}")
    
    return choice_to_sampler


def build_choice_to_sampler_no_ref():
    """
    For synthetic-only experiments with no reference TKG: support 'uniform' + parametric samplers.
    """
    MENU = {
        **{name: (lambda sh=sc[0], sca=sc[1]: make_gamma_weight_sampler(sh, sca))
           for name, sc in GAMMA_OPTIONS.items()},
        "zipf(s=1.2,q=1.0)": (lambda: make_zipf_weight_sampler(1.2, 1.0)),
        "zipf(s=1.4,q=1.0)": (lambda: make_zipf_weight_sampler(1.4, 1.0)),
        "zipf(s=1.7,q=0.5)": (lambda: make_zipf_weight_sampler(1.7, 0.5)),
        "pareto(a=1.3,xm=1.0)": (lambda: make_pareto_weight_sampler(1.3, 1.0)),
        "pareto(a=1.6,xm=1.0)": (lambda: make_pareto_weight_sampler(1.6, 1.0)),
        "lognorm(mu=0,s=1.0)": (lambda: make_lognorm_weight_sampler(0.0, 1.0)),
        "lognorm(mu=0,s=1.4)": (lambda: make_lognorm_weight_sampler(0.0, 1.4)),
        "negbin(r=1.2,p=0.35)": (lambda: make_negbin_weight_sampler(1.2, 0.35)),
        "negbin(r=2.0,p=0.25)": (lambda: make_negbin_weight_sampler(2.0, 0.25)),
    }
    def choice_to_sampler(choice: str, which: str):
        if choice == "uniform":
            return None
        if choice in MENU:
            return MENU[choice]()
        raise ValueError(f"Unknown distribution choice (no-ref mode): {choice}")
    return choice_to_sampler


# Time lag helpers
def _poisson_sampler(lam):
    import scipy.stats as st
    return lambda seed=None: int(st.poisson(mu=float(lam)).rvs(1, random_state=seed)[0])

def _uniform_sampler(lo, hi_inclusive):
    lo = int(lo); hi = int(hi_inclusive)
    def f(seed=None):
        rng = _rng_from_seed(seed)
        # randint high is exclusive; add 1
        return int(rng.randint(lo, hi + 1))
    return f

def _clip_sampler(base_sampler, lo, hi):
    lo = int(lo); hi = int(hi)
    return lambda seed=None: int(max(lo, min(hi, int(base_sampler(seed)))))

def _repeat_lag(sampler, k=3):
    """
    Helper: make a 3-entry lag spec [ (1,sampler), (1,sampler), (1,sampler) ].
    1-hop will use first, 2-hop first two, 3-hop all three.
    """
    return [(1, sampler) for _ in range(k)]

# For 1-hop use first entry; for 2-hop use first two; for 3-hop use all three.
LAG_PROFILE_MENU = {
    # Very tight recency: 1–3 steps back
    "u1_3": _repeat_lag(_uniform_sampler(1, 3)),

    # Short lags: 1–5 steps
    "u1_5": _repeat_lag(_uniform_sampler(1, 5)),

    # Moderately short: 1–8 steps
    "u1_8": _repeat_lag(_uniform_sampler(1, 8)),

    # Wider but still reasonable: 1-11 steps
    "u1_11": _repeat_lag(_uniform_sampler(1, 11)),

    # Poisson-ish, clipped to 1–5 (slightly more mass near small lags)
    "poi3_5": _repeat_lag(_clip_sampler(_poisson_sampler(3), 1, 5)),

    # Poisson-ish, clipped to 1–10 (slightly more mass near small lags)
    "poi3_10": _repeat_lag(_clip_sampler(_poisson_sampler(3), 1, 10)),
}

def build_time_lag_list(k_hops: int, profile_name: str):
    tpl = LAG_PROFILE_MENU[profile_name]
    if k_hops < 1 or k_hops > 3:
        raise ValueError(f"k_hops must be 1..3, got {k_hops}")
    return tpl[:k_hops]


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
        # Params
        "pat_distr_ents": safe_params.get("_pat_distr_ents_choice"),
        "pat_distr_rels": safe_params.get("_pat_distr_rels_choice"),
        "cfg_n_ents": safe_params.get("n_ents"),
        "cfg_n_rels": safe_params.get("n_rels"),
        "cfg_n_tws": safe_params.get("n_tws"),
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


# Recency and frequency baseline helpers
def _mk_query_index(df_all: pd.DataFrame):
    """
    Build hist[(h,r)] = list of (t, tail) sorted by t ascending
    """
    df_all = df_all.sort_values(["t"]).reset_index(drop=True)
    hist = {}
    for h, r, t, tail in df_all[["head","rel","t","tail"]].itertuples(index=False, name=None):
        hist.setdefault((int(h), int(r)), []).append((int(t), int(tail)))
    return hist

def _iter_test_queries(test_df: pd.DataFrame, max_q: int):
    # per-row queries (h, r, t, true_tail)
    if max_q is not None and max_q < len(test_df):
        test_df = test_df.sample(max_q, random_state=0)
    for h, r, t, tail in test_df[["head","rel","t","tail"]].itertuples(index=False, name=None):
        yield int(h), int(r), int(t), int(tail)

def _hits_at_k_from_ranklist(true_tail: int, rank_list: list[int], Ks: list[int]) -> dict:
    pos = {tail: i for i, tail in enumerate(rank_list)}
    out = {}
    for k in Ks:
        out[f"@{k}"] = 1.0 if (true_tail in pos and pos[true_tail] < k) else 0.0
    return out

def _recency_ranklist(hist_list: list[tuple[int,int]], t: int) -> list[int]:
    """Return unique tails ranked by most recent occurrence before t."""
    # walk backwards and collect first-seen tails
    seen, ranked = set(), []
    for t_i, tail in reversed(hist_list):
        if t_i >= t:
            continue
        if tail not in seen:
            seen.add(tail)
            ranked.append(tail)
    return ranked

def _frequency_ranklist(hist_list: list[tuple[int,int]], t: int, history_len: int) -> list[int]:
    """Look back to last `history_len` occurrences before t; rank by frequency desc, tiebreak by recency."""
    window = []
    for t_i, tail in reversed(hist_list):
        if t_i >= t:
            continue
        window.append((t_i, tail))
        if len(window) >= history_len:
            break
    if not window:
        return []
    # counts + most recent tiebreaker
    freq, last_t = defaultdict(int), {}
    for t_i, tail in window:
        freq[tail] += 1
        last_t[tail] = max(last_t.get(tail, -10**18), t_i)
    # sort by (-count, -last_t)
    ranked = sorted(freq.keys(), key=lambda x: (-freq[x], -last_t[x]))
    return ranked

def evaluate_baselines(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    recency_k_list: list[int],
    frequency_k_list: list[int],
    history_len_list: list[int],
    max_queries: int = None
) -> dict:
    """
    Returns a flat dict, e.g.:
      {
        "recency@1": 0.24, "recency@3": ...,
        "frequency(h=5)@1": ..., "frequency(h=50)@10": ...
      }
    """
    # pool history from *all* prior edges (train+valid+test with t'<t); typical is train+valid, but this is robust
    df_all = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    hist = _mk_query_index(df_all)

    # iterate queries
    n = 0
    acc_rec = {k: 0.0 for k in recency_k_list}
    acc_freq = {(h, k): 0.0 for h in history_len_list for k in frequency_k_list}

    for h, r, t, true_tail in _iter_test_queries(test_df, max_queries):
        lst = hist.get((h, r))
        if not lst:
            continue
        # RECAP: evaluate per-row truth (tail at exactly (h,r,t))
        n += 1

        if recency_k_list:
            rlist = _recency_ranklist(lst, t)
            for k in recency_k_list:
                acc_rec[k] += _hits_at_k_from_ranklist(true_tail, rlist, [k])[f"@{k}"]

        for H in history_len_list or []:
            rlist = _frequency_ranklist(lst, t, history_len=H)
            for k in frequency_k_list:
                acc_freq[(H, k)] += _hits_at_k_from_ranklist(true_tail, rlist, [k])[f"@{k}"]

    # finalize
    out = {}
    denom = max(n, 1)
    for k in recency_k_list:
        out[f"recency@{k}"] = acc_rec[k] / denom
    for H in history_len_list or []:
        for k in frequency_k_list:
            out[f"frequency(h={H})@{k}"] = acc_freq[(H, k)] / denom
    return out

def evaluate_baselines_consequence_only(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    *,
    recency_k_list=(1,3,5,10),
    frequency_k_list=(1,3,5,10),
    history_len_list=(3,10),
    max_queries: int = None
) -> dict:
    """
    Evaluate baselines **only on test consequences** (rows with is_consequence==1),
    using history from **all edges in train+valid+test** but strictly before t*.
    """
    # Build history from all edges
    history_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    hist = _mk_query_index(history_df)

    # Select test consequence queries
    qdf = test_df.loc[(test_df["is_consequence"] == 1), ["head","rel","t","tail"]].copy()
    if qdf.empty:
        # Defensive fallback: evaluate on all test if flags missing
        qdf = test_df[["head","rel","t","tail"]].copy()
    
    # Cap on number of queries
    if max_queries is not None and max_queries < len(qdf):
        qdf = qdf.sample(max_queries, random_state=0)

    # Accumulate H@K
    n = 0
    acc_rec = {k: 0.0 for k in recency_k_list}
    acc_freq = {(H, k): 0.0 for H in history_len_list for k in frequency_k_list}

    for h, r, t_star, true_tail in qdf.itertuples(index=False, name=None):
        lst = hist.get((int(h), int(r)))
        if not lst:
            continue
        n += 1

        # recency
        rlist = _recency_ranklist(lst, int(t_star))
        pos = {a:i for i,a in enumerate(rlist)}
        for k in recency_k_list:
            acc_rec[k] += 1.0 if (int(true_tail) in pos and pos[int(true_tail)] < k) else 0.0

        # frequency
        for H in history_len_list:
            flist = _frequency_ranklist(lst, int(t_star), H)
            p2 = {a:i for i,a in enumerate(flist)}
            for k in frequency_k_list:
                acc_freq[(H,k)] += 1.0 if (int(true_tail) in p2 and p2[int(true_tail)] < k) else 0.0

    denom = max(n, 1)
    out = {f"recency@{k}": acc_rec[k] / denom for k in recency_k_list}
    for H in history_len_list:
        for k in frequency_k_list:
            out[f"frequency(h={H})@{k}"] = acc_freq[(H, k)] / denom
    return out

def score_baseline_similarity(syn_b: dict, ref_b: dict) -> float:
    """
    Average absolute difference across common baseline keys.
    (All Hits@K are in [0,1], so denom=1.)
    """
    keys = sorted(set(syn_b.keys()) & set(ref_b.keys()))
    if not keys:
        return 0.0
    diffs = [abs(float(syn_b[k]) - float(ref_b[k])) for k in keys]
    return float(np.mean(diffs))


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
    base_config: dict,
    export_dir: str,
    metric_weights: dict,
    *,
    recency_k_list: list[int],
    frequency_k_list: list[int],
    history_len_list: list[int],
    baseline_weight: float,
    baseline_max_queries_ref: int,
    baseline_max_queries_syn: int,
    ref_tkg_dir: str = None,
    target_metrics: dict = None,
    target_baselines: dict = None,
    n1_range=None,
    n2_range=None,
    n3_range=None,
    p_force_range=None,
    n_force_range=None,
    pat_ents_choices_cli=None,
    pat_rels_choices_cli=None,
):
    # Target metrics
    if target_metrics is not None:
        target = dict(target_metrics)
    else:
        if ref_tkg_dir is None:
            raise ValueError("objective_factory: need either ref_tkg_dir or target_metrics.")
        target = prepare_target_metrics(ref_tkg_dir)

    # Target distributions
    if ref_tkg_dir is not None:
        ref_tkg_df = load_tkg_dir(ref_tkg_dir)
        tgt_dists = compute_distributions(ref_tkg_df)
    else:
        tgt_dists = None  # distributions are optional

    REF_COUNTS = {
        "n_ents": int(target["n_ents"]),
        "n_rels": int(target["n_rels"]),
        "n_tws":  int(target["n_tws"]),
    }

    # Persist target artifacts to export_root (if available)
    root_target_metrics = os.path.join(export_dir, "target_metrics.json")
    if not os.path.exists(root_target_metrics):
        with open(root_target_metrics, "w") as f:
            json.dump(target, f, indent=2)
    root_target_dists = os.path.join(export_dir, "target_distributions.json")
    if tgt_dists is not None and not os.path.exists(root_target_dists):
        with open(root_target_dists, "w") as f:
            json.dump(tgt_dists, f, indent=2)

    # Reference TKG baseline performance
    if baseline_weight > 0:
        if target_baselines is not None:
            target_baselines_local = dict(target_baselines)
        elif ref_tkg_dir is not None:
            # compute from reference TKG
            ref_train = _read_edgelist_any(os.path.join(ref_tkg_dir, "train.txt"))
            ref_valid = _read_edgelist_any(os.path.join(ref_tkg_dir, "valid.txt"))
            ref_test = _read_edgelist_any(os.path.join(ref_tkg_dir, "test.txt"))

            ref_baseline_path = os.path.join(export_dir, "target_baselines.json")
            if os.path.exists(ref_baseline_path):
                with open(ref_baseline_path, "r") as f:
                    target_baselines_local = json.load(f)
            else:
                target_baselines_local = evaluate_baselines(
                    train_df=ref_train, valid_df=ref_valid, test_df=ref_test,
                    recency_k_list=recency_k_list,
                    frequency_k_list=frequency_k_list,
                    history_len_list=history_len_list,
                    max_queries=baseline_max_queries_ref,
                )
                with open(ref_baseline_path, "w") as f:
                    json.dump(target_baselines_local, f, indent=2)
        else:
            # No way to define baseline similarity; silently disable it
            print("[objective_factory] baseline_weight>0 but no ref_tkg_dir/target_baselines; disabling baseline term.", file=sys.stderr)
            baseline_weight = 0.0
            target_baselines_local = {}
    else:
        target_baselines_local = {}
    
    # Define choices & mapping for pat_distr_*
    if ref_tkg_dir is not None:
        choice_to_sampler = build_choice_to_sampler(ref_tkg_dir)
        distr_choices = (
            ["empirical", "uniform"]
            + list(GAMMA_OPTIONS.keys())
            + [
                "zipf(s=1.2,q=1.0)", "zipf(s=1.4,q=1.0)", "zipf(s=1.7,q=0.5)",
                "pareto(a=1.3,xm=1.0)", "pareto(a=1.6,xm=1.0)",
                "lognorm(mu=0,s=1.0)", "lognorm(mu=0,s=1.4)",
                "negbin(r=1.2,p=0.35)", "negbin(r=2.0,p=0.25)",
            ]
        )
        distr_choices_ent = distr_choices  # same menu for entities
    else:
        # No "empirical" option
        choice_to_sampler = build_choice_to_sampler_no_ref()
        distr_choices = (
            ["uniform"]
            + list(GAMMA_OPTIONS.keys())
            + [
                "zipf(s=1.2,q=1.0)", "zipf(s=1.4,q=1.0)", "zipf(s=1.7,q=0.5)",
                "pareto(a=1.3,xm=1.0)", "pareto(a=1.6,xm=1.0)",
                "lognorm(mu=0,s=1.0)", "lognorm(mu=0,s=1.4)",
                "negbin(r=1.2,p=0.35)", "negbin(r=2.0,p=0.25)",
            ]
        )
        distr_choices_ent = distr_choices
    
    def _filter_menu(base, allowed):
        if not allowed:
            return base
        allowed = set(allowed)
        out = [x for x in base if x in allowed]
        if not out:
            raise ValueError(f"No overlap between requested pat_distr_* choices {allowed} and base menu {base}")
        return out

    distr_choices_ent = _filter_menu(distr_choices_ent, pat_ents_choices_cli)
    distr_choices = _filter_menu(
        distr_choices,
        (pat_rels_choices_cli or pat_ents_choices_cli)
    )

    def objective_optuna(trial):
        # Suggest params
        params = {}
        params['seed_base'] = 0

        # Entities/relations/timestamps (based on reference TKG)
        params['n_ents'] = REF_COUNTS['n_ents']
        params['n_rels'] = REF_COUNTS['n_rels']
        params['n_tws']  = REF_COUNTS['n_tws']

        # Choose pat_distr_* options
        e_choice = trial.suggest_categorical("pat_distr_ents", distr_choices_ent)
        r_choice = trial.suggest_categorical("pat_distr_rels", distr_choices)
        # params['_pat_distr_ents_fn'] = _choice_to_sampler(e_choice, "ents")
        # params['_pat_distr_rels_fn'] = _choice_to_sampler(r_choice, "rels")
        params['_pat_distr_ents_fn'] = choice_to_sampler(e_choice, "ents")
        params['_pat_distr_rels_fn'] = choice_to_sampler(r_choice, "rels")
        # Keep human-readable choices for logging
        params['_pat_distr_ents_choice'] = e_choice
        params['_pat_distr_rels_choice'] = r_choice

        # Pattern counts
        if n1_range is not None:
            lo, hi, step = n1_range
            params['n_1_hop'] = trial.suggest_int("n_1_hop", lo, hi, step=step)
        else:
            params['n_1_hop'] = trial.suggest_int("n_1_hop", 25, 350, step=25)

        if n2_range is not None:
            lo, hi, step = n2_range
            params['n_2_hop'] = trial.suggest_int("n_2_hop", lo, hi, step=step)
        else:
            params['n_2_hop'] = trial.suggest_int("n_2_hop", 25, 250, step=25)

        if n3_range is not None:
            lo, hi, step = n3_range
            params['n_3_hop'] = trial.suggest_int("n_3_hop", lo, hi, step=step)
        else:
            params['n_3_hop'] = trial.suggest_int("n_3_hop", 25, 200, step=25)

        # Booleans pattern hyperparameters
        params['require_unique_triples'] = True  # EB: Fix to True
        params['prohibit_selfconnections'] = False  # EB: Fix to False
        params['prohibit_new_consequence_relations'] = False  # EB: Fix to False
        params['require_sequential_rule'] = False  # EB: Fix to False
        params['prevent_quad_collisions'] = True  # EB: Fix to True
        params['prevent_triple_collisions'] = False  # EB: Fix to False

        # Random wiring mode (disabled)
        params['use_density_dist'] = False
        params['rnd_avg_density'] = 0

        # Skip-consequence prob (disabled)
        params['p_skip_consequence'] = 0

        # Forcing mechanism (binomial-ish)
        params['use_force_distr'] = False

        if p_force_range is not None:
            lo, hi, step = p_force_range
            params['p_force'] = trial.suggest_float("p_force", lo, hi, step=step)
        else:
            params['p_force'] = trial.suggest_float("p_force", 0.1, 0.5, step=0.1)

        if n_force_range is not None:
            lo, hi, step = n_force_range
            params['n_force'] = trial.suggest_int("n_force", lo, hi, step=step)
        else:
            params['n_force'] = trial.suggest_int("n_force", 1, 5, step=1)

        # Build config for this trial
        cfg = make_config_from_trial(base_config, params, export_dir, trial.number)
        os.makedirs(cfg['export_dir'], exist_ok=True)

        # Generate one dataset
        try:
            generate_one(cfg, run_id=0)
            syn_metrics = extract_synthetic_metrics_from_dir(cfg['export_dir'])

            syn_distributions = extract_synthetic_distributions_from_dir(cfg['export_dir'])

            # Synthetic baselines, optionally subsampled
            run0 = os.path.join(cfg["export_dir"], "run_0")
            if not os.path.isdir(run0):
                run0 = cfg["export_dir"]

            syn_train = _read_edgelist_any(os.path.join(run0, "train.txt"))
            syn_valid = _read_edgelist_any(os.path.join(run0, "valid.txt"))
            syn_test = _read_edgelist_any(os.path.join(run0, "test.txt"))
            # syn_train = _read_split_any(run0, "train")
            # syn_valid = _read_split_any(run0, "valid")
            # syn_test  = _read_split_any(run0, "test")

            syn_baselines = evaluate_baselines(
                train_df=syn_train, valid_df=syn_valid, test_df=syn_test,
                recency_k_list=recency_k_list,
                frequency_k_list=frequency_k_list,
                history_len_list=history_len_list,
                max_queries=baseline_max_queries_syn,
            )

            score_core = score_metrics(syn_metrics, target, metric_weights)
            score_base = (
                score_baseline_similarity(syn_baselines, target_baselines_local)
                if baseline_weight > 0 else 0.0
            )
            score = float(score_core + baseline_weight * score_base)

            # Attach baseline results
            trial.set_user_attr("syn_baselines", syn_baselines)
            trial.set_user_attr("ref_baselines", target_baselines)
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
                real_distributions=tgt_dists,
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
            real_distributions=tgt_dists,
        )
        with open(os.path.join(cfg['export_dir'], "syn_baselines.json"), "w") as f:
            json.dump(syn_baselines, f, indent=2)
        print(
            f"[trial {trial.number:03d}] "
            f"score={score:.4f} (core={score_core:.4f}, base*w={baseline_weight*score_base:.4f}) | "
            f"E/R dists={params.get('_pat_distr_ents_choice')}/{params.get('_pat_distr_rels_choice')} | "
            f"n_hops=({params['n_1_hop']},{params['n_2_hop']},{params['n_3_hop']}) | "
            f"n_ents={params['n_ents']} n_rels={params['n_rels']} n_tws={params['n_tws']}",
            flush=True
        )

        # Dashboard logging
        trial.set_user_attr("syn_metrics", syn_metrics)
        trial.set_user_attr("config_fragment", {
            k: cfg[k] for k in [
                'n_ents','n_rels','n_tws','n_1_hop','n_2_hop','n_3_hop',
                'require_unique_triples','prohibit_selfconnections',
                'prohibit_new_consequence_relations','require_sequential_rule',
                'prevent_quad_collisions','prevent_triple_collisions',
                'rnd_avg_density','p_skip_consequence'
            ] if k in cfg
        } | {
            'pat_distr_ents': params.get('_pat_distr_ents_choice'),
            'pat_distr_rels': params.get('_pat_distr_rels_choice'),
        })
        return score

    return objective_optuna

def relabel_run_dir_in_place(run_dir: str, graph_mode: str = "tkg"):
    """
    Inlined, robust relabeler modeled after relabel_full_from_existing.py:
    - Reads pattern2id + train/valid/test.txt
    - Applies the same backtracking semantics as run.py (accelerated per-relation index)
    - Writes train/valid/test.txt back with 'pattern' labels filled (both antecedent and consequence)
    """
    import numpy as np, pandas as pd
    from temporalpattern import TemporalPattern
    import re

    # --- small helpers mirrored from relabel_full_from_existing.py ---
    PH_RE = re.compile(r"^e\d+$")
    def _is_ph(x): return isinstance(x, (str, np.str_)) and PH_RE.match(str(x)) is not None
    def _norm_ph(x): return str(x) if _is_ph(x) else x
    def _seed(mapping, key, value):
        key = _norm_ph(key)
        if not _is_ph(key): return True
        v = int(value)
        if key in mapping: return int(mapping[key]) == v
        mapping[key] = v; return True
    def _all_diff_ok(m): 
        v = list(m.values()); return len(v) == len(set(v))
    def _pattern_placeholders(ants, cons):
        ordered = []
        def add(x):
            x = _norm_ph(x)
            if _is_ph(x) and x not in ordered: ordered.append(x)
        for h,_,ta in ants: add(h); add(ta)
        ch,_,cta = cons; add(ch); add(cta)
        return ordered
    def _binding_to_str(m, placeholders):
        return "|".join(f"{k}={int(m[k])}" for k in placeholders if k in m)
    def _uniq_list_cell(x):
        if isinstance(x, list): return [v for v in x if v not in (None, "", "[]")]
        if x in (None, "", "[]"): return []
        return [str(x)]

    class RelIndex:
        __slots__=("t","head","tail","row_idx")
        def __init__(self, t, head, tail, row_idx):
            order = np.argsort(t, kind="mergesort")
            self.t = t[order]; self.head = head[order]; self.tail = tail[order]; self.row_idx = row_idx[order]
        def window(self, t_min, t_max):
            lo = np.searchsorted(self.t, t_min, side="left")
            hi = np.searchsorted(self.t, t_max, side="right")
            return lo, hi

    def build_rel_index(df_all: pd.DataFrame):
        rel2idx = {}
        arr_rel  = df_all["rel"].to_numpy(np.int64, copy=False)
        arr_head = df_all["head"].to_numpy(np.int64, copy=False)
        arr_tail = df_all["tail"].to_numpy(np.int64, copy=False)
        arr_t    = df_all["t"].to_numpy(np.int64, copy=False)
        arr_row  = np.arange(len(df_all), dtype=np.int64)
        for r in np.unique(arr_rel):
            m = (arr_rel == r)
            rel2idx[int(r)] = RelIndex(arr_t[m].copy(), arr_head[m].copy(), arr_tail[m].copy(), arr_row[m].copy())
        return rel2idx

    def candidates_for_ant(rel_index, ant, t_anchor, lag, is_kg):
        h, r, ta = _norm_ph(ant[0]), int(ant[1]), _norm_ph(ant[2])
        if is_kg:
            lo, hi = 0, len(rel_index.t)
        else:
            min_lag, max_lag = lag
            t_min = int(t_anchor) - int(max_lag)
            t_max = int(t_anchor) - int(min_lag)
            lo, hi = rel_index.window(t_min, t_max)
            if lo >= hi: return None
        sel = np.ones(hi - lo, dtype=bool)
        if not _is_ph(h):  sel &= (rel_index.head[lo:hi] == int(h))
        if not _is_ph(ta): sel &= (rel_index.tail[lo:hi] == int(ta))
        if not np.any(sel): return None
        return (rel_index.row_idx[lo:hi][sel],
                rel_index.head[lo:hi][sel],
                rel_index.tail[lo:hi][sel],
                rel_index.t[lo:hi][sel])

    def backtrack_rev_with_paths(df_all, rel2idx, ants, lags, t_anchors, step, mapping, path, is_kg):
        n = len(ants)
        if step == n:
            yield mapping, path
            return
        orig_idx = n - 1 - step
        ant = ants[orig_idx]; r = int(ant[1])
        rel_index = rel2idx.get(r)
        if rel_index is None: return
        lag = lags[orig_idx]
        for t_anchor in t_anchors:
            block = candidates_for_ant(rel_index, ant, t_anchor, lag, is_kg)
            if block is None: continue
            idxs, heads, tails, ts = block
            for i in range(len(idxs)):
                new_map = dict(mapping)
                h, ta = _norm_ph(ant[0]), _norm_ph(ant[2])
                if _is_ph(h):
                    if not _seed(new_map, h, heads[i]): continue
                elif int(h) != int(heads[i]): continue
                if _is_ph(ta):
                    if not _seed(new_map, ta, tails[i]): continue
                elif int(ta) != int(tails[i]): continue
                if not _all_diff_ok(new_map): continue
                new_path = path + [(orig_idx, int(idxs[i]))]
                yield from backtrack_rev_with_paths(df_all, rel2idx, ants, lags, [int(ts[i])], step+1, new_map, new_path, is_kg)

    is_kg = (str(graph_mode).lower() == "kg")
    pat_path = os.path.join(run_dir, "pattern2id.txt")
    tr_path  = os.path.join(run_dir, "train.txt")
    va_path  = os.path.join(run_dir, "valid.txt")
    te_path  = os.path.join(run_dir, "test.txt")
    for p in (pat_path, tr_path, va_path, te_path):
        if not os.path.exists(p): raise FileNotFoundError(f"Missing: {p}")

    pat_df = pd.read_csv(pat_path, sep="\t", header=None, names=["pattern","n_hops","id"])
    splits = []
    for p in (tr_path, va_path, te_path):
        df = pd.read_csv(p, sep="\t", header=None,
                         names=["head","rel","tail","t","wt","pattern"],
                         dtype={"head":np.int64,"rel":np.int64,"tail":np.int64,"t":np.int64,"wt":float,"pattern":object})
        df["pattern"] = df["pattern"].apply(_uniq_list_cell)
        splits.append(df)
    df_all = pd.concat(splits, ignore_index=True)

    rel2idx = build_rel_index(df_all)

    for _, row in pat_df.iterrows():
        pid = int(row["id"])
        tp = TemporalPattern(); tp.from_label(row["pattern"])

        raw_ants = list(tp.antecedent)
        ants = [(_norm_ph(h), r, _norm_ph(ta)) for (h, r, ta) in raw_ants]
        ch, cr, cta = tp.consequence
        cons_h, cons_r, cons_ta = _norm_ph(ch), int(cr), _norm_ph(cta)
        lags = list(tp.time_lags)

        placeholders = _pattern_placeholders(ants, (cons_h, cons_r, cons_ta))
        same_ph_cons = (_is_ph(cons_h) and _is_ph(cons_ta) and cons_h == cons_ta)

        mask = (df_all["rel"].to_numpy() == cons_r)
        if not _is_ph(cons_h):  mask &= (df_all["head"].to_numpy() == int(cons_h))
        if not _is_ph(cons_ta): mask &= (df_all["tail"].to_numpy() == int(cons_ta))
        if same_ph_cons:        mask &= (df_all["head"].to_numpy() == df_all["tail"].to_numpy())

        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            continue

        sub = df_all.iloc[cand_idx]
        grp_keys = np.core.records.fromarrays(
            (sub["head"].to_numpy(), sub["tail"].to_numpy(), sub["t"].to_numpy()),
            names="h,tail,t"
        )
        order = np.argsort(grp_keys, kind="mergesort")
        grp_keys = grp_keys[order]
        grp_idx_sorted = cand_idx[order]
        bounds = np.flatnonzero(np.r_[True, grp_keys[1:] != grp_keys[:-1], True])

        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            rows = grp_idx_sorted[b0:b1]
            h_val = int(df_all.at[rows[0], "head"])
            ta_val = int(df_all.at[rows[0], "tail"])
            t_anchor = 0 if is_kg else int(df_all.at[rows[0], "t"])

            init_map = {}
            if same_ph_cons:
                if not _seed(init_map, cons_h, h_val): continue
            else:
                if _is_ph(cons_h)  and not _seed(init_map, cons_h,  h_val): continue
                if _is_ph(cons_ta) and not _seed(init_map, cons_ta, ta_val): continue
            if not _all_diff_ok(init_map): continue

            if len(ants) == 0:
                bind_str = _binding_to_str(init_map, placeholders)
                lab_c = f"{pid}_c_{bind_str}"
                for ridx in rows:
                    cell = df_all.at[ridx, "pattern"]
                    if lab_c not in cell:
                        cell.append(lab_c); df_all.at[ridx, "pattern"] = cell
                continue

            for mapping, path in backtrack_rev_with_paths(df_all, rel2idx, ants, lags, [t_anchor], 0, init_map, [], is_kg):
                exp_h = mapping[cons_h] if _is_ph(cons_h) else cons_h
                exp_t = mapping[cons_ta] if _is_ph(cons_ta) else cons_ta
                if (h_val != int(exp_h)) or (ta_val != int(exp_t)):
                    continue
                bind_str = _binding_to_str(mapping, placeholders)

                seen = set()
                for ante_idx, edge_idx in path:
                    if edge_idx in rows:  # never both consequence+antecedent
                        continue
                    ah, ar, at = ants[ante_idx]
                    ahv = mapping[_norm_ph(ah)] if _is_ph(_norm_ph(ah)) else _norm_ph(ah)
                    atv = mapping[_norm_ph(at)] if _is_ph(_norm_ph(at)) else _norm_ph(at)
                    if not (int(df_all.at[edge_idx,"head"]) == int(ahv)
                            and int(df_all.at[edge_idx,"rel"]) == int(ar)
                            and int(df_all.at[edge_idx,"tail"]) == int(atv)):
                        continue
                    lab_a = f"{pid}_a{ante_idx}_{bind_str}"
                    if lab_a in seen:  # avoid duplicate role/bind
                        continue
                    cell = df_all.at[edge_idx, "pattern"]
                    if lab_a not in cell:
                        cell.append(lab_a); df_all.at[edge_idx, "pattern"] = cell
                    seen.add(lab_a)

                lab_c = f"{pid}_c_{bind_str}"
                for ridx in rows:
                    cell = df_all.at[ridx, "pattern"]
                    if lab_c not in cell:
                        cell.append(lab_c); df_all.at[ridx, "pattern"] = cell

    # Write back splits
    n_tr, n_va, n_te = (len(splits[0]), len(splits[1]), len(splits[2]))
    def _write(path, frame): frame.to_csv(path, sep="\t", index=False, header=False)
    start = 0
    tr_out = df_all.iloc[start:start+n_tr]; start += n_tr
    va_out = df_all.iloc[start:start+n_va]; start += n_va
    te_out = df_all.iloc[start:start+n_te]
    _write(os.path.join(run_dir, "train.txt"), tr_out)
    _write(os.path.join(run_dir, "valid.txt"), va_out)
    _write(os.path.join(run_dir, "test.txt"),  te_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_tkg_dir", required=False, help="Path to reference TKG folder with train/valid/test.txt")
    ap.add_argument("--budget", type=int, default=20, help="Number of datasets (trials) to generate")
    ap.add_argument("--export_root", default="opt_runs", help="Where to write trial results")
    ap.add_argument("--base_config", default=None,
                    help="Optional path to a JSON file with base config overrides (merged into your default).")
    ap.add_argument("--weights", default=None,
                    help="JSON dict of metric weights, e.g. '{\"n_ents\":2,\"avg_degree\":3}'")
    ap.add_argument("--skip_labeling",
                    action="store_true",
                    help="Skip the post-generation labeling step inside run() to speed up trials.")
    # Additional criteria: Similarity of the recency and frequency baselines on synthetic and reference TKG
    ap.add_argument("--recency_k", default="",
                    help="Comma-separated Ks for recency Hits@K, e.g. '1,3,10'. Blank disables.")
    ap.add_argument("--frequency_k", default="",
                    help="Comma-separated Ks for frequency Hits@K, e.g. '1,3,10'. Blank disables.")
    ap.add_argument("--history_len", default="",
                    help="Comma-separated history lengths for frequency baseline, e.g. '5,20'. Blank disables.")
    ap.add_argument("--baseline_weight", type=float, default=1.0,
                    help="Weight for baseline-similarity error term in the objective. 0 disables.")
    ap.add_argument("--baseline_max_queries_ref", type=int, default=None,
                    help="Optional cap on # of test queries used for reference baseline eval. None means all queries are run.")
    ap.add_argument("--baseline_max_queries_syn", type=int, default=5_000,
                    help="Optional cap on # of test queries used for synthetic baseline eval to speed up trials.")
    # Parallelization
    # Note: Not well implemented, just slows things down
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Number of parallel trials to run in this process (threads).")
    ap.add_argument("--storage", default="",
                    help="(Optional) Optuna storage URL for multi-process parallelism (e.g. 'sqlite:///path/to/optuna.db').")
    ap.add_argument("--study_name", default="tkg_opt",
                    help="Study name (used only if --storage is provided).")
    # Directly passing target metrics/baselines
    ap.add_argument(
        "--target_metrics",
        default="",
        help="JSON dict or path to JSON file with target descriptive metrics (keys: n_ents, n_rels, n_tws, n_quads, n_unique_triples, avg_degree, deg_p25, deg_p75)."
    )
    ap.add_argument(
        "--target_baselines",
        default="",
        help="JSON dict or path to JSON file with target recency/frequency Hits@K baselines."
    )
    # Search space parameters
    ap.add_argument(
        "--n_1_hop_range",
        default="25:350:25",
        help="min:max:step for n_1_hop search space"
    )
    ap.add_argument(
        "--n_2_hop_range",
        default="25:250:25",
        help="min:max:step for n_2_hop search space"
    )
    ap.add_argument(
        "--n_3_hop_range",
        default="25:200:25",
        help="min:max:step for n_3_hop search space"
    )
    ap.add_argument(
        "--p_force_range",
        default="0.1:0.5:0.1",
        help="min:max:step for p_force search space"
    )
    ap.add_argument(
        "--n_force_range",
        default="1:5:1",
        help="min:max:step for n_force search space"
    )
    ap.add_argument(
        "--lag_profile",
        default="u1_3",
        choices=list(LAG_PROFILE_MENU.keys()),
        help="Name of lag profile to use for 1/2/3-hop time_lag_* entries"
    )
    ap.add_argument(
        "--pat_distr_ents_choices",
        default="",
        help="Comma-separated menu of pat_distr_ents choices to allow (subset of the built-in menu). Empty = full menu."
    )
    ap.add_argument(
        "--pat_distr_rels_choices",
        default="",
        help="Comma-separated menu of pat_distr_rels choices. Empty = same as ents."
    )

    
    args = ap.parse_args()

    def _parse_int_list(s):
        s = (s or "").strip()
        return [int(x) for x in s.split(",") if x.strip().isdigit()]
    
    def _parse_range(spec, cast):
        spec = (spec or "").strip()
        if not spec:
            return None
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad range spec '{spec}', expected 'min:max:step'")
        lo, hi, step = cast(parts[0]), cast(parts[1]), cast(parts[2])
        return lo, hi, step
    
    def _parse_choice_list(s):
        s = (s or "").strip()
        if not s:
            return []
        return [x.strip() for x in s.split() if x.strip()]
    
    target_metrics_cli = _load_json_arg(args.target_metrics)
    target_baselines_cli = _load_json_arg(args.target_baselines)

    if not args.ref_tkg_dir and target_metrics_cli is None:
        raise ValueError(
            "You must provide either --ref_tkg_dir or --target_metrics (or both). "
            "Without a reference TKG you need explicit target metrics."
        )

    recency_k_list = _parse_int_list(args.recency_k)
    frequency_k_list = _parse_int_list(args.frequency_k)
    history_len_list = _parse_int_list(args.history_len)

    os.makedirs(args.export_root, exist_ok=True)

    # Base config template; Note: most search spaces defined in objective_optuna
    base_config = {
        'export_dir': args.export_root,
        'seed': 0,
        'debug': False,
        'skip_labeling': False,
        'split': (.8, .1, .1),
        'fast_force_consequence': True,
        'n_runs': 1,
        'n_jobs': 1,
        'n_ents': 50,
        'n_rels': 20,
        'n_tws': 50,
        'pat_distr_ents': None,
        'pat_distr_rels': None,
        'require_unique_triples': True,
        'prohibit_selfconnections': False,
        'prohibit_new_consequence_relations': True,
        'require_sequential_rule': False,
        'prevent_quad_collisions': True,
        'prevent_triple_collisions': True,
        'max_instantiation_resamples': 20,
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

    # Adjust time lags
    lag_profile = args.lag_profile
    base_config["time_lag_1_hop"] = build_time_lag_list(1, lag_profile)
    base_config["time_lag_2_hop"] = build_time_lag_list(2, lag_profile)
    base_config["time_lag_3_hop"] = build_time_lag_list(3, lag_profile)

    # Optional JSON override for base
    if args.base_config:
        with open(args.base_config, "r") as fh:
            override = json.load(fh)
        base_config.update(override)
    
    # Apply CLI skip to all trials
    if args.skip_labeling:
        base_config['skip_labeling'] = True

    # Metric weights
    metric_weights = None
    if args.weights:
        metric_weights = json.loads(args.weights)
    
    # Search space parameters
    n1_range = _parse_range(args.n_1_hop_range, int)
    n2_range = _parse_range(args.n_2_hop_range, int)
    n3_range = _parse_range(args.n_3_hop_range, int)
    p_force_range = _parse_range(args.p_force_range, float)
    n_force_range = _parse_range(args.n_force_range, int)
    pat_ents_choices_cli = _parse_choice_list(args.pat_distr_ents_choices)
    pat_rels_choices_cli = _parse_choice_list(args.pat_distr_rels_choices)

    objective_optuna = objective_factory(
        base_config=base_config,
        export_dir=args.export_root,
        metric_weights=metric_weights or None,
        recency_k_list=recency_k_list,
        frequency_k_list=frequency_k_list,
        history_len_list=history_len_list,
        baseline_weight=float(args.baseline_weight),
        baseline_max_queries_ref=args.baseline_max_queries_ref,
        baseline_max_queries_syn=args.baseline_max_queries_syn,
        ref_tkg_dir=args.ref_tkg_dir,
        target_metrics=target_metrics_cli,
        target_baselines=target_baselines_cli,
        n1_range=n1_range,
        n2_range=n2_range,
        n3_range=n3_range,
        p_force_range=p_force_range,
        n_force_range=n_force_range,
        pat_ents_choices_cli=pat_ents_choices_cli,
        pat_rels_choices_cli=pat_rels_choices_cli,
    )

    # Create sampler
    sampler = optuna.samplers.TPESampler(
        seed=0,
        n_startup_trials=20,  # Purely random at the beginning
        multivariate=True,
        group=True,
    )

    # Create study (in-memory or backed by storage for multi-process)
    if args.storage:
        # Centralized storage for coordination across multiple processes
        study = optuna.create_study(
            direction="minimize",
            storage=args.storage,
            study_name=args.study_name,
            load_if_exists=True,
            sampler=sampler,
        )
    else:
        # No storage, single-process study
        study = optuna.create_study(direction="minimize", sampler=sampler)
    
    study.optimize(
        objective_optuna,
        n_trials=args.budget,
        show_progress_bar=True,
        n_jobs=max(1, args.n_jobs),
    )

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
    
    # If we skipped labeling, rerun the best trial with labeling on
    if args.skip_labeling:
        try:
            print("\n[post] Relabeling best trial in place (robust path)...")
            best_dir = os.path.join(args.export_root, "best_trial")
            relabel_run_dir_in_place(best_dir, graph_mode="tkg")
            # Optionally copy to an explicit labeled folder
            # shutil.copytree(best_dir, os.path.join(args.export_root, "best_trial_labeled"), dirs_exist_ok=True)
            print(f"[post] Relabeling complete: {best_dir}")
        except Exception as e:
            print(f"[post][WARN] Failed to relabel best trial: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Run commands:
# DONE ckg05
python3 optimize_tkg.py \
    --ref_tkg_dir /nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/ICEWS14 \
    --budget 800 \
    --skip_labeling \
    --recency_k '1,3,10' \
    --frequency_k '1,3,10' \
    --history_len '1,3,10' \
    --baseline_weight 10.0 \
    --export_root opt_runs_icews14_20251110

# Interrupted early
python3 optimize_tkg.py \
    --ref_tkg_dir /nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/WIKI \
    --budget 100 \
    --export_root opt_runs_wiki_20250916
    --skip_labeling

# Interrupted early
python3 optimize_tkg.py \
    --ref_tkg_dir /nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/YAGO \
    --budget 100 \
    --export_root opt_runs_yago_20250916
    --skip_labeling

Use --weights to emphasize certain metrics, e.g.:
--weights '{"n_ents":2,"n_rels":2,"n_quads":3,"avg_degree":3,"deg_p25":1,"deg_p75":1,"n_tws":1,"n_unique_triples":1}'
"""
