import numpy as np
import pandas as pd
import scipy

from collections import defaultdict
from copy import deepcopy
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm

import argparse
import importlib.util
import inspect
import json
import os
import random
import re
import sys
import uuid

from patterns import instantiate_patterns_from_df
from pattern_library import load_or_generate_patterns
from temporalpattern import TemporalPattern
from utils import is_subpattern
from utils_pattern_graphing import render_patterns_file


# Debugging utilities
_lab_re = re.compile(r"^(?P<pid>\d+)_(?P<role>c|a(?P<ai>\d+))_(?P<bind>.*)$")

_PH_RE = re.compile(r'^e\d+$')


# Helpers to sample from empirical TKG entity and relation frequencies
def _rng_from_seed(seed):
    # Accept int/None or an existing RandomState
    return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

def _read_edgelist_any(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, engine="python")
    if df.shape[1] < 4:
        raise ValueError(f"File {path} must have at least 4 columns.")
    df = df.iloc[:, :4].copy()
    df.columns = ["head", "rel", "tail", "t"]
    for c in ["head", "rel", "tail", "t"]:
        df[c] = pd.to_numeric(df[c], errors="raise", downcast="integer")
    return df

def _load_tkg_dir(edgelist_dir: str) -> pd.DataFrame:
    parts = []
    for name in ["train.txt", "valid.txt", "test.txt"]:
        p = os.path.join(edgelist_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
        parts.append(_read_edgelist_any(p))
    df = pd.concat(parts, ignore_index=True)
    for c in ["head", "rel", "tail", "t"]:
        df[c] = df[c].astype(np.int64)
    return df

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

def _make_empirical_weight_sampler(probs: np.ndarray):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    def sampler(n: int, seed=None):
        rng = _rng_from_seed(seed)
        idx = rng.choice(len(probs), size=int(n), replace=True, p=probs)
        # Use the chosen base probs as weights; tiny jitter reduces ties deterministically
        w = probs[idx] + 1e-12 * rng.rand(int(n))
        return w
    return sampler

def _jsonify(obj):
    """Make any object JSON-serializable (best-effort)."""
    # Numpy scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # Basic containers
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    # Callables -> readable description
    if callable(obj):
        name = getattr(obj, "__name__", "<lambda>")
        try:
            src = inspect.getsource(obj).strip()
        except Exception:
            src = repr(obj)
        return {"__callable__": name, "repr": src}
    # Anything else: try JSON, fallback to str
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

def dump_config_snapshot(export_dir: str, config: dict):
    """Write the effective config to disk in a robust, JSON-safe way."""
    snap_path = os.path.join(export_dir, "config_used.json")
    with open(snap_path, "w") as f:
        json.dump(_jsonify(config), f, indent=2)

# Cache so we only read each TKG directory once
_EMP_CACHE = {}  # path -> {"ents": sampler, "rels": sampler}

def _resolve_pat_sampler(config_value, fallback_path_key: str, kind: str, config: dict):
    """
    Resolve a sampler for 'ents' or 'rels'.
    Priority:
      1) If config_value is callable -> use it.
      2) If config_value is a string -> treat as TKG dir path.
      3) Else, if config[fallback_path_key] (e.g., 'pat_distr_ents_from_tkg') is set -> use that path.
      4) Else -> return None (uniform).
    Returns a function (n:int, seed) -> weights vector OR None.
    """
    # Callable stays callable
    if callable(config_value):
        return config_value

    # Allow passing a path directly in pat_distr_* (string)
    path = None
    if isinstance(config_value, str):
        path = config_value.strip()

    # Also allow explicit *_from_tkg keys
    if path is None:
        path = config.get(fallback_path_key, None)

    if path is None:
        return None

    # Load/cached samplers
    if path not in _EMP_CACHE:
        df = _load_tkg_dir(path)
        ent_probs = _empirical_probs_entities(df)
        rel_probs = _empirical_probs_relations(df)
        _EMP_CACHE[path] = {
            "ents": _make_empirical_weight_sampler(ent_probs),
            "rels": _make_empirical_weight_sampler(rel_probs),
        }
    return _EMP_CACHE[path][kind]


# Helpers to create entity, relation, and time window indices

def create_entity2id(config, seed) -> pd.DataFrame:
    sampler = _resolve_pat_sampler(
        config.get('pat_distr_ents', None),
        fallback_path_key='pat_distr_ents_from_tkg',
        kind='ents',
        config=config
    )
    if sampler is not None:
        wts = sampler(config['n_ents'], seed)
    else:
        wts = [1] * config['n_ents']
    return pd.DataFrame({
        'name': range(config['n_ents']),
        'id': range(config['n_ents']),
        'wt': wts,
    })

def create_relation2id(config, seed) -> pd.DataFrame:
    sampler = _resolve_pat_sampler(
        config.get('pat_distr_rels', None),
        fallback_path_key='pat_distr_rels_from_tkg',
        kind='rels',
        config=config
    )
    if sampler is not None:
        wts = sampler(config['n_rels'], seed)
    else:
        wts = [1] * config['n_rels']
    return pd.DataFrame({
        'name': range(config['n_rels']),
        'id': range(config['n_rels']),
        'wt': wts,
    })

def create_time2id(config) -> pd.DataFrame:
    return pd.DataFrame({
        'name': range(config['n_tws']),
        'id': range(config['n_tws']),
    })

def create_pattern2id(patterns: 'List[TemporalPattern]') -> pd.DataFrame:
    return pd.DataFrame({
        'pattern': [pat.__label__() for pat in patterns],
        'n_hops': [pat.n_hops for pat in patterns],
        'id': range(len(patterns)),
    })

def create_edgelist() -> pd.DataFrame:
    return pd.DataFrame({
        'head': [],
        'rel': [],
        'tail': [],
        't': [],
        'wt': [],
        'pattern': [],
    })

def add_new_pattern(
    config: 'Dict[str,]',
    patterns: 'List[TemporalPattern]',
    pattern_quadruples: 'List[Tuple]',
    pattern_creation_func,
    time_lag,
    entity2id: pd.DataFrame,
    relation2id: pd.DataFrame,
) -> None:
    """ Add new pattern as long as it is not a subpattern of any existing pattern.
    Note, adds patterns to patterns and pattern_quadruples in place.
    """
    new_pat = False
    retry = 0
    while (not new_pat) | (retry < config['max_retries']):
        pat = pattern_creation_func(entity2id, relation2id, time_lag)
        # Find out if quadruple is a subpattern
        quad = pat.__quadruples__()
        if not is_subpattern(quad, pattern_quadruples):
            patterns.append(pat)
            pattern_quadruples.append(quad)
            new_pat = True
        retry += 1

def instantiate_antecedent_entities(
    pattern: 'TemporalPattern',
    entity2id: pd.DataFrame,
    seed: int,
) -> None:
    """ Helper to instantiate pattern antecedent entities based on per-entity probs
    """
    antecedent = pattern.antecedent
    # Get unique placeholder entities
    placeholder_entities = set([
        el for tup in antecedent for el in tup if isinstance(el, str)
    ])
    # Sample without replacement from entity2id
    sampled_entities = entity2id.sample(
        len(placeholder_entities), weights=entity2id['wt'],
        replace=False, random_state=seed,
    )['id'].tolist()
    placeholder2entity = {
        placeholder: entity for placeholder, entity in zip(
            placeholder_entities, sampled_entities
        )
    }
    # Instantiate antecedent
    instantiated_antecedent = []
    for ant in antecedent:
        instantiated_antecedent.append((
            placeholder2entity.get(ant[0], ant[0]),
            ant[1],
            placeholder2entity.get(ant[2], ant[2]),
        ))
    return instantiated_antecedent

def get_satisfying_idxs(
    pattern: TemporalPattern, edgelist: pd.DataFrame, prev_t: int = -1,
    satisfying_idxs: 'List[int]' = [], prev_idxs = [],
) -> 'List[int]':
    """ Get ids of triples that satisfy pattern in edgelist
    """
    if pattern.n_hops == 0:
        # Final check for consequence
        cons = pattern.consequence
        time_lag = pattern.time_lags[0]
        triples = edgelist[
            (edgelist['head'] == cons[0]) &
            (edgelist['rel'] == cons[1]) &
            (edgelist['tail'] == cons[2]) &
            (edgelist['t'] >= prev_t+time_lag[0] if prev_t != -1 else edgelist['t'] > -np.inf) &
            (edgelist['t'] <= prev_t+time_lag[1] if prev_t != -1 else edgelist['t'] < np.inf)
        ]
        if triples.shape[0] > 0:
            # If some valid consequence is found, return all its satisfying
            # triples' locations
            return list(set(satisfying_idxs+prev_idxs+triples.index.tolist()))
    else:
        # Otherwise, check first antecedent recursively
        ante = pattern.antecedent[0]
        time_lag = pattern.time_lags[0] if prev_t != -1 else None
        triples = edgelist[
            (edgelist['head'] == ante[0]) &
            (edgelist['rel'] == ante[1]) &
            (edgelist['tail'] == ante[2]) &
            (edgelist['t'] >= prev_t+time_lag[0] if prev_t != -1 else edgelist['t'] > -np.inf) &
            (edgelist['t'] <= prev_t+time_lag[1] if prev_t != -1 else edgelist['t'] < np.inf)
        ]
        new_idxs = triples.index.tolist()
        new_satisfying_idxs = []
        for idx in new_idxs:
            new_pattern = TemporalPattern(
                antecedent=list(pattern.antecedent)[1:],
                consequence=pattern.consequence,
                time_lags=list(pattern.time_lags)[1:] if prev_t != -1 else list(pattern.time_lags),
                n_hops=pattern.n_hops-1,
            )
            new_satisfying_idxs.extend(get_satisfying_idxs(
                new_pattern, edgelist,
                prev_t=edgelist.loc[idx]['t'],
                satisfying_idxs=satisfying_idxs,
                prev_idxs=list(set(prev_idxs+[idx])),
            ))
        satisfying_idxs.extend(new_satisfying_idxs)
    return list(set(satisfying_idxs))

def _parse_bind(bind_str):
    # Helper to parse a binding label into a mapping from placeholders to entities
    # Example: 'e1=35|e2=27|e3=10' -> {'e1':35, 'e2':27, 'e3':10}
    out = {}
    if not bind_str:
        return out
    for part in bind_str.split("|"):
        if not part:  # allow trailing separators just in case
            continue
        k, v = part.split("=")
        out[k] = int(v)
    return out

def _validate_labels(edgelist, pattern2id_df, max_bad=50):
    # Re-interpret every label against the pattern definition and the row it sits on.
    # Build a quick lookup: pid -> (ants, cons)
    pid2pat = {}
    for pid, label in zip(pattern2id_df['id'], pattern2id_df['pattern']):
        tp = TemporalPattern()
        tp.from_label(label)
        pid2pat[int(pid)] = (list(tp.antecedent), tp.consequence)

    bad = 0
    for idx, row in edgelist.iterrows():
        labs = row['pattern']
        if not isinstance(labs, list):
            continue
        for lab in labs:
            if isinstance(lab, int):   # e.g., -1 random marker
                continue
            m = _lab_re.match(str(lab))
            if not m:
                continue
            pid = int(m.group('pid'))
            role = m.group('role')
            ante_i = m.group('ai')
            bind = _parse_bind(m.group('bind'))

            if pid not in pid2pat:
                dbg(f"[VAL:unknown-pid] idx={idx} lab={lab}")
                bad += 1
                if bad >= max_bad: return

            ants, cons = pid2pat[pid]
            if role == 'c':
                ch, cr, ct = cons
                eh = bind.get(ch, ch) if isinstance(ch, str) else ch
                et = bind.get(ct, ct) if isinstance(ct, str) else ct
                if not (int(eh) == int(row['head']) and int(cr) == int(row['rel']) and int(et) == int(row['tail'])):
                    dbg(f"[VAL:cons-mismatch] idx={idx} lab={lab} "
                          f"row=({int(row['head'])},{int(row['rel'])},{int(row['tail'])}) "
                          f"exp=({int(eh)},{int(cr)},{int(et)})")
                    bad += 1
                    if bad >= max_bad: return
            else:
                ai = int(ante_i)
                ah, ar, at = ants[ai]
                eh = bind.get(ah, ah) if isinstance(ah, str) else ah
                et = bind.get(at, at) if isinstance(at, str) else at
                if not (int(eh) == int(row['head']) and int(ar) == int(row['rel']) and int(et) == int(row['tail'])):
                    dbg(f"[VAL:ante-mismatch] idx={idx} lab={lab} "
                          f"row=({int(row['head'])},{int(row['rel'])},{int(row['tail'])}) "
                          f"exp=({int(eh)},{int(ar)},{int(et)})")
                    bad += 1
                    if bad >= max_bad: return

def run(config: 'Dict[str,]', run_id: int):
    """ Create TKGs according to configuration from config.py file
    """
    # Counts across this run
    pattern_counts_raw = defaultdict(int)      # counts *all* bindings yielded (before your dedupe)
    pattern_counts_unique = defaultdict(int)   # counts unique bindings after your dedupe

    # Debugging output limit
    DEBUG = bool(config.get('debug', False))
    mis_budget = defaultdict(int)
    MIS_LIMIT = int(config.get('debug_mis_limit_per_pid', 3))

    # Skip backtracking to instantiate consequences if there's no random wiring
    FAST_FORCE_CONSEQUENCE = bool(config.get('fast_force_consequence', True))
    NO_RANDOM_WIRING = (config.get('rnd_avg_density_distr') is None and float(config.get('rnd_avg_density', 0.0)) <= 0.0)

    def dbg(*args, **kwargs):
        if DEBUG:
            print(*args, **kwargs)

    def _is_ph(x):
        # Helper to identify placeholders in pattern tuples
        return isinstance(x, (str, np.str_)) and _PH_RE.match(str(x)) is not None
    
    def _norm_ph(x):
        # Normalize placeholder to plain python str 'eK'
        return str(x) if _is_ph(x) else x
    
    def _seed(mapping, key, value):
        """ Write placeholder -> value if consistent. Return True if OK, False if conflict.
        - Normalizes the key (so numpy.str_ etc. won't trip us)
        - No-ops for concrete ints (non-placeholders)
        """
        key = _norm_ph(key)
        if not _is_ph(key):
            return True  # nothing to bind for concrete values

        v = int(value)
        if key in mapping:
            return int(mapping[key]) == v  # conflict if different
        mapping[key] = v
        return True

    def _all_diff_ok(mapping):
        # Helper to enforce inequality across all currently bound placeholders
        vals = list(mapping.values())
        return len(vals) == len(set(vals))
    
    def _consistent_extend(mapping, ant, row):
        """
        Try to extend mapping with bindings from antecedent triple `ant`
        matched by a `row` (head, rel, tail, t). Return new mapping if
        consistent, else None.
        """
        new_map = dict(mapping)
        h, r, ta = _norm_ph(ant[0]), ant[1], _norm_ph(ant[2])

        # relation must match exactly
        if r != row.rel:
            return None

        # head
        if _is_ph(h):
            if not _seed(new_map, h, row.head):
                return None
        else:
            if h != row.head:
                return None

        # tail
        if _is_ph(ta):
            if not _seed(new_map, ta, row.tail):
                return None
        else:
            if ta != row.tail:
                return None

        # all-different safeguard on whatever we've bound so far
        if not _all_diff_ok(new_map):
            return None

        return new_map

    def _ante_cache_key(ant, t_curr, lag):
        # Helper to get key for antecedent cache (time-lag window and instantiated elements)
        h, r, ta = _norm_ph(ant[0]), ant[1], _norm_ph(ant[2])
        return (
            h if not _is_ph(h) else None,
            r,
            ta if not _is_ph(ta) else None,
            t_curr,
            lag
        )

    def _candidate_rows_for_ant(edgelist, ant, t_curr, lag):
        """
        Get candidate rows for antecedent ant given current time anchor t_curr and
        lag = (min, max) interpreted as: t_ant ∈ [t_curr - max, t_curr - min]. Only
        filter by concrete head/tail (if available) and by relation and time.
        """
        h, r, ta = _norm_ph(ant[0]), ant[1], _norm_ph(ant[2])
        key = _ante_cache_key((h, r, ta), t_curr, tuple(lag))
        cached = _cand_cache.get(key)
        if cached is not None:
            return cached

        min_lag, max_lag = lag
        # Doing retrospective filtering, so t_min and t_max use max_lag and min_lag, respectively
        t_min = t_curr - max_lag
        t_max = t_curr - min_lag
        mask = (
            (edgelist['rel'] == r) &
            (edgelist['t'] >= t_min) &
            (edgelist['t'] <= t_max)
        )
        if not _is_ph(h):
            # Bitwise and for instantiated head
            mask &= (edgelist['head'] == h)
        if not _is_ph(ta):
            # Bitwise and for instantiated tail
            mask &= (edgelist['tail'] == ta)
        # Keep only needed columns for speed in iterrows/itertuples
        cand = edgelist.loc[mask, ['head', 'rel', 'tail', 't']]
        _cand_cache[key] = cand
        return cand
    
    def _match_antecedents_rev(edgelist, ants_rev, lags_rev, t_anchors, mapping):
        """
        Generator backtracking over antecedents in reverse (most recent to oldest).
        - ants_rev, lags_rev are lists in reverse order relative to pattern definition
        - t_anchors is a list of allowed anchor times for this antecedent step
        - mapping is current placeholder to entity assignment
        """
        if not ants_rev:
            yield mapping
            return

        ant = ants_rev[0]
        lag = lags_rev[0]
        rest_ants = ants_rev[1:]
        rest_lags = lags_rev[1:]

        # For each current anchor, find candidates for this antecedent
        for t_anchor in t_anchors:
            # Filter by relation and time lags (and instantiated entities, if available)
            cand = _candidate_rows_for_ant(edgelist, ant, t_anchor, lag)
            if cand.empty:
                continue
            # Iterate over candidate quadruples
            for row in cand.itertuples(index=False):
                # Check for a consistent mapping from placeholders to entities
                new_map = _consistent_extend(mapping, ant, row)
                if new_map is None:
                    continue
                # Recurse: Previous antecedents must anchor to this row.t
                yield from _match_antecedents_rev(
                    edgelist, rest_ants, rest_lags, [row.t], new_map
                )

    def _pattern_placeholders(ants, cons):
        # Sorted list of unique placeholder names used anywhere in the pattern.
        ordered = []
        def add(x):
            x = _norm_ph(x)
            if _is_ph(x) and x not in ordered:
                ordered.append(x)
        for h, _, ta in ants:   # antecedents in their original order
            add(h); add(ta)
        ch, _, cta = cons       # then consequence
        add(ch); add(cta)
        return ordered
    
    def _binding_to_str(mapping, placeholders):
        # Deterministic binding string like 'e1=35|e2=27', only placeholders present in mapping
        parts = []
        for k in placeholders:
            if k in mapping:
                v = int(mapping[k])
                parts.append(f"{k}={v}")
        return "|".join(parts)
    
    def _match_antecedents_rev_with_paths(edgelist, ants, lags, t_anchors, step, mapping, path):
        """
        Backtrack over antecedents from newest to oldest, carrying *original* indices.
        - ants: list of antecedents in original order [a0, a1, ..., a_{n-1}]
        - lags: list of lags in original order [lag(a0->a1), ..., lag(a_{n-1}->cons)]
        - step: how many antecedents we have matched so far from the end (0..n)
        Yields (mapping, path) where path = list[(orig_idx, edgelist_row_index)].
        """
        n = len(ants)
        if step == n:
            yield mapping, path
            return

        # The 'step'-th antecedent from the END has original index:
        orig_idx = n - 1 - step
        ant = ants[orig_idx]
        lag = lags[orig_idx]  # lag from this antecedent to the *next* event in time

        for t_anchor in t_anchors:
            cand = _candidate_rows_for_ant(edgelist, ant, t_anchor, lag)
            if cand.empty:
                continue
            for row in cand.itertuples(index=True, name='E'):
                new_map = _consistent_extend(mapping, ant, row)
                if new_map is None:
                    continue
                new_path = path + [(orig_idx, int(row.Index))]
                # Anchor moves back to this antecedent's time
                yield from _match_antecedents_rev_with_paths(
                    edgelist=edgelist,
                    ants=ants,
                    lags=lags,
                    t_anchors=[int(row.t)],
                    step=step + 1,
                    mapping=new_map,
                    path=new_path
                )
    
    def _append_label(idx, label):
        # Helper to append a string label into edgelist.loc[idx, 'pattern']
        cell = edgelist.at[idx, 'pattern']
        if not isinstance(cell, list):
            if pd.isna(cell):
                cell = []
            elif isinstance(cell, (tuple, set)):
                cell = list(cell)
            else:
                cell = [cell]
        if label not in cell:
            cell.append(label)
            edgelist.at[idx, 'pattern'] = cell
    
    def _flatten_labels(series):
        # Helper to aggregate pattern labels
        out = []
        for cell in series:
            if isinstance(cell, list):
                out.extend(cell)
            elif pd.isna(cell):
                continue
            else:
                out.append(str(cell))
        def _key(v):
            if isinstance(v, (int, np.integer)):
                return (0, int(v))
            return (1, str(v))
        return sorted(set(out), key=_key)

    # Ensure each row has its own list object in 'pattern'
    def _uniq_list_cell(x):
        if isinstance(x, list):
            return list(x)
        if pd.isna(x):
            return []
        return [x]
    
    # Get run-specific seed
    if config['seed'] is not None:
        seed = (run_id * 10_000) + config['seed']
    else:
        seed = None
    rnd_state = np.random.RandomState(seed)
    
    # Create ids for entities, relations, and time windows
    entity2id = create_entity2id(config, rnd_state)
    relation2id = create_relation2id(config, rnd_state)
    time2id = create_time2id(config)

    # Instantiate patterns
    # Get all valid 1-, 2-, and 3-hop patterns or generate them
    pattern_dfs = {
        1: load_or_generate_patterns(
            1, "patterns/",
            config['require_unique_triples'],
            config['prohibit_selfconnections'],
            config['prohibit_new_consequence_relations'],
            config['require_sequential_rule'],
        ),
        2: load_or_generate_patterns(
            2, "patterns/",
            config['require_unique_triples'],
            config['prohibit_selfconnections'],
            config['prohibit_new_consequence_relations'],
            config['require_sequential_rule'],
        ),
        3: load_or_generate_patterns(
            3, "patterns/",
            config['require_unique_triples'],
            config['prohibit_selfconnections'],
            config['prohibit_new_consequence_relations'],
            config['require_sequential_rule'],
        ),
    }
    # Start from 3-hop patterns, then 2-hop, then 1-hop
    # Prohibit any new patterns from being contained (antecedent and consequence) in the antecedent of
    # an existing larger pattern or being identical to an already chosen same-sized pattern
    patterns = []
    pattern_quadruples = []

    # 3-hop patterns
    patterns_3hop = instantiate_patterns_from_df(
        pattern_df=pattern_dfs[3],
        relation2id=relation2id,
        n=config['n_3_hop'],
        time_lags=config['time_lag_3_hop'],
        seed=rnd_state,
        pattern_quadruples=pattern_quadruples,
        max_retries=config['max_retries'],
    )
    patterns += patterns_3hop
    pattern_quadruples += [pat.__quadruples__() for pat in patterns_3hop]

    # 2-hop patterns
    patterns_2hop = instantiate_patterns_from_df(
        pattern_df=pattern_dfs[2],
        relation2id=relation2id,
        n=config['n_2_hop'],
        time_lags=config['time_lag_2_hop'],
        seed=rnd_state,
        pattern_quadruples=pattern_quadruples,
        max_retries=config['max_retries'],
    )
    patterns += patterns_2hop
    pattern_quadruples += [pat.__quadruples__() for pat in patterns_2hop]

    # 1-hop patterns
    patterns_1hop = instantiate_patterns_from_df(
        pattern_df=pattern_dfs[1],
        relation2id=relation2id,
        n=config['n_1_hop'],
        time_lags=config['time_lag_1_hop'],
        seed=rnd_state,
        pattern_quadruples=pattern_quadruples,
        max_retries=config['max_retries'],
    )
    patterns += patterns_1hop
    pattern_quadruples += [pat.__quadruples__() for pat in patterns_1hop]

    # Create dataframe of pattern ids
    pattern2id = create_pattern2id(patterns)
    dbg('pattern2id:', pattern2id)

    # Apply patterns
    edgelist = create_edgelist()
    pbar_tws = tqdm(range(config['n_tws']))
    for t in pbar_tws:
        pbar_tws.set_description(f'Time window: {t}')

        # First, randomly wire entities
        dfs_i = []
        if (config.get('rnd_avg_density_distr') is not None) or (float(config.get('rnd_avg_density', 0.0)) > 0.0):
            pbar_ent = tqdm(entity2id['id'])
            for ent_id in pbar_ent:
                pbar_ent.set_description(f'Entity: {ent_id}')
                # Sample entities to use as tails
                if config['rnd_avg_density_distr']:
                    dens = config['rnd_avg_density_distr'](rnd_state)
                else:
                    dens = config['rnd_avg_density']
                # Handle random density specifications in the range (0,1)
                if (dens > 0) and (dens < 1):
                    rnd = random.random()
                    # With specified probability, sample one random edge
                    if rnd < dens:
                        dens = 1
                    # Otherwise, no random edge is sampled
                    else:
                        dens = 0
                else:
                    dens = int(dens)
                if dens == 0:
                    # No random edge
                    continue
                tails = entity2id['id'].sample(dens, replace=True, random_state=rnd_state)
                # Sample relations to connect them
                rels = relation2id['id'].sample(dens, replace=True, random_state=rnd_state)
                df_i = pd.DataFrame({
                    'head': [ent_id]*dens,
                    'rel': rels.values,
                    'tail': tails.values,
                    't': [t]*dens,
                    'wt': [1]*dens,
                    'pattern': [[-1] for _ in range(dens)],  # -1 indicates a randomly wired edge
                })
                dfs_i.append(df_i)
        edgelist = pd.concat([edgelist]+dfs_i)
        if edgelist.shape[0] and edgelist['head'].dtype != np.int64:
            edgelist[['head','rel','tail','t']] = edgelist[['head','rel','tail','t']].astype('int64')

        # Iterate over patterns
        # heads, rels, tails, pats = [], [], [], []  # TODO Remove
        # cons_ts stores per-row consequence times
        heads, rels, tails, pats, cons_ts = [], [], [], [], []
        # cons_pids = []  # Store per-pattern relevant edge indices
        dfs_pat = []
        for label, pattern_id in zip(pattern2id['pattern'], pattern2id['id']):
            # Cache for pattern instantiation
            # Cache: {(antecedent_tuple, t_anchor): DataFrame[['head','rel','tail','t']]}
            _cand_cache = {}

            # Instantiate pattern from label
            pattern = TemporalPattern()
            pattern.from_label(label)
            dbg('pattern:', pattern.__label__())

            # Artificially create valid patterns 
            # Get the number of times to instantiate this pattern in this time window
            n_instantiations = 0
            if config['n_hops2n_force_distr'][pattern.n_hops]:
                # Sample from distribution, if not None
                n_instantiations = int(config['n_hops2n_force_distr'][pattern.n_hops](rnd_state))
            else:
                # Otherwise, sample from binomial distribution defined by p_force and n_force
                n_instantiations = int(scipy.stats.binom.rvs(
                    n=config['n_hops2n_force'][pattern.n_hops],
                    p=config['n_hops2p_force'][pattern.n_hops],
                    size=1, random_state=rnd_state,
                )[0])

            # rnd = rnd_state.rand()
            # if rnd < config['n_hops2p_force'][pattern.n_hops]:
            #     # Create the antecedent in this and subsequent windows
            #     # Track time window of current antecedent as we create them
            #     t_i = int(t)
            #     # Instantiate pattern based on entity2id weights and wrt placeholders
            #     instantiated_antecedent = instantiate_antecedent_entities(
            #         pattern, entity2id, rnd_state,
            #     )
            #     heads_pat, rels_pat, tails_pat, ts_pat = [], [], [], []
            #     for antecedent, time_lag in zip(instantiated_antecedent, pattern.time_lags):
            #         heads_pat.append(antecedent[0])
            #         rels_pat.append(antecedent[1])
            #         tails_pat.append(antecedent[2])
            #         ts_pat.append(t_i)
            #         # Increment t_i according to time_lag min and max
            #         t_i += random.randint(time_lag[0], time_lag[1])
            #     df_pat = pd.DataFrame({
            #         'head': heads_pat,
            #         'rel': rels_pat,
            #         'tail': tails_pat,
            #         't': ts_pat,
            #         'wt': [1]*len(heads_pat),
            #         'pattern': [[] for _ in range(len(heads_pat))],
            #     })
            #     dfs_pat.append(df_pat)
            
            #     if rnd_state.rand() < config['p_skip_consequence']:
            #         # Skip the consequence even though antecedents may be satisfied
            #         continue

            # TODO Replaced above
            for _ in range(n_instantiations):
                # Create the antecedent in this and subsequent windows
                # Track time window of current antecedent as we create them
                t_i = int(t)
                # Instantiate pattern based on entity2id weights and wrt placeholders
                instantiated_antecedent = instantiate_antecedent_entities(
                    pattern, entity2id, rnd_state,
                )
                heads_pat, rels_pat, tails_pat, ts_pat = [], [], [], []
                for antecedent, time_lag in zip(instantiated_antecedent, pattern.time_lags):
                    heads_pat.append(antecedent[0])
                    rels_pat.append(antecedent[1])
                    tails_pat.append(antecedent[2])
                    ts_pat.append(t_i)
                    # Increment t_i according to time_lag min and max
                    t_i += random.randint(time_lag[0], time_lag[1])
                df_pat = pd.DataFrame({
                    'head': heads_pat,
                    'rel': rels_pat,
                    'tail': tails_pat,
                    't': ts_pat,
                    'wt': [1]*len(heads_pat),
                    'pattern': [[] for _ in range(len(heads_pat))],
                })
                dfs_pat.append(df_pat)
            
                if rnd_state.rand() < config['p_skip_consequence']:
                    # Skip the consequence even though antecedents may be satisfied
                    continue

                # FAST-PATH: emit the consequence now (no backtracking) when there's no random wiring
                if FAST_FORCE_CONSEQUENCE and NO_RANDOM_WIRING:
                    # Build placeholder -> entity map from the instantiated antecedents
                    ph_map = {}
                    for (tpl, inst) in zip(pattern.antecedent, instantiated_antecedent):
                        ah, ar, at = tpl
                        ih, ir, it = inst
                        if isinstance(ah, str):
                            ph_map.setdefault(str(ah), int(ih))
                        if isinstance(at, str):
                            ph_map.setdefault(str(at), int(it))

                    cons_h, cons_r, cons_ta = pattern.consequence
                    cons_time = int(t_i)  # after the last lag, this is the consequence time

                    # Resolve placeholders in the consequence
                    # If a placeholder wasn't in the antecedents (shouldn't happen given constraints), skip.
                    if isinstance(cons_h, str):
                        if cons_h not in ph_map: 
                            continue
                        head_val = int(ph_map[cons_h])
                    else:
                        head_val = int(cons_h)

                    if isinstance(cons_ta, str):
                        if cons_ta not in ph_map:
                            continue
                        tail_val = int(ph_map[cons_ta])
                    else:
                        tail_val = int(cons_ta)

                    # Same-placeholder guard: (eK, r, eK) => head must equal tail
                    if isinstance(cons_h, str) and isinstance(cons_ta, str) and cons_h == cons_ta:
                        tail_val = head_val

                    # Respect p_skip_consequence and horizon
                    if cons_time < int(config['n_tws']) and rnd_state.rand() >= float(config.get('p_skip_consequence', 0.0)):
                        heads.append(head_val)
                        rels.append(int(cons_r))
                        tails.append(tail_val)
                        cons_ts.append(cons_time)
                        pats.append([pattern_id])

                        # cons_pids.append(pattern_id)

            # TODO Skip the matcher when the fast path is active
            # If we emitted consequences via the fast-path, skip the expensive backtracking for this pattern
            if FAST_FORCE_CONSEQUENCE and NO_RANDOM_WIRING:
                continue

            # Apply valid patterns
            dbg('Applying ...')
            
            # Prepare reverse lists for backtracking
            ants_rev = list(pattern.antecedent[::-1])
            lags_rev = list(pattern.time_lags[::-1])

            # Lazily iterate bindings and dedupe bindings
            seen_bind_keys = set()

            for bmap in _match_antecedents_rev(
                edgelist=edgelist,
                ants_rev=ants_rev,
                lags_rev=lags_rev,
                t_anchors=[t],
                mapping={}
            ):
                # Count every binding the matcher yields
                pattern_counts_raw[pattern_id] += 1

                # Canonical key to dedupe identical placeholder assignments
                key = tuple(sorted(bmap.items()))
                if key in seen_bind_keys:
                    continue
                seen_bind_keys.add(key)

                # Count unique bindings for this (pattern, t)
                pattern_counts_unique[pattern_id] += 1

                cons_h, cons_r, cons_ta = pattern.consequence

                # Must be bound if they’re placeholders
                if (_is_ph(cons_h) and cons_h not in bmap) or (_is_ph(cons_ta) and cons_ta not in bmap):
                    continue

                # Same-placeholder consequence guard: (e1, r, e1) -> must produce head==tail
                if _is_ph(cons_h) and _is_ph(cons_ta) and cons_h == cons_ta:
                    if int(bmap[cons_h]) != int(bmap[cons_ta]):
                        continue
                
                head_val = bmap[cons_h] if _is_ph(cons_h) else cons_h
                tail_val = bmap[cons_ta] if _is_ph(cons_ta) else cons_ta

                # Optional: enforce all-different *including* consequence positions
                tmp = dict(bmap)
                if not _seed(tmp, cons_h, head_val):  # no-op for non-placeholders
                    continue
                if not _seed(tmp, cons_ta, tail_val):
                    continue
                if not _all_diff_ok(tmp):
                    continue
    
                dbg(f"[APPLY] t={t} pid={pattern_id} cons={pattern.consequence} bind={dict(bmap)} -> ({int(head_val)}, {int(cons_r)}, {int(tail_val)})")

                heads.append(int(head_val))
                rels.append(int(cons_r))
                tails.append(int(tail_val))
                cons_ts.append(int(t))
                pats.append([pattern_id])

                # cons_pids.append(pattern_id)

        # Add new forced patterns to edgelist
        edgelist = pd.concat([edgelist]+dfs_pat)
        # Add all new consequences to edgelist
        df_con = pd.DataFrame({
            'head': heads,
            'rel': rels,
            'tail': tails,
            # 't': [t]*len(heads),
            't': cons_ts,
            'wt': [1]*len(heads),
            'pattern': [[] for _ in range(len(heads))],
            # 'con_pid': cons_pids,
            # Labeling them now is okay, but because the artificial creation is
            # forward-looking, some patterns may extend beyond our range of time
            # windows, making them invalid in the span of time windows we care
            # about. Instead, we label all edges for patterns later.
            # 'pattern': pats,
        })
        edgelist = pd.concat([
            edgelist,
            df_con,
        ])
    
    # Reindex edgelist
    edgelist = edgelist.reset_index(drop=True)

    dup_cnt = edgelist['pattern'].apply(id).duplicated(keep=False).sum()
    if dup_cnt:
        dbg(f"[ALIAS] Before unique-ifying: {dup_cnt} rows still share 'pattern' list objects (unexpected).")

    edgelist['pattern'] = edgelist['pattern'].apply(_uniq_list_cell)

    # Optional: quick diagnostic – how many rows still share list objects?
    dup_cnt = edgelist['pattern'].apply(id).duplicated(keep=False).sum()
    if dup_cnt:
        dbg(f"[ALIAS] After unique-ifying: {dup_cnt} rows still share 'pattern' list objects (unexpected).")

    # Post-creation, placeholder-aware labeling with roles and bindings
    if not config.get('skip_labeling', False):
        dbg('Labelling ...')
        # Buffer labels and write once at the end
        # labels_buffer = defaultdict(set)
        for label, pattern_id in zip(pattern2id['pattern'], pattern2id['id']):
            _cand_cache = {}

            pattern = TemporalPattern()
            pattern.from_label(label)
            dbg('Labelling pattern:', pattern.__label__())
            
            # ants = list(pattern.antecedent)
            raw_ants = list(pattern.antecedent)
            ants = [(_norm_ph(h), r, _norm_ph(ta)) for (h, r, ta) in raw_ants]

            # cons_h, cons_r, cons_ta = pattern.consequence
            ch, cr, cta = pattern.consequence
            cons_h, cons_r, cons_ta = _norm_ph(ch), cr, _norm_ph(cta)

            lags = list(pattern.time_lags)

            # Reverse for backtracking and carry original antecedent indices
            ants_rev = ants[::-1]
            lags_rev = lags[::-1]
            idxs_rev = list(range(len(ants)-1, -1, -1))

            # All placeholders used anywhere in this pattern
            placeholders = _pattern_placeholders(ants, (cons_h, cons_r, cons_ta))

            # Candidate consequence edges by relation and any concrete head/tail
            mask = (edgelist['rel'] == cons_r)
            if not _is_ph(cons_h):
                mask &= (edgelist['head'] == cons_h)
            if not _is_ph(cons_ta):
                mask &= (edgelist['tail'] == cons_ta)

            same_ph_cons = (_is_ph(cons_h) and _is_ph(cons_ta) and cons_h == cons_ta)

            cand_df = edgelist.loc[mask, ['head', 'tail', 't']]
            if same_ph_cons:
                cand_df = cand_df[cand_df['head'] == cand_df['tail']]

            # Group by exact consequence triple+time; do the search once per distinct group
            for (h_val, ta_val, t_anchor), group_index in cand_df.groupby(['head','tail','t'], sort=False).groups.items():
                idxs = list(group_index)   # all row indices with this exact (h, r, t, tail)

                # Seed placeholders from the consequence itself
                init_map = {}
                if same_ph_cons:
                    # head==tail already guaranteed by prefilter
                    if not _seed(init_map, cons_h, h_val):   # cons_h == cons_ta here
                        continue
                else:
                    if _is_ph(cons_h) and not _seed(init_map, cons_h, h_val):
                        continue
                    if _is_ph(cons_ta) and not _seed(init_map, cons_ta, ta_val):
                        continue

                if not _all_diff_ok(init_map):
                    continue

                # 0-hop pattern: just label all rows in this group
                if len(ants) == 0:
                    bind_str = _binding_to_str(init_map, placeholders)
                    lab_c = f"{pattern_id}_c_{bind_str}"
                    for idx in idxs:
                        _append_label(idx, lab_c)
                    continue

                seen_role_binding = set()

                # Run the backtracker ONCE for this (h_val, ta_val, t_anchor)
                for mapping, path in _match_antecedents_rev_with_paths(
                    edgelist=edgelist, ants=ants, lags=lags,
                    t_anchors=[int(t_anchor)], step=0, mapping=init_map, path=[],
                ):
                    # Guard: mapping must reproduce this consequence (should pass now that we seeded)
                    exp_h = mapping[cons_h] if _is_ph(cons_h) else cons_h
                    exp_t = mapping[cons_ta] if _is_ph(cons_ta) else cons_ta
                    if (int(h_val) != int(exp_h)) or (int(ta_val) != int(exp_t)):
                        if mis_budget[(pattern_id, 'c-guard')] < MIS_LIMIT:
                            dbg(f"[MIS:c-guard] pid={pattern_id} row=({int(h_val)},{int(cons_r)},{int(ta_val)}) "
                                f"exp=({int(exp_h)},{int(cons_r)},{int(exp_t)}) mapping={mapping}")
                        mis_budget[(pattern_id, 'c-guard')] += 1
                        continue

                    bind_str = _binding_to_str(mapping, placeholders)

                    # Label antecedents used by this instantiation
                    for ante_idx, edge_idx in path:
                        # Never tag the same row as both consequence and antecedent
                        if edge_idx in idxs:
                            continue
                        ant = ants[ante_idx]
                        ah = mapping[_norm_ph(ant[0])] if _is_ph(_norm_ph(ant[0])) else _norm_ph(ant[0])
                        ar = ant[1]
                        at = mapping[_norm_ph(ant[2])] if _is_ph(_norm_ph(ant[2])) else _norm_ph(ant[2])

                        er_head = int(edgelist.at[edge_idx, 'head'])
                        er_rel  = int(edgelist.at[edge_idx, 'rel'])
                        er_tail = int(edgelist.at[edge_idx, 'tail'])
                        if (er_head != int(ah)) or (er_rel != int(ar)) or (er_tail != int(at)):
                            if mis_budget[(pattern_id, 'a')] < MIS_LIMIT:
                                dbg(f"[MIS:a] pid={pattern_id} ante_idx={ante_idx} edge_idx={edge_idx} "
                                    f"row=({er_head},{er_rel},{er_tail}) exp=({int(ah)},{int(ar)},{int(at)}) "
                                    f"bind={bind_str}")
                            mis_budget[(pattern_id, 'a')] += 1
                            continue

                        lab_a = f"{pattern_id}_a{ante_idx}_{bind_str}"
                        if lab_a not in seen_role_binding:
                            _append_label(edge_idx, lab_a)
                            seen_role_binding.add(lab_a)

                    # Label the consequence for ALL rows in this group (same binding)
                    lab_c = f"{pattern_id}_c_{bind_str}"
                    for idx in idxs:
                        if lab_c not in seen_role_binding:
                            _append_label(idx, lab_c)
        
        dbg('Finished labelling')

        # Validity check: check that every label matches its row.
        dbg('[VAL] checking label integrity ...')
        _validate_labels(edgelist, pattern2id)
    else:
        # Skiping labeling greatly speeds up trials in optimize_tkg.py
        dbg('Skipping labeling and validation')

    # Cut off edgelist at n_tws (because forced patterns may have extended past n_tws)
    edgelist = edgelist[edgelist['t'] < config['n_tws']]

    # Aggregate duplicate edges, union labels (as strings)
    edgelist = edgelist.groupby(['head', 'rel', 'tail', 't']).agg({
        'wt': 'sum',
        'pattern': _flatten_labels,
    }).reset_index().sort_values(['t', 'head', 'tail', 'rel']).reset_index(drop=True)

    # Deduplicate per-edge patterns id list
    edgelist['head'] = edgelist['head'].astype(int)
    edgelist['rel'] = edgelist['rel'].astype(int)
    edgelist['tail'] = edgelist['tail'].astype(int)
    edgelist['t'] = edgelist['t'].astype(int)
    
    # Export relevant files
    export_dir = os.path.join(config['export_dir'], f'run_{run_id}')
    os.makedirs(export_dir, exist_ok=True)
    entity2id.to_csv(
        os.path.join(export_dir, 'entity2id.txt'), sep='\t', index=False, header=False)
    relation2id.to_csv(
        os.path.join(export_dir, 'relation2id.txt'), sep='\t', index=False, header=False)
    time2id.to_csv(
        os.path.join(export_dir, 'timestamp2id.txt'), sep='\t', index=False, header=False)
    pattern2id.to_csv(
        os.path.join(export_dir, 'pattern2id.txt'), sep='\t', index=False, header=False)
    with open(os.path.join(export_dir, 'stat.txt'), 'w') as f:
        f.writelines(f'{entity2id.id.nunique()}\t{relation2id.id.nunique()}\t0')
    
    # Create and export graph pattern visualizations
    try:
        pat_txt = os.path.join(export_dir, 'pattern2id.txt')
        out_imgs = os.path.join(export_dir, 'pattern_templates')
        n_rendered = render_patterns_file(pat_txt, out_imgs, layout="circular")
        dbg(f"[run] Rendered {n_rendered} pattern templates to {out_imgs}")
    except Exception as e:
        dbg(f"[run] WARNING: failed to render pattern templates: {e}")
    
    if edgelist.empty:
        raise RuntimeError(
            "No edges were generated. With rnd_avg_density=0 and zero forcing/patterns, "
            "the graph is empty. Increase n_force/p_force and/or pattern counts."
        )

    # Temporal Train-Valid-Test split
    timestamps_unq = pd.Series(edgelist['t'].unique())
    end_train, end_valid, end_test = \
        int(timestamps_unq.quantile(config['split'][0])), \
        int(timestamps_unq.quantile(config['split'][0] + config['split'][1])), \
        int(timestamps_unq.max())
    if (end_train == end_valid) and (config['split'][1] != 0):
        # Allow user to specify 0% validation set
        raise ValueError(f'Split into train and valid sets failed because of quantile collision: {end_train}')
    if (end_valid == end_test) and (config['split'][2] != 0):
        # Allow user to specify 0% test set
        raise ValueError(f'Split into valid and test sets failed because of quantile collision: {end_valid}')
    train_df = edgelist[edgelist['t'] <= end_train]
    valid_df = edgelist[(edgelist['t'] > end_train) & (edgelist['t'] <= end_valid)]
    test_df = edgelist[edgelist['t'] > end_valid]
    cols_export = [
        'head', 
        'rel',
        'tail',
        't',
        'wt',
        'pattern',
    ]
    train_df[cols_export].to_csv(
        os.path.join(export_dir, 'train.txt'), sep='\t', index=False, header=False)
    valid_df[cols_export].to_csv(
        os.path.join(export_dir, 'valid.txt'), sep='\t', index=False, header=False)
    test_df[cols_export].to_csv(
        os.path.join(export_dir, 'test.txt'), sep='\t', index=False, header=False)
    
    # Write config to export directory, for reproducibility
    dump_config_snapshot(export_dir, config)

    # Pattern instantiation summary (based on construction counts)
    summary = pattern2id[['id', 'pattern', 'n_hops']].copy()
    summary['instantiated_bindings_raw'] = summary['id'].map(lambda pid: pattern_counts_raw.get(pid, 0)).astype(int)
    summary['instantiated_bindings_unique'] = summary['id'].map(lambda pid: pattern_counts_unique.get(pid, 0)).astype(int)

    # Top 15 by unique count
    dbg("\n=== Pattern instantiation summary (this run) ===")
    dbg(summary.sort_values('instantiated_bindings_unique', ascending=False).head(15).to_string(index=False))

    # Also write the full table to disk
    export_dir = os.path.join(config['export_dir'], f'run_{run_id}')
    os.makedirs(export_dir, exist_ok=True)
    summary_path = os.path.join(export_dir, 'pattern_instantiation_counts.tsv')
    summary.to_csv(summary_path, sep='\t', index=False)
    dbg(f"Saved per-pattern instantiation counts to: {summary_path}")


if __name__ == "__main__":
    def load_configs_from_file(path: str):
        """Load a Python module from an arbitrary file path and return its `configs` list."""
        path = os.path.abspath(path)
        mod_name = f"user_cfg_{uuid.uuid4().hex}"  # unique module name per load
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config module from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # executes the file (keeps lambdas/callables intact)
        if not hasattr(mod, "configs"):
            raise AttributeError(f"{path} does not define a variable named `configs`")
        cfgs = getattr(mod, "configs")
        if not isinstance(cfgs, (list, tuple)):
            raise TypeError(f"`configs` in {path} must be a list/tuple of dicts")
        return list(cfgs)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", action="append", required=True,
        help="Path to a Python file that defines `configs` (can pass multiple)."
    )
    parser.add_argument(
        "--jobs-per-config", type=int, default=None,
        help="Override `n_jobs` inside each config (useful for shared machines)."
    )
    parser.add_argument(
        "--nest-export", action="store_true",
        help="Nest each config’s outputs under export_dir/<config_stem>/ to avoid collisions. Recommended when running multiple configs at once."
    )
    args = parser.parse_args()

    # Load all configs from all files
    all_cfgs = []
    for cfg_path in args.config:
        cfgs_here = load_configs_from_file(cfg_path)
        # Optionally nest export_dir per config file to avoid collisions
        if args.nest_export:
            stem = Path(cfg_path).stem
            for i, cfg in enumerate(cfgs_here):
                cfg = deepcopy(cfg)
                base_export = cfg.get("export_dir", "data_out")
                cfg["export_dir"] = os.path.join(base_export, stem)
                all_cfgs.append(cfg)
        else:
            all_cfgs.extend(cfgs_here)

    # Run each config’s runs in parallel as you already do
    for cfg in all_cfgs:
        n_runs = int(cfg["n_runs"])
        n_jobs = int(args.jobs_per_config or cfg["n_jobs"])
        n_jobs = min(n_jobs, n_runs)

        Parallel(n_jobs=n_jobs)(
            delayed(run)(cfg, run_id) for run_id in range(n_runs)
        )

"""
python run.py -c configs/example.py

python run.py -c configs/example1.py -c configs/example2.py --nest-export
"""