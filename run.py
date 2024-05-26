import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

import os
import random
import shutil

from config import configs
from patterns import create_1_hop_pattern, \
    create_2_hop_pattern, \
    create_3_hop_pattern
from temporalpattern import TemporalPattern
from utils import is_subpattern


def create_entity2id(config) -> pd.DataFrame:
    if config['pat_distr_ents']:
        wts = config['pat_distr_ents'](config['n_ents'])
    else:
        wts = [1]*config['n_ents']
    return pd.DataFrame({
        'name': range(config['n_ents']),
        'id': range(config['n_ents']),
        'wt': wts,
    })

def create_relation2id(config) -> pd.DataFrame:
    if config['pat_distr_rels']:
        wts = config['pat_distr_rels'](config['n_rels'])
    else:
        wts = [1]*config['n_rels']
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
        # Find out if quadruple is 
        quad = pat.__quadruples__()
        if ~is_subpattern(quad, pattern_quadruples):
            patterns.append(pat)
            pattern_quadruples.append(quad)
            new_pat = True
        retry += 1

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

def run(config: 'Dict[str,]', run_id: int):
    """ Create TKGs according to configuration from config.py file
    """
    # Create ids for entities, relations, and time windows
    entity2id = create_entity2id(config)
    relation2id = create_relation2id(config)
    time2id = create_time2id(config)

    # Instantiate patterns
    # Start from 3-hop patterns, then 2-hop, then 1-hop
    # Prohibit any new patterns from being contained (antecedent and consequence) in the antecedent of
    # an existing larger pattern or being identical to an already chosen same-sized pattern
    patterns = []
    pattern_quadruples = []
    for _ in range(config['n_3_hop']):
        add_new_pattern(
            config, patterns, pattern_quadruples, create_3_hop_pattern,
            config['time_lag_3_hop'], entity2id, relation2id,
        )
    for _ in range(config['n_2_hop']):
        add_new_pattern(
            config, patterns, pattern_quadruples, create_2_hop_pattern,
            config['time_lag_2_hop'], entity2id, relation2id,
        )
    for _ in range(config['n_1_hop']):
        add_new_pattern(
            config, patterns, pattern_quadruples, create_1_hop_pattern,
            config['time_lag_1_hop'], entity2id, relation2id,
        )
    # Create dataframe of pattern ids
    pattern2id = create_pattern2id(patterns)

    # Apply patterns
    edgelist = create_edgelist()
    pbar_tws = tqdm(range(config['n_tws']))
    for t in pbar_tws:
        pbar_tws.set_description(f'Time window: {t}')
        # First randomly wire entities
        pbar_ent = tqdm(entity2id['id'])
        dfs_i = []
        for ent_id in pbar_ent:
            pbar_ent.set_description(f'Entity: {ent_id}')
            # Sample entities to use as tails
            if config['rnd_avg_density_distr']:
                dens = config['rnd_avg_density_distr']()
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
            tails = entity2id['id'].sample(dens, replace=True)
            # Sample relations to connect them
            rels = relation2id['id'].sample(dens, replace=True)
            df_i = pd.DataFrame({
                'head': [ent_id]*dens,
                'rel': rels.values,
                'tail': tails.values,
                't': [t]*dens,
                'wt': [1]*dens,
                'pattern': [[-1]]*dens,  # -1 indicates a randomly wired edge
            })
            dfs_i.append(df_i)
        edgelist = pd.concat([edgelist]+dfs_i)

        # Iterate over patterns
        heads, rels, tails, pats = [], [], [], []
        dfs_pat = []
        for label, pattern_id in zip(pattern2id['pattern'], pattern2id['id']):
            # Instantiate pattern from label
            pattern = TemporalPattern()
            pattern.from_label(label)

            # Artificially create valid patterns 
            rnd = random.random()
            if rnd < config['n_hops2p_force'][pattern.n_hops]:
                # Create the antecedent in this and subsequent windows
                # Track time window of current antecedent as we create them
                t_i = int(t)
                heads_pat, rels_pat, tails_pat, ts_pat = [], [], [], []
                for antecedent, time_lag in zip(pattern.antecedent, pattern.time_lags):
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
                    'pattern': [[]]*len(heads_pat),
                })
                dfs_pat.append(df_pat)
            
            # Apply valid patterns
            rnd = random.random()
            if rnd < config['p_skip_consequence']:
                # Skip the consequence even though antecedents may be satisfied
                continue
            # Test whether antecedents are satisfied in prior windows
            antecedents_satisfied = False
            # Iterate over antecedents in reverse order (most recent to least recent)
            t_i = [t]  # Track current time window(s) for antecedent validation
            for antecedent, time_lag in zip(pattern.antecedent[::-1], pattern.time_lags[::-1]):
                # Check whether the antecedent exists in the edgelist
                df_ants = [
                    edgelist[
                        (edgelist['head'] == antecedent[0]) &
                        (edgelist['rel'] == antecedent[1]) &
                        (edgelist['tail'] == antecedent[2]) &
                        (edgelist['t'] <= t_-time_lag[0]) & 
                        (edgelist['t'] >= t_-time_lag[1])
                    ] for t_ in t_i
                ]
                df_ants = [df for df in df_ants if df.shape[0] > 0]
                if len(df_ants) == 0:
                    # No satisfied antecedent
                    antecedents_satisfied = False
                    continue
                t_i = pd.concat(df_ants)['t'].unique().tolist()
                antecedents_satisfied = True
            # If all antecedents are satisfied, create consequence in current time window
            if antecedents_satisfied:
                heads.append(pattern.consequence[0])
                rels.append(pattern.consequence[1])
                tails.append(pattern.consequence[2])
                pats.append([pattern_id])
        # Add new forced patterns to edgelist
        edgelist = pd.concat([edgelist]+dfs_pat)
        # Add all new consequences to edgelist
        df_con = pd.DataFrame({
            'head': heads,
            'rel': rels,
            'tail': tails,
            't': [t]*len(heads),
            'wt': [1]*len(heads),
            'pattern': [[]]*len(heads),
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

    # Cut off edgelist at n_tws (because forced patterns may have extended past n_tws)
    edgelist = edgelist[edgelist['t'] < config['n_tws']]
    # Aggregate duplicate edges
    edgelist = edgelist.groupby(['head', 'rel', 'tail', 't']).agg({
        'wt': 'sum',
        'pattern': lambda x: sorted(list(set([el for ids in x for el in ids]))),
    }).reset_index().sort_values(['t', 'head', 'tail', 'rel']).reset_index(drop=True)
    
    # Post-creation, label all valid patterns
    for label, pattern_id in zip(pattern2id['pattern'], pattern2id['id']):
        # Instantiate pattern from label
        pattern = TemporalPattern()
        pattern.from_label(label)

        satisfying_idxs = get_satisfying_idxs(
            pattern, edgelist, prev_t=-1, satisfying_idxs=[],
        )
        for idx in satisfying_idxs:
            edgelist.loc[idx]['pattern'].append(pattern_id)

    # Deduplicate patterns
    edgelist.loc[:,'pattern'] = edgelist['pattern'].apply(lambda x: sorted(list(set(x))))
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
    
    # Copy config to export directory, for reproducibility
    shutil.copy2('config.py', export_dir)


if __name__ == "__main__":
    # Iterate over configuration files
    # Note: This could also be parallelized or otherwise done in a better way
    for config in configs:
        n_runs = config['n_runs']
        n_jobs = config['n_jobs']
        n_jobs = min(n_jobs, n_runs)  # Can't have more jobs than runs

        Parallel(n_jobs=n_jobs)(
            delayed(run)(config, run_id) for run_id in range(n_runs)
        )
