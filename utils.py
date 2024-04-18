from itertools import combinations, product

import random


def is_subpattern(subpattern: 'List[Tuple]', patterns: 'List[Tuple]') -> bool:
    """ Test whether subpattern is a subpattern of any member of patterns
    """
    n = len(subpattern)
    pattern_subsets = set([
        tuple(pattern[idx:idx+n]) for pattern in patterns
        for idx in range(len(pattern)-n+1)
    ])
    if tuple(subpattern) in pattern_subsets:
        return True
    return False

def entities_intersect(entities1: 'List[int]', entities2: 'List[int]') -> bool:
    """ Indicate whether entities1 and entities2 have any intersection
    """
    if len(set(entities1).intersection(entities2)) > 0:
        return True
    return False

def entities_connect_triples(
    e1: int, e2: int, triple1: 'Tuple[int,int,int]', triple2: 'Tuple[int,int,int]'
) -> bool:
    """ Indicate whether entities e1 and e2 connect triples triple1 and triple2
    """
    if (e1 in {triple1[0], triple1[2]}) & (e2 in {triple2[0], triple2[2]}):
        return True
    elif (e2 in {triple1[0], triple1[2]}) & (e1 in {triple2[0], triple2[2]}):
        return True
    return False

def entities_connect_components(
    e1: int,
    e2: int,
    comp1: 'List[Tuple[int,int,int]]',
    comp2: 'List[Tuple[int,int,int]]'
) -> bool:
    """ Indicate whether entities e1 and e2 connect components comp1 and comp2
    """
    for triple1, triple2 in product(comp1, comp2):
        if entities_connect_triples(e1, e2, triple1, triple2):
            return True
    return False

def combinations_of_increasing_size(iterable, a, b):
    """ Return combinations of iterable of size from a to b, inclusive
    """
    all_combinations = []
    for size in range(a, b+1):
        combs = combinations(iterable, size)
        for comb in combs:
            yield comb

def force_swap_to_entities(
    idxs_to_force: 'List[int]',
    sampled_entities: 'List[int]',
    swap_to_entities: 'List[int]',
    seed: int = None,
) -> None:
    """ Force at least one of idxs_to_swap to be switching in sampled_entities
    to one of swap_to_entities. Note, alters swap_to_entities in place.
    """
    random.seed(seed)
    to_swap = random.choice(
        list(combinations_of_increasing_size(idxs_to_force, 1, len(idxs_to_force)))
    )
    for idx in to_swap:
        swap_to = random.choice(swap_to_entities)
        sampled_entities[idx] = swap_to

def force_connect_components(
    idxs_to_force: 'List[int,int]',
    sampled_entities: 'List[int]',
    comp1: 'List[int]',
    comp2: 'List[int]',
    seed: int = None,
) -> None:
    """ Force at least one of idxs_to_force to be switched so as to connect comp1
    and comp2. Note, alters sampled_entities in place.
    """
    if len(idxs_to_force) != 2:
        raise ValueError(
            'force_connecte_components only implemented for idxs_to_force of length 2'
        )
    random.seed(seed)
    comps = [comp1, comp2]
    if sampled_entities[idxs_to_force[0]] in comps[0]:
        sampled_entities[idxs_to_force[1]] = random.choice(comps[1])
    elif sampled_entities[idxs_to_force[0]] in comps[1]:
        sampled_entities[idxs_to_force[1]] = random.choice(comps[0])
    elif sampled_entities[idxs_to_force[1]] in comps[0]:
        sampled_entities[idxs_to_force[0]] = random.choice(comps[1])
    elif sampled_entities[idxs_to_force[1]] in comps[1]:
        sampled_entities[idxs_to_force[0]] = random.choice(comps[0])
    else:
        random.shuffle(comps)
        sampled_entities[idxs_to_force[0]] = random.choice(comps[0])
        sampled_entities[idxs_to_force[1]] = random.choice(comps[1])

def create_time_lag_tuples(
    time_lags: 'List[Tuple(float,float)]',
    antecedent: 'List[Tuple[int,int,int]]',
) -> 'List[Tuple[float,float]]':
    """ Create time_lag_tuples used to instantiate patterns. Contains logic to prohibit: identical
    antecedents from having 0 time lag between them, the consequence from having 0 lag from the
    last antecedent.
    """
    time_lag_tuples = []
    for idx, time_lag in enumerate(time_lags):
        lag_min, lag_max = time_lag[0], time_lag[1]
        time_lag_tuple = []
        if type(lag_min) in [float, int]:
            pass
        else:
            lag_min = lag_min()
        if type(lag_max) in [float, int]:
            pass
        else:
            lag_max = lag_max()
        # Prohibit identical antecedents from having 0 lag_min between them
        if (idx < len(antecedent)-1) and (antecedent[idx] == antecedent[idx+1]):
            lag_min = max(1, lag_min)
        # Prohibit consequence from having 0 lag_min from the last antecedent
        if idx == len(antecedent)-1:
            lag_min = max(1, lag_min)
        # Enforce that lag_max is no smaller than lag_min
        time_lag_tuples.append((lag_min, max(lag_min, lag_max)))
    return time_lag_tuples
