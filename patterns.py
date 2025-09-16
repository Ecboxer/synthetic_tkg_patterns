import pandas as pd

import random
import re

from collections import Counter, defaultdict
from typing import Iterable, Hashable

from temporalpattern import TemporalPattern
from utils import entities_intersect, \
    entities_connect_triples, \
    entities_connect_components, \
    force_swap_to_entities, \
    force_connect_components, \
    create_time_lag_tuples, \
    canonicalize_entities_and_relations, \
    is_subpattern
from itertools import product


def adjacent_pairs_share_entity(sets_in_order: Iterable[set]) -> bool:
    """
    Given a list of 2-element sets of entities (e.g., [{a,b}, {c,d}, {e,f}, {g,h}]),
    enforce the cyclic rule: every adjacent pair (i, i+1 mod n) must intersect.
    """
    S = list(map(frozenset, sets_in_order))
    n = len(S)
    return n > 0 and all(S[i] & S[(i+1) % n] for i in range(n))

def forms_single_cycle(
    sets_in_order: Iterable[set],
    require_adjacent_intersection: bool = True,
) -> bool:
    """
    Each item is a set/tuple of size 2: {u, v}.
    Returns True iff these edges form ONE simple cycle (undirected):
      - graph is connected
      - every node has degree exactly 2
    """
    # normalize edges and collect nodes
    edges = [tuple(sorted(s)) for s in sets_in_order]
    if not edges:
        return False

    # Optional: quick screen using adjacent intersections
    if require_adjacent_intersection and not adjacent_pairs_share_entity(sets_in_order):
        return False

    # degrees and adjacency (ignoring multiplicity for connectivity)
    deg = Counter()
    adj = defaultdict(set)
    nodes = set()
    for edge in edges:
        u = edge[0]
        # edge could be a self-loop
        try: v = edge[1]
        except IndexError: v = edge[0]
        deg[u] += 1; deg[v] += 1
        adj[u].add(v); adj[v].add(u)
        nodes.add(u); nodes.add(v)

    # degree must be exactly 2 everywhere
    if any(deg[x] != 2 for x in nodes):
        return False

    # connectedness (BFS/DFS on simple graph)
    seen = set()
    stack = [next(iter(nodes))]
    while stack:
        x = stack.pop()
        if x in seen: 
            continue
        seen.add(x)
        stack.extend(adj[x] - seen)
    if len(seen) != len(nodes):
        return False

    # one cycle implies |E| == |V| in a simple cycle
    if len(edges) != len(nodes):
        return False

    return True


# Pattern validity checkers
def is_valid_1hop_pattern(
    ant1, cons,
    require_unique=True,
    prohibit_selfconnections=False,
    prohibit_new_consequence_relations=True,
    require_sequential_rule=False,
):
    # Helper to test validity of 1-hop patterns
    # Get entiites
    e1s = {ant1[0], ant1[2]}
    ec = {cons[0], cons[2]}

    # Constraint A: (Optional) Require distinct triples
    if require_unique and len({ant1, cons}) < 2:
        return False
    # Constraint B: (Optional) Prohibit self-connections
    if prohibit_selfconnections and ((len(e1s) < 2) or (len(ec) < 2)):
        return False
    # Constraint C: (Optional) Require sequentialness (consecutive triples must share some entity)
    if require_sequential_rule:
        # TODO Split up the hyperparameter for requiring a cycle with the hyperparameter
        # requiring adjacent edges to intersect.
        edge_sets = [e1s, ec]
        cycle_check = forms_single_cycle(
            edge_sets,
            require_adjacent_intersection=require_sequential_rule,
        )
        if not cycle_check:
            return False
    # Constraint 1: Consequence entities must be in antecedent
    if not ec.issubset(e1s):
        return False
    # Constraint 2: (Optional) Consequence relation must be in antecedent
    if prohibit_new_consequence_relations and (cons[1] != ant1[1]):
        return False
    return True

def is_valid_2hop_pattern(
    ant1, ant2, cons,
    require_unique=True,
    prohibit_selfconnections=False,
    prohibit_new_consequence_relations=True,
    require_sequential_rule=False,
):
    # Helper to test validity of 2-hop patterns
    # Get entities
    e1s = {ant1[0], ant1[2]}
    e2s = {ant2[0], ant2[2]}
    ec = {cons[0], cons[2]}

    # Constraint A: (Optional) Require distinct triples
    if require_unique and len({ant1, ant2, cons}) < 3:
        return False
    # Constraint B: (Optional) Prohibit self-connections
    if prohibit_selfconnections and ((len(e1s) < 2) or (len(e2s) < 2) or (len(ec) < 2)):
        return False
    # Constraint C: (Optional) Require sequentialness (consecutive triples must share some entity)
    if require_sequential_rule:
        # TODO Split up the hyperparameter for requiring a cycle with the hyperparameter
        # requiring adjacent edges to intersect.
        edge_sets = [e1s, e2s, ec]
        cycle_check = forms_single_cycle(
            edge_sets,
            require_adjacent_intersection=require_sequential_rule,
        )
        if not cycle_check:
            return False
    # Constraint 1: Consequence entities must be in antecedent
    if not ec.issubset(e1s | e2s):
        return False
    # Constraint 2: (Optional) Consequence relation must be in antecedent
    if prohibit_new_consequence_relations and (cons[1] not in {ant1[1], ant2[1]}):
        return False

    # Test intersection of antecedent entities
    intersects = len(e1s & e2s) > 0
    # Test whether the consequence connects the antecedents
    connects_disjoint = (
        (cons[0] in e1s and cons[2] in e2s) or
        (cons[2] in e1s and cons[0] in e2s)
    )

    # Constraint 3: Antecedents must be connected OR be joined by the consequence
    if not (intersects or connects_disjoint):
        return False
    return True

def is_valid_3hop_pattern(
    ant1, ant2, ant3, cons,
    require_unique=True,
    prohibit_selfconnections=False,
    prohibit_new_consequence_relations=True,
    require_sequential_rule=False,
):
    # Helper to test validity of 3-hop patterns
    # Get entities
    e1s = {ant1[0], ant1[2]}
    e2s = {ant2[0], ant2[2]}
    e3s = {ant3[0], ant3[2]}
    ec = {cons[0], cons[2]}

    # Constraint A: (Optional) Require distinct triples
    if require_unique and len({ant1, ant2, ant3, cons}) < 4:
        return False
    # Constraint B: (Optional) Prohibit self-connections
    if prohibit_selfconnections and ((len(e1s) < 2) or (len(e2s) < 2) or (len(e3s) < 2) or (len(ec) < 2)):
        return False
    # Constraint C: (Optional) Require sequentialness (consecutive triples must share some entity)
    if require_sequential_rule:
        # TODO Split up the hyperparameter for requiring a cycle with the hyperparameter
        # requiring adjacent edges to intersect.
        edge_sets = [e1s, e2s, e3s, ec]
        cycle_check = forms_single_cycle(
            edge_sets,
            require_adjacent_intersection=require_sequential_rule,
        )
        if not cycle_check:
            return False
    # Constraint 1: Consequence entities must be in antecedent
    if not ec.issubset(e1s | e2s | e3s):
        return False
    # Constraint 2: (Optional) Consequence relation must be in antecedent
    if prohibit_new_consequence_relations and (cons[1] not in {ant1[1], ant2[1], ant3[1]}):
        return False

    # Test whether antecedent 2 is connected to antecedent 1 AND antecedent 3 is
    # connected to antecedents 1 or 2 ("sequential")
    connected_seq = (
        (ant2[0] in e1s or ant2[2] in e1s) and
        (ant3[0] in (e1s | e2s) or ant3[2] in (e1s | e2s))
    )
    # Test whether antecedent 2 is disconnected from antecedent 1
    first_second_disjoint = not (ant2[0] in e1s and ant2[2] in e1s)
    # Test whether antecedent 3 connects antecedents 1 and 2
    third_connects = (
        (ant3[0] in e1s and ant3[2] in e2s) or
        (ant3[2] in e1s and ant3[0] in e2s)
    )
    # Test whether antecedent 3 intersects with either previous antecedent
    third_partial_connect = any(e in e1s | e2s for e in (ant3[0], ant3[2]))
    # Test whether antecedent 3 intersects with antecedent 1
    third_connect_first = any(e in e1s for e in (ant3[0], ant3[2]))
    # Test whether antecedent 3 intersects with antecedent 2
    third_connect_second = any(e in e2s for e in (ant3[0], ant3[2]))
    # Test whether the consequence connects the first and third antecedents to the second
    cons_connects_first_and_third_to_second = (
        (cons[0] in (e1s | e3s) and cons[2] in e2s) or
        (cons[2] in (e1s | e3s) and cons[0] in e2s)
    )
    # Test whether the consequence connects the second and third antecedents to the first
    cons_connects_second_and_third_to_first = (
        (cons[0] in (e2s | e3s) and cons[2] in e1s) or
        (cons[2] in (e2s | e3s) and cons[0] in e1s)
    )
    # Test whether the consequence connects the first and secnod antecedents to the third
    cons_connects_first_and_second_to_third = (
        (cons[0] in (e1s | e2s) and cons[2] in e3s) or
        (cons[2] in (e1s | e2s) and cons[0] in e3s)
    )

    # Constraint 3: Antecedent and consequence connectivity
    # Check sequential connectivity, any sequentially connected antecedents pass
    if not connected_seq:
        # If the first and second antecedents are disjoint then either ...
        if first_second_disjoint:
            # The third antecedent must connect the disjoint antecedents OR
            # (the third antecedent must connect to one of the first two antecedents
            # AND the consequence must connect the disjoint antecedents)
            if not (
                third_connects or (
                    (third_connect_first and cons_connects_first_and_third_to_second) or
                    (third_connect_second and cons_connects_second_and_third_to_first)
                )
            ):
                return False
        # If the third antecedent is disjoint from the previous antecedents then ...
        elif not third_partial_connect:
            # The consequence must connect the first two antecedents and the disjoint third
            if not cons_connects_first_and_second_to_third:
                return False
    return True

def generate_n_hop_patterns(
    n_hops=1,
    require_unique_triples=True,
    prohibit_selfconnections=False,
    prohibit_new_consequence_relations=True,
    require_sequential_rule=False,
):
    # Get maximum number of valid entities and relations based on n_hops
    n_hops2num_entities = {
        1: 2,
        2: 4,
        3: 5,
    }
    n_hops2num_relations = {
        1: 2,
        2: 3,
        3: 4,
    }
    num_entities = n_hops2num_entities.get(n_hops)
    num_relations = n_hops2num_relations.get(n_hops)

    # Wrapper to generate all 1-, 2-, or 3-hop patterns
    # Define entities and relations
    entities = [f"e{i+1}" for i in range(num_entities)]
    relations = [f"r{i+1}" for i in range(num_relations)]

    # Create all possible triples
    triples = list(product(entities, relations, entities))
    patterns = set()
    rows = []

    # Create all possible antecedent sequences and consequences and check for validity
    if n_hops == 1:
        for ant1 in triples:
            ents = set([ant1[0], ant1[2]])
            for cons in product(ents, relations, ents):
                if is_valid_1hop_pattern(
                    ant1, cons,
                    require_unique=require_unique_triples,
                    prohibit_selfconnections=prohibit_selfconnections,
                    prohibit_new_consequence_relations=prohibit_new_consequence_relations,
                    require_sequential_rule=require_sequential_rule,
                ):
                    key = canonicalize_entities_and_relations(ant1, cons)
                    if key not in patterns:
                        patterns.add(key)
                        a1, c = key
                        rows.append({
                            'antecedent1_h': a1[0], 'antecedent1_r': a1[1], 'antecedent1_t': a1[2],
                            'consequence_h': c[0], 'consequence_r': c[1], 'consequence_t': c[2],
                        })
    elif n_hops == 2:
        for ant1, ant2 in product(triples, repeat=2):
            ents = set([ant1[0], ant1[2], ant2[0], ant2[2]])
            for cons in product(ents, relations, ents):
                if is_valid_2hop_pattern(
                    ant1, ant2, cons,
                    require_unique=require_unique_triples,
                    prohibit_selfconnections=prohibit_selfconnections,
                    prohibit_new_consequence_relations=prohibit_new_consequence_relations,
                    require_sequential_rule=require_sequential_rule,
                ):
                    key = canonicalize_entities_and_relations(ant1, ant2, cons)
                    if key not in patterns:
                        patterns.add(key)
                        a1, a2, c = key
                        rows.append({
                            'antecedent1_h': a1[0], 'antecedent1_r': a1[1], 'antecedent1_t': a1[2],
                            'antecedent2_h': a2[0], 'antecedent2_r': a2[1], 'antecedent2_t': a2[2],
                            'consequence_h': c[0], 'consequence_r': c[1], 'consequence_t': c[2],
                        })
    elif n_hops == 3:
        for ant1, ant2, ant3 in product(triples, repeat=3):
            ents = set([ant1[0], ant1[2], ant2[0], ant2[2], ant3[0], ant3[2]])
            for cons in product(ents, relations, ents):
                if is_valid_3hop_pattern(
                    ant1, ant2, ant3, cons,
                    require_unique=require_unique_triples,
                    prohibit_selfconnections=prohibit_selfconnections,
                    prohibit_new_consequence_relations=prohibit_new_consequence_relations,
                    require_sequential_rule=require_sequential_rule,
                ):
                    key = canonicalize_entities_and_relations(ant1, ant2, ant3, cons)
                    if key not in patterns:
                        patterns.add(key)
                        a1, a2, a3, c = key
                        rows.append({
                            'antecedent1_h': a1[0], 'antecedent1_r': a1[1], 'antecedent1_t': a1[2],
                            'antecedent2_h': a2[0], 'antecedent2_r': a2[1], 'antecedent2_t': a2[2],
                            'antecedent3_h': a3[0], 'antecedent3_r': a3[1], 'antecedent3_t': a3[2],
                            'consequence_h': c[0], 'consequence_r': c[1], 'consequence_t': c[2],
                        })
    return pd.DataFrame(rows)

def instantiate_patterns_from_df(
    pattern_df: pd.DataFrame,
    relation2id: pd.DataFrame,
    n: int,
    time_lags: 'List[Tuple(float,float)]',
    seed: int = None,
    pattern_quadruples: 'List[Tuple[int,int,int]]' = [],
    max_retries: int = 1,
) -> 'List[TemporalPattern]':
    """
    Sample `n` patterns from canonical pattern_df and instantiate relations (not entities).

    Args:
        pattern_df (pd.DataFrame): Canonical patterns with placeholders (e1, r1, etc.)
        relation2id (pd.DataFrame) Available relation IDs (with 'id' and optional 'wt' columns)
        n (int): Number of patterns to instantiate
        time_lags (List[float]): Time lags with which antecedents and consequences
            can occur validly. Either a list of float tuples or of functions that
            can create such floats when called.
        seed (int): Random seed for reproducibility. Default None.
        pattern_quadruples (List[Tuple[int,int,int]]): List of existing patterns to check for subpatterns against
        max_retries (int): Maximum number of attempts at reinstantiation before skipping

    Returns:
        List[TemporalPattern]
    """
    # Get weights over relations
    weights = relation2id.get('wt', [1] * len(relation2id))

    # Sample patterns to use, with replacement
    sampled_patterns = pattern_df.sample(
        n, replace=True, random_state=seed if seed else None,
    ).reset_index(drop=True)
    patterns = []

    outer_attempt = 0  # Overall relation instantiation attempt tracker
    for _, row in sampled_patterns.iterrows():
        inner_attempt = 0  # Per-pattern attempt traker
        while inner_attempt <= max_retries:
            # Determine how many unique relation placeholders we have
            rel_placeholders = sorted(set(col for col in row.index if re.search(r'_r$', col)))
            n_rels = len(rel_placeholders)

            # Sample relations to fill them, with replacement
            sampled_rels = relation2id.sample(
                n_rels, weights=weights, replace=True,
                random_state=seed
            )['id'].tolist()
            
            # Mapping from placeholders to instantiated relations
            rel_map = {f"r{i+1}": sampled_rels[i] for i in range(n_rels)}

            # Create antecedent triples
            antecedent = []
            i = 1
            while f"antecedent{i}_h" in row:
                ant = (
                    row[f"antecedent{i}_h"],
                    rel_map[row[f"antecedent{i}_r"]],
                    row[f"antecedent{i}_t"],
                )
                antecedent.append(ant)
                i += 1

            # Create consequence triple
            cons = (
                row["consequence_h"],
                rel_map[row["consequence_r"]],
                row["consequence_t"]
            )

            # Create time lags
            time_lag_tuples = create_time_lag_tuples(time_lags, antecedent, seed=seed)

            # Create final pattern
            pat = TemporalPattern(
                antecedent=antecedent,
                consequence=cons,
                time_lags=time_lag_tuples,
                n_hops=len(antecedent),
            )

            # Check if it's a subpattern
            quad = pat.__quadruples__()
            if not is_subpattern(quad, pattern_quadruples, ignore_timing=True):
                # Store pattern and exit while loop
                patterns.append(pat)
                pattern_quadruples.append(quad)
                break

            # Increment attempts
            outer_attempt += 1
            inner_attempt += 1

    return patterns
