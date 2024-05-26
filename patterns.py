import pandas as pd

import random

from temporalpattern import TemporalPattern
from utils import entities_intersect, \
    entities_connect_triples, \
    entities_connect_components, \
    force_swap_to_entities, \
    force_connect_components, \
    create_time_lag_tuples


def create_1_hop_pattern(
    entity2id: pd.DataFrame,
    relation2id: pd.DataFrame,
    time_lags: 'List[Tuple(float,float)]',
    seed: int = None,
) -> TemporalPattern:
    """ Create a 1-hop temporal pattern
    (e1, r1, e2, t1) → (e3, r2, e4, t2)
    Such that:
        - e3 and e4 \in {e1, e2}
        - t2 > t1
    Args:
        entity2id (pd.DataFrame): Dataframe with entity ids in column 'id'
        relation2id (pd.DataFrame): Dataframe with relation ids in column 'id'
        time_lags (List[float]): Time lags with which antecedents and consequences
            can occur validly. Either a list of float tuples or of functions that
            can create such floats when called.
        seed (int): Random seed, default None
    """
    random.seed(seed)
    # Randomly select all initial entities and relations to be used
    sampled_entities = entity2id.sample(
        4, weights='wt', replace=True, random_state=seed)['id'].tolist()
    sampled_relations = relation2id.sample(
        2, weights='wt', replace=True, random_state=seed)['id'].tolist()
    # Define antecedent
    antecedent = [
        (
            sampled_entities[0],  # e1
            sampled_relations[0],  # r1
            sampled_entities[1],  # e2
        )
    ]

    # Consquence must satisfy the constraint that both of its entities are in
    # some antecedent
    if sampled_entities[2] not in sampled_entities[:2]:
        # Force consequent entity to switch to an antecedent entity
        force_swap_to_entities([2], sampled_entities, sampled_entities[:2], seed)
    if sampled_entities[3] not in sampled_entities[:2]:
        force_swap_to_entities([3], sampled_entities, sampled_entities[:2], seed)
    # Consequence must have a relation in some antecedent
    if sampled_relations[1] not in sampled_relations[:1]:
        force_swap_to_entities([1], sampled_relations, sampled_relations[:1], seed)
    
    consequence = (
        sampled_entities[2],  # e3
        sampled_relations[1],  # r2
        sampled_entities[3],  # e4
    )

    time_lag_tuples = create_time_lag_tuples(time_lags, antecedent)

    return TemporalPattern(
        antecedent=antecedent,
        consequence=consequence,
        time_lags=time_lag_tuples,
        n_hops=1,
    )

def create_2_hop_pattern(
    entity2id: pd.DataFrame,
    relation2id: pd.DataFrame,
    time_lags: 'List[Tuple(float,float)]',
    seed: int = None,
) -> TemporalPattern:
    """ Create a 2-hop temporal pattern
    (e1, r1, e2, t1) & (e3, r2, e4, t2) → (e5, r3, e6, t3)
    Such that:
        - (e3 or e4 \in {e1, e2}) and (e5 and e6 \in {e1, e2, e3, e4})
        - t3 > t2 >= t1
        - (e1, r1, e2, t1) != (e3, r2, e4, t2)

    Args:
        entity2id (pd.DataFrame): Dataframe with entity ids in column 'id'
        relation2id (pd.DataFrame): Dataframe with relation ids in column 'id'
        time_lags (List[float]): Time lags with which antecedents and consequences
            can occur validly. Either a list of float tuples or of functions that
            can create such floats when called.
        seed (int): Random seed, default None
    """
    random.seed(seed)
    # Randomly select all initial entities and relations to be used
    sampled_entities = entity2id.sample(
        6, weights='wt', replace=True, random_state=seed)['id'].tolist()
    sampled_relations = relation2id.sample(
        3, weights='wt', replace=True, random_state=seed)['id'].tolist()
    
    if ~entities_intersect(sampled_entities[2:4], sampled_entities[:2]):
        # Force the second antecedent to intersect the first
        force_swap_to_entities([2,3], sampled_entities, sampled_entities[:2], seed)
    
    # Define antecedent
    antecedent = [
        (
            sampled_entities[0],  # e1
            sampled_relations[0],  # r1
            sampled_entities[1],  # e2
        ),
        (
            sampled_entities[2],  # e3
            sampled_relations[1],  # r2
            sampled_entities[3],  # e4
        )
    ]
    
    # Consquence must have both of its entities in some antecedent
    if sampled_entities[4] not in sampled_entities[:4]:
        # Force consequent entity to switch to an antecedent entity
        force_swap_to_entities([4], sampled_entities, sampled_entities[:4], seed)
    if sampled_entities[5] not in sampled_entities[:4]:
        force_swap_to_entities([5], sampled_entities, sampled_entities[:4], seed)
    # Consequence must have a relation in some antecedent
    if sampled_relations[2] not in sampled_relations[:2]:
        force_swap_to_entities([2], sampled_relations, sampled_relations[:2], seed)
    
    # Define consequence
    consequence = (
        sampled_entities[4],  # e5
        sampled_relations[2],  # r3
        sampled_entities[5],  # e6
    )

    time_lag_tuples = create_time_lag_tuples(time_lags, antecedent)
    
    return TemporalPattern(
        antecedent=antecedent,
        consequence=consequence,
        time_lags=time_lag_tuples,
        n_hops=2,
    )

def create_3_hop_pattern(
    entity2id: pd.DataFrame,
    relation2id: pd.DataFrame,
    time_lags: 'List[Tuple(float,float)]',
    seed: int = None,
) -> TemporalPattern:
    """ Create a 3-hop temporal pattern
    (e1, r1, e2, t1) & (e3, r2, e4, t2) & (e5, r3, e6, t3) → (e7, r4, e8, t4)
    Such that:
        - All antecedents intersect sequentially: (e3 or e4 \in {e1, e2}) and (e5 or e6 \in {e1, e2, e3, e4}) and (e7 and e8 \in {e1, e2, e3, e4, e5, e6})
        - Second antecedent does not intersect first: ~(e3 and e4 \in {e1, e2})
        - Third antecedent connects them: ((e5 \in {e1, e2} and e6 \in {e3, e4}) or (e5 \in {e3, e4} and e6 \in {e1, e2})) then (e7 or e8 \in {e1, e2, e3, e4, e5, e6}),
        - t4 > t3 >= t2 >= t1
        - (e1, r1, e2, t1) != (e3, r2, e4, t2) and (e1, r1, e2, t1) != (e5, r3, e6, t3) and (e1, r1, e2, t1) != (e3, r2, e4, t2)

    Args:
        entity2id (pd.DataFrame): Dataframe with entity ids in column 'id'
        relation2id (pd.DataFrame): Dataframe with relation ids in column 'id'
        time_lags (List[float]): Time lags with which antecedents and consequences
            can occur validly. Either a list of float tuples or of functions that
            can create such floats when called.
        seed (int): Random seed, default None
    """
    random.seed(seed)
    # Randomly select all initial entities and relations to be used
    sampled_entities = entity2id.sample(
        8, weights='wt', replace=True, random_state=seed)['id'].tolist()
    sampled_relations = relation2id.sample(
        4, weights='wt', replace=True, random_state=seed)['id'].tolist()
    
    # Note: This check on the third antecedent can be moved after the check on the
    # intersection of the first and second antecedents, since that one also modified
    # the third antecedent.
    # Third antecedent must intersect at least one prior antecedent
    if ~entities_intersect(sampled_entities[:4], sampled_entities[4:6]):
        # Enforce that the third antecedent include entities from at least one prior
        # antecedent
        force_swap_to_entities([4,5], sampled_entities, sampled_entities[:4], seed)
    # Second antecedent must intersect first or they must be connected by third antecedent
    if ~entities_intersect(sampled_entities[2:4], sampled_entities[:2]):
        # Second antecedent does not intersect first antecedent
        # Enforce that the third antecedent connects them
        if not entities_connect_triples(
            sampled_entities[4], sampled_entities[5],
            (sampled_entities[0], sampled_relations[0], sampled_entities[1]),
            (sampled_entities[2], sampled_relations[1], sampled_entities[3]),
        ):
            # Third antecedent must connect prior antecedents
            force_connect_components(
                [4,5], sampled_entities,
                sampled_entities[:2],
                sampled_entities[2:4],
                seed,
            )
    
    # Define antecedent
    antecedent = [
        (
            sampled_entities[0],  # e1
            sampled_relations[0],  # r1
            sampled_entities[1],  # e2
        ),
        (
            sampled_entities[2],  # e3
            sampled_relations[1],  # r2
            sampled_entities[3],  # e4
        ),
        (
            sampled_entities[4],  # e5
            sampled_relations[2],  # r3
            sampled_entities[5],  # e6
        ),
    ]
    
    # Consequence must have both of its entities in some antecedent
    if sampled_entities[6] not in sampled_entities[:6]:
        # Force consequent entity to switch to an antecedent entity
        force_swap_to_entities([6], sampled_entities, sampled_entities[:6], seed)
    if sampled_entities[7] not in sampled_entities[:6]:
        force_swap_to_entities([7], sampled_entities, sampled_entities[:6], seed)
    # Consequence must have a relation in some antecedent
    if sampled_relations[3] not in sampled_relations[:3]:
        force_swap_to_entities([3], sampled_relations, sampled_relations[:3], seed)
    
    # Define consequence
    consequence = (
        sampled_entities[6],  # e7
        sampled_relations[3],  # r4
        sampled_entities[7],  # e8
    )
    
    time_lag_tuples = create_time_lag_tuples(time_lags, antecedent)

    return TemporalPattern(
        antecedent=antecedent,
        consequence=consequence,
        time_lags=time_lag_tuples,
        n_hops=3,
    )
