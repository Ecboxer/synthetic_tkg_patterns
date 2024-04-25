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
        - e3 or e4 \in {e1, e2}
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
    # Consquence must satisfy the constraint that at least one of its entities is in
    # some antecedent
    if ~entities_intersect(sampled_entities[2:4], sampled_entities[:2]):
        # Force at least of the consequent entities to switch to an antecedent entity
        force_swap_to_entities([2,3], sampled_entities, sampled_entities[:2], seed)
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
        - (e3 or e4 \in {e1, e2}) and (e5 or e6 \in {e1, e2, e3, e4}) OR
        - ~(e3 and e4 \in {e1, e2}) and ((e5 \in {e1, e2} and e6 \in {e3, e4}) or
            (e5 \in {e3, e4} and e6 \in {e1, e2}))
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
    # Second antecedent determines how we define the consequence
    if entities_intersect(sampled_entities[2:4], sampled_entities[:2]):
        # At least one entity intersects with the first antecedent
        # Consquence must have at least one of its entities in some antecedent
        if ~entities_intersect(sampled_entities[4:6], sampled_entities[:4]):
            # Enforce that one or more consequence entity be chosen from the antecedents
            force_swap_to_entities([4,5], sampled_entities, sampled_entities[:4], seed)
    else:
        # Non-intersecting first and second antecedents
        # Consequence must connect them
        if not entities_connect_triples(
            sampled_entities[4], sampled_entities[5], antecedent[0], antecedent[1]
        ):
            # Enforce that the consequence connects the two antecedents
            force_connect_components(
                [4,5], sampled_entities, sampled_entities[:2], sampled_entities[2:4], seed
            )
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
        - All antecedents intersect: (e3 or e4 \in {e1, e2}) and (e5 or e6 \in {e1, e2, e3, e4})
            and (e7 or e8 \in {e1, e2, e3, e4, e5, e6})
        - Second antecedent does not intersect first: ~(e3 and e4 \in {e1, e2})
            - Third antecedent connects them: ((e5 \in {e1, e2} and e6 \in {e3, e4}) or
                (e5 \in {e3, e4} and e6 \in {e1, e2})) then
                (e7 or e8 \in {e1, e2, e3, e4, e5, e6}),
            - Third antecedent connects to first antecedent: (e5 or e6 \in {e1, e2}) then
                (e7 \in {e1, e2, e5, e6} and e8 \in {e3, e4})) or (e7 \in {e3, e4} and
                e8 \in {e1, e2, e5, e6})), or
            - Third antecedent connects to second antecedent: (e5 or e6 \in {e3, e4}) then
                (e7 \in {e1, e2} and e8 \in {e3, e4, e5, e6})) or (e7 \in {e3, e4, e5, e6}
                and e8 \in {e1, e2}))
        - Third antecedent does not intersect first two: (e3 or e4 \in {e1, e2}) and
            ~(e5 and e6 \in {e1, e2, e3, e4}) and ((e7 \in {e1, e2, e3, e4} and e8 \in {e5, e6})
            or (e7 \in {e5, e6} and e8 \in {e1, e2, e3, e4}))
        - t4 > t3 >= t2 >= t1
        - (e1, r1, e2, t1) != (e3, r2, e4, t2) and (e1, r1, e2, t1) != (e5, r3, e6, t3) and
            (e1, r1, e2, t1) != (e3, r2, e4, t2)

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
    # Third antecedent must intersect at least one prior antecedent
    if ~entities_intersect(sampled_entities[:4], sampled_entities[4:6]):
        # Enforce that the third antecedent include entities from at least one prior
        # antecedent
        force_swap_to_entities([4,5], sampled_entities, sampled_entities[:4], seed)
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
    # Second and third antecedents determines how we define the consequence
    if entities_intersect(sampled_entities[2:4], sampled_entities[:2]) & \
        entities_intersect(sampled_entities[4:6], sampled_entities[:4]):
        # Antecedents are connected
        # Consquence must have at least one of its entities in some antecedent
        if (sampled_entities[6] not in sampled_entities[:6]) & \
            (sampled_entities[7] not in sampled_entities[:6]):
            # Enforce that one or more consequence entity be chosen from the antecedents
            force_swap_to_entities([6,7], sampled_entities, sampled_entities[:6], seed)
    elif ~entities_intersect(sampled_entities[2:4], sampled_entities[:2]):
        # Second antecedent does not intersect first antecedent
        # Determine conditions for consequence based on how the third antecedent intersects
        # the prior antecedents
        if entities_connect_triples(
            sampled_entities[4], sampled_entities[5], antecedent[0], antecedent[1]
        ):
            # Third antecedent connects prior antecedents
            # Consquence must have at least one of its entities in some antecedent
            if ~entities_intersect(sampled_entities[6:8], sampled_entities[:6]):
                # Enforce that one or more consequence entity be chosen from the antecedents
                force_swap_to_entities([6,7], sampled_entities, sampled_entities[:6], seed)
        elif entities_intersect(sampled_entities[4:6], sampled_entities[:2]):
            # Third antecedent connects with first antecedent
            # Consequence must connect antecedents
            if not entities_connect_components(
                sampled_entities[6], sampled_entities[7],
                [antecedent[0], antecedent[2]], [antecedent[1]]
            ):
                # Enforce that the consequence connects the disconnected antecedents
                force_connect_components(
                    [6,7], sampled_entities,
                    sampled_entities[:2]+sampled_entities[4:6],
                    sampled_entities[2:4],
                    seed,
                )
        else:
            # Third antecedent connects with second antecedent
            # Consequence must connect antecedents
            if not entities_connect_components(
                sampled_entities[6], sampled_entities[7],
                [antecedent[0]], [antecedent[1], antecedent[2]]
            ):
                # Enforce that the consequence connects the disconnected antecedents
                force_connect_components(
                    [6,7], sampled_entities,
                    sampled_entities[:2],
                    sampled_entities[2:6],
                    seed,
                )
    else:
        # Third antecedent does not intersect prior intersecting antecedents
        # Consequence must connect them
        if not entities_connect_components(
            sampled_entities[6], sampled_entities[7],
            [antecedent[0], antecedent[1]], [antecedent[2]]
        ):
            # Enforce that the consequence connects the disconnected antecedents
            force_connect_components(
                [6,7], sampled_entities, sampled_entities[:4], sampled_entities[4:6], seed,
            )

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
