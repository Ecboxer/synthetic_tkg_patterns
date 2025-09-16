import os
import pandas as pd

from patterns import generate_n_hop_patterns


def load_or_generate_patterns(
    n_hops: int, cache_dir: str,
    require_unique_triples: bool = True,
    prohibit_selfconnections: bool = False,
    prohibit_new_consequence_relations: bool = True,
    require_sequential_rule: bool = False,
):
    path = os.path.join(
        cache_dir,
        f"all_patterns_{n_hops}_hop_unique_{require_unique_triples}_prohibit_self_{prohibit_selfconnections}_prohibit_new_cons_rels_{prohibit_new_consequence_relations}_seq_rule_{require_sequential_rule}.xlsx"
    )
    if os.path.exists(path):
        return pd.read_excel(path)
    else:
        df = generate_n_hop_patterns(
            n_hops=n_hops,
            require_unique_triples=require_unique_triples,
            prohibit_selfconnections=prohibit_selfconnections,
            prohibit_new_consequence_relations=prohibit_new_consequence_relations,
            require_sequential_rule=require_sequential_rule,
        )
        df.to_excel(path, index=False)
        return df
