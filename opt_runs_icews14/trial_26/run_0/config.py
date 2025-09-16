import scipy


configs = [
    # Uniform distribution over entities and relations
    {
        # Path information
        # Export directory
        'export_dir': 'data_test_refactor',

        # Seed: int for reproducibility or None for random generation
        'seed': 0,

        # Debug: Controls debug print statements
        'debug': True,

        # Train-Valid-Test split
        'split': (.8, .1, .1),

        # Basic stats
        # Number of runs, each run will have a dedicated directory inside export_dir
        'n_runs': 1,
        # Number of jobs
        'n_jobs': 1,
        # Number of entities
        'n_ents': 50,
        # Number of relations
        'n_rels': 50,
        # Number of time windows
        'n_tws': 30,

        # Patterns
        # Distribution over entities, determining which are used to populate pattern templates.
        # Should be a function taking a number of entities as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_ents': None,  #lambda x, seed: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x, random_state=seed),
        # Distribution over relations, determining which are used to populate pattern templates
        # Should be a function taking a number of relations as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_rels': None,  #lambda x, seed: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x, random_state=seed),
        # Whether patterns must be composed of unique triples (i.e., no antecedents or consequence
        # within the same pattern can share the same (head, relation, tail) triple)
        'require_unique_triples': False,
        # Whether patterns are prohibited from including self-connections of the form (e1, r, e1)
        'prohibit_selfconnections': False,
        # Prohibit relation in consequence from being different from all relations in the antecedents
        # True => all consequences will have relations that appear in at least one of their antecedents
        # False => consequences may have relations that never appeared in any of their antecedents
        'prohibit_new_consequence_relations': True,
        # Require sequential-ness of rule, namely any consecutive antecedents must share at least one
        # entity (with the same restriction on the consequence and the first and last antecedents).
        # Functionally, this requires "cycle"-format rules, like TLogic's
        'require_sequential_rule': False,
        # Number of 3-hop patterns
        'n_3_hop': 10,
        # Time lag for 3-hop patterns
        'time_lag_3_hop': [
            (0, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
            (0, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        # Number of 2-hop patterns
        'n_2_hop': 10,
        # Time lag for 2-hop patterns
        'time_lag_2_hop': [
            (0, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        # Number of 1-hop patterns
        'n_1_hop': 10,
        # Time lag for 1-hop patterns
        'time_lag_1_hop': [
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        # Maximum number of times to search for a valid pattern to instantiate
        # before moving on
        'max_retries': 3,

        # Edge list creation
        # Density with which we randomly wire entities per window (uniformly at random)
        # Overridden by rnd_avg_density_distr if it is not None
        'rnd_avg_density': .25,
        # Function that returns an integer to be used for average density per entity
        # Setting to None will cause rnd_avg_density to be used instead
        'rnd_avg_density_distr': None,  #lambda seed: scipy.stats.poisson.rvs(1, size=1, random_state=seed)[0],
        # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
        # satisfied in previous time windows)
        # TODO: For non-zero values, needs to be updated to account for multiple instantiations per pattern, per time window
        'p_skip_consequence': 0,
        # Probability that we create artificially create edges that validate a given pattern
        # (create edges that satisfay all antecedents), per pattern
        # Note: n_hops2p_force and n_hops2n_force are overruled if n_hops2n_force_distr is not None
        'n_hops2p_force': {
            1: 1,  #.08,
            2: 1,  #.08,
            3: 1,  #.08,
        },
        # Number of attempts to force a given pattern
        'n_hops2n_force': {
            1: 1,
            2: 1,
            3: 1,
        },
        # Mapping from number of hops to a distribution of number of instantiations to force
        # Note: Overrules n_hops2p_force and n_hops2n_force if not None
        'n_hops2n_force_distr': {
            1: lambda seed: scipy.stats.binom.rvs(n=1, p=.5, size=1, random_state=seed)[0],
            2: lambda seed: scipy.stats.binom.rvs(n=1, p=.5, size=1, random_state=seed)[0],
            3: lambda seed: scipy.stats.binom.rvs(n=1, p=.5, size=1, random_state=seed)[0],
        }
    },
]