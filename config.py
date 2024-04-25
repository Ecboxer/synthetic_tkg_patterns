import scipy


config = {
    # Path information
    # Export directory
    'export_dir': 'data_debug',

    # Train-Valid-Test split
    'split': (1., 0, 0),

    # Basic stats
    # Number of runs, each run will have a dedicated directory inside export_dir
    'n_runs': 1,
    # Number of entities
    'n_ents': 10,
    # Number of relations
    'n_rels': 1,
    # Number of time windows
    'n_tws': 3,

    # Patterns
    # Distribution over entities, determining which are used to populate pattern templates.
    # Should be a function taking a number of entities as input and returning the same number
    # of weights. Defaults to uniform distribution
    'pat_distr_ents': lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=1, size=x),
    # Distribution over relations, determining which are used to populate pattern templates
    # Should be a function taking a number of relations as input and returning the same number
    # of weights. Defaults to uniform distribution
    'pat_distr_rels': lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=1, size=x),
    # Number of 3-hop patterns
    'n_3_hop': 10,
    # Time lag for 3-hop patterns
    'time_lag_3_hop': [
        (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    ],
    # Number of 2-hop patterns
    'n_2_hop': 10,
    # Time lag for 2-hop patterns
    'time_lag_2_hop': [
        (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    ],
    # Number of 1-hop patterns
    'n_1_hop': 10,
    # Time lag for 1-hop patterns
    'time_lag_1_hop': [
        (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    ],
    # Maximum number of times to search for a valid pattern to instantiate
    # before moving on
    'max_retries': 10,

    # Edge list creation
    # Density with which we randomly wire entities per window
    # Overridden by rnd_avg_density_distr if it is not None
    'rnd_avg_density': 5,
    # Function that returns an integer to be used for average density per entity
    # Setting to None will cause rnd_avg_density to be used instead
    'rnd_avg_density_distr': lambda: scipy.stats.poisson.rvs(5, size=1)[0],
    # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
    # satisfied in previous time windows)
    'p_skip_consequence': .1,
    # Probability that we create artificially create edges that validate a given pattern
    # (create edges that satisfay all antecedents), per pattern
    'n_hops2p_force': {
        1: .1,
        2: .1,
        3: .1,
    },
}