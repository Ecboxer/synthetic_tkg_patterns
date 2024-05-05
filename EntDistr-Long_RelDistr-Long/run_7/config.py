import scipy


configs = [
    # # Uniform distribution over entities and relations
    # {
    #     # Path information
    #     # Export directory
    #     'export_dir': 'EntDistr-Unif_RelDistr-Unif0',

    #     # Train-Valid-Test split
    #     'split': (.8, .1, .1),

    #     # Basic stats
    #     # Number of runs, each run will have a dedicated directory inside export_dir
    #     'n_runs': 10,
    #     # Number of jobs
    #     'n_jobs': 10,
    #     # Number of entities
    #     'n_ents': 5_000,
    #     # Number of relations
    #     'n_rels': 200,
    #     # Number of time windows
    #     'n_tws': 365,

    #     # Patterns
    #     # Distribution over entities, determining which are used to populate pattern templates.
    #     # Should be a function taking a number of entities as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_ents': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Distribution over relations, determining which are used to populate pattern templates
    #     # Should be a function taking a number of relations as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_rels': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Number of 3-hop patterns
    #     'n_3_hop': 100,
    #     # Time lag for 3-hop patterns
    #     'time_lag_3_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 2-hop patterns
    #     'n_2_hop': 100,
    #     # Time lag for 2-hop patterns
    #     'time_lag_2_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 1-hop patterns
    #     'n_1_hop': 100,
    #     # Time lag for 1-hop patterns
    #     'time_lag_1_hop': [
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Maximum number of times to search for a valid pattern to instantiate
    #     # before moving on
    #     'max_retries': 1,

    #     # Edge list creation
    #     # Density with which we randomly wire entities per window
    #     # Overridden by rnd_avg_density_distr if it is not None
    #     'rnd_avg_density': 1,
    #     # Function that returns an integer to be used for average density per entity
    #     # Setting to None will cause rnd_avg_density to be used instead
    #     'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
    #     # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
    #     # satisfied in previous time windows)
    #     'p_skip_consequence': 0,
    #     # Probability that we create artificially create edges that validate a given pattern
    #     # (create edges that satisfay all antecedents), per pattern
    #     'n_hops2p_force': {
    #         1: .1,
    #         2: .1,
    #         3: .1,
    #     },
    # },
    # # Uniform distribution over entities, long-tail (gamma) distribution over relations
    # {
    #     # Path information
    #     # Export directory
    #     'export_dir': 'EntDistr-Unif_RelDistr-Long',

    #     # Train-Valid-Test split
    #     'split': (.8, .1, .1),

    #     # Basic stats
    #     # Number of runs, each run will have a dedicated directory inside export_dir
    #     'n_runs': 10,
    #     # Number of jobs
    #     'n_jobs': 10,
    #     # Number of entities
    #     'n_ents': 5_000,
    #     # Number of relations
    #     'n_rels': 200,
    #     # Number of time windows
    #     'n_tws': 365,

    #     # Patterns
    #     # Distribution over entities, determining which are used to populate pattern templates.
    #     # Should be a function taking a number of entities as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_ents': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Distribution over relations, determining which are used to populate pattern templates
    #     # Should be a function taking a number of relations as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_rels': lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Number of 3-hop patterns
    #     'n_3_hop': 100,
    #     # Time lag for 3-hop patterns
    #     'time_lag_3_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 2-hop patterns
    #     'n_2_hop': 100,
    #     # Time lag for 2-hop patterns
    #     'time_lag_2_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 1-hop patterns
    #     'n_1_hop': 100,
    #     # Time lag for 1-hop patterns
    #     'time_lag_1_hop': [
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Maximum number of times to search for a valid pattern to instantiate
    #     # before moving on
    #     'max_retries': 1,

    #     # Edge list creation
    #     # Density with which we randomly wire entities per window
    #     # Overridden by rnd_avg_density_distr if it is not None
    #     'rnd_avg_density': 1,
    #     # Function that returns an integer to be used for average density per entity
    #     # Setting to None will cause rnd_avg_density to be used instead
    #     'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
    #     # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
    #     # satisfied in previous time windows)
    #     'p_skip_consequence': 0,
    #     # Probability that we create artificially create edges that validate a given pattern
    #     # (create edges that satisfay all antecedents), per pattern
    #     'n_hops2p_force': {
    #         1: .1,
    #         2: .1,
    #         3: .1,
    #     },
    # },
    # Long-tail (gamma) distribution over entities, uniform distribution over relations
    {
        # Path information
        # Export directory
        'export_dir': 'EntDistr-Long_RelDistr-Unif',

        # Train-Valid-Test split
        'split': (.8, .1, .1),

        # Basic stats
        # Number of runs, each run will have a dedicated directory inside export_dir
        'n_runs': 10,
        # Number of jobs
        'n_jobs': 10,
        # Number of entities
        'n_ents': 5_000,
        # Number of relations
        'n_rels': 200,
        # Number of time windows
        'n_tws': 365,

        # Patterns
        # Distribution over entities, determining which are used to populate pattern templates.
        # Should be a function taking a number of entities as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_ents': lambda x: scipy.stats.gamma.rvs(.1, loc=0, scale=10, size=x),
        # Distribution over relations, determining which are used to populate pattern templates
        # Should be a function taking a number of relations as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_rels': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
        # Number of 3-hop patterns
        'n_3_hop': 100,
        # Time lag for 3-hop patterns
        'time_lag_3_hop': [
            (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
            (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
            (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        ],
        # Number of 2-hop patterns
        'n_2_hop': 100,
        # Time lag for 2-hop patterns
        'time_lag_2_hop': [
            (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
            (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        ],
        # Number of 1-hop patterns
        'n_1_hop': 100,
        # Time lag for 1-hop patterns
        'time_lag_1_hop': [
            (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        ],
        # Maximum number of times to search for a valid pattern to instantiate
        # before moving on
        'max_retries': 1,

        # Edge list creation
        # Density with which we randomly wire entities per window
        # Overridden by rnd_avg_density_distr if it is not None
        'rnd_avg_density': 1,
        # Function that returns an integer to be used for average density per entity
        # Setting to None will cause rnd_avg_density to be used instead
        'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
        # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
        # satisfied in previous time windows)
        'p_skip_consequence': 0,
        # Probability that we create artificially create edges that validate a given pattern
        # (create edges that satisfay all antecedents), per pattern
        'n_hops2p_force': {
            1: .1,
            2: .1,
            3: .1,
        },
    },
    # Long-tail (gamma) distribution over entities, long-tail (gamma) distribution over relations
    {
        # Path information
        # Export directory
        'export_dir': 'EntDistr-Long_RelDistr-Long',

        # Train-Valid-Test split
        'split': (.8, .1, .1),

        # Basic stats
        # Number of runs, each run will have a dedicated directory inside export_dir
        'n_runs': 10,
        # Number of jobs
        'n_jobs': 10,
        # Number of entities
        'n_ents': 5_000,
        # Number of relations
        'n_rels': 200,
        # Number of time windows
        'n_tws': 365,

        # Patterns
        # Distribution over entities, determining which are used to populate pattern templates.
        # Should be a function taking a number of entities as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_ents': lambda x: scipy.stats.gamma.rvs(.1, loc=0, scale=10, size=x),
        # Distribution over relations, determining which are used to populate pattern templates
        # Should be a function taking a number of relations as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_rels': lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
        # Number of 3-hop patterns
        'n_3_hop': 100,
        # Time lag for 3-hop patterns
        'time_lag_3_hop': [
            (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
            (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
            (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        ],
        # Number of 2-hop patterns
        'n_2_hop': 100,
        # Time lag for 2-hop patterns
        'time_lag_2_hop': [
            (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
            (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        ],
        # Number of 1-hop patterns
        'n_1_hop': 100,
        # Time lag for 1-hop patterns
        'time_lag_1_hop': [
            (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
        ],
        # Maximum number of times to search for a valid pattern to instantiate
        # before moving on
        'max_retries': 1,

        # Edge list creation
        # Density with which we randomly wire entities per window
        # Overridden by rnd_avg_density_distr if it is not None
        'rnd_avg_density': 1,
        # Function that returns an integer to be used for average density per entity
        # Setting to None will cause rnd_avg_density to be used instead
        'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
        # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
        # satisfied in previous time windows)
        'p_skip_consequence': 0,
        # Probability that we create artificially create edges that validate a given pattern
        # (create edges that satisfay all antecedents), per pattern
        'n_hops2p_force': {
            1: .1,
            2: .1,
            3: .1,
        },
    },
    # # More 1-hop than 3-hop patterns (4x)
    # {
    #     # Path information
    #     # Export directory
    #     'export_dir': '1hop-400_2hop-100_3hop-25',

    #     # Train-Valid-Test split
    #     'split': (.8, .1, .1),

    #     # Basic stats
    #     # Number of runs, each run will have a dedicated directory inside export_dir
    #     'n_runs': 10,
    #     # Number of jobs
    #     'n_jobs': 10,
    #     # Number of entities
    #     'n_ents': 5_000,
    #     # Number of relations
    #     'n_rels': 200,
    #     # Number of time windows
    #     'n_tws': 365,

    #     # Patterns
    #     # Distribution over entities, determining which are used to populate pattern templates.
    #     # Should be a function taking a number of entities as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_ents': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Distribution over relations, determining which are used to populate pattern templates
    #     # Should be a function taking a number of relations as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_rels': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Number of 3-hop patterns
    #     'n_3_hop': 25,
    #     # Time lag for 3-hop patterns
    #     'time_lag_3_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 2-hop patterns
    #     'n_2_hop': 100,
    #     # Time lag for 2-hop patterns
    #     'time_lag_2_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 1-hop patterns
    #     'n_1_hop': 400,
    #     # Time lag for 1-hop patterns
    #     'time_lag_1_hop': [
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Maximum number of times to search for a valid pattern to instantiate
    #     # before moving on
    #     'max_retries': 1,

    #     # Edge list creation
    #     # Density with which we randomly wire entities per window
    #     # Overridden by rnd_avg_density_distr if it is not None
    #     'rnd_avg_density': 1,
    #     # Function that returns an integer to be used for average density per entity
    #     # Setting to None will cause rnd_avg_density to be used instead
    #     'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
    #     # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
    #     # satisfied in previous time windows)
    #     'p_skip_consequence': 0,
    #     # Probability that we create artificially create edges that validate a given pattern
    #     # (create edges that satisfay all antecedents), per pattern
    #     'n_hops2p_force': {
    #         1: .4,
    #         2: .1,
    #         3: .025,
    #     },
    # },
    # # More 1-hop than 3-hop patterns (2x)
    # {
    #     # Path information
    #     # Export directory
    #     'export_dir': '1hop-200_2hop-100_3hop-50',

    #     # Train-Valid-Test split
    #     'split': (.8, .1, .1),

    #     # Basic stats
    #     # Number of runs, each run will have a dedicated directory inside export_dir
    #     'n_runs': 10,
    #     # Number of jobs
    #     'n_jobs': 10,
    #     # Number of entities
    #     'n_ents': 5_000,
    #     # Number of relations
    #     'n_rels': 200,
    #     # Number of time windows
    #     'n_tws': 365,

    #     # Patterns
    #     # Distribution over entities, determining which are used to populate pattern templates.
    #     # Should be a function taking a number of entities as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_ents': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Distribution over relations, determining which are used to populate pattern templates
    #     # Should be a function taking a number of relations as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_rels': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Number of 3-hop patterns
    #     'n_3_hop': 50,
    #     # Time lag for 3-hop patterns
    #     'time_lag_3_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 2-hop patterns
    #     'n_2_hop': 100,
    #     # Time lag for 2-hop patterns
    #     'time_lag_2_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 1-hop patterns
    #     'n_1_hop': 200,
    #     # Time lag for 1-hop patterns
    #     'time_lag_1_hop': [
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Maximum number of times to search for a valid pattern to instantiate
    #     # before moving on
    #     'max_retries': 1,

    #     # Edge list creation
    #     # Density with which we randomly wire entities per window
    #     # Overridden by rnd_avg_density_distr if it is not None
    #     'rnd_avg_density': 1,
    #     # Function that returns an integer to be used for average density per entity
    #     # Setting to None will cause rnd_avg_density to be used instead
    #     'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
    #     # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
    #     # satisfied in previous time windows)
    #     'p_skip_consequence': 0,
    #     # Probability that we create artificially create edges that validate a given pattern
    #     # (create edges that satisfay all antecedents), per pattern
    #     'n_hops2p_force': {
    #         1: .2,
    #         2: .1,
    #         3: .05,
    #     },
    # },
    # # More 3-hop than 1-hop patterns (2x)
    # {
    #     # Path information
    #     # Export directory
    #     'export_dir': '1hop-50_2hop-100_3hop-200',

    #     # Train-Valid-Test split
    #     'split': (.8, .1, .1),

    #     # Basic stats
    #     # Number of runs, each run will have a dedicated directory inside export_dir
    #     'n_runs': 10,
    #     # Number of jobs
    #     'n_jobs': 10,
    #     # Number of entities
    #     'n_ents': 5_000,
    #     # Number of relations
    #     'n_rels': 200,
    #     # Number of time windows
    #     'n_tws': 365,

    #     # Patterns
    #     # Distribution over entities, determining which are used to populate pattern templates.
    #     # Should be a function taking a number of entities as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_ents': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Distribution over relations, determining which are used to populate pattern templates
    #     # Should be a function taking a number of relations as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_rels': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Number of 3-hop patterns
    #     'n_3_hop': 200,
    #     # Time lag for 3-hop patterns
    #     'time_lag_3_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 2-hop patterns
    #     'n_2_hop': 100,
    #     # Time lag for 2-hop patterns
    #     'time_lag_2_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 1-hop patterns
    #     'n_1_hop': 50,
    #     # Time lag for 1-hop patterns
    #     'time_lag_1_hop': [
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Maximum number of times to search for a valid pattern to instantiate
    #     # before moving on
    #     'max_retries': 1,

    #     # Edge list creation
    #     # Density with which we randomly wire entities per window
    #     # Overridden by rnd_avg_density_distr if it is not None
    #     'rnd_avg_density': 1,
    #     # Function that returns an integer to be used for average density per entity
    #     # Setting to None will cause rnd_avg_density to be used instead
    #     'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
    #     # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
    #     # satisfied in previous time windows)
    #     'p_skip_consequence': 0,
    #     # Probability that we create artificially create edges that validate a given pattern
    #     # (create edges that satisfay all antecedents), per pattern
    #     'n_hops2p_force': {
    #         1: .05,
    #         2: .1,
    #         3: .2,
    #     },
    # },
    # # More 3-hop than 1-hop patterns (4x)
    # {
    #     # Path information
    #     # Export directory
    #     'export_dir': '1hop-25_2hop-100_3hop-400',

    #     # Train-Valid-Test split
    #     'split': (.8, .1, .1),

    #     # Basic stats
    #     # Number of runs, each run will have a dedicated directory inside export_dir
    #     'n_runs': 10,
    #     # Number of jobs
    #     'n_jobs': 10,
    #     # Number of entities
    #     'n_ents': 5_000,
    #     # Number of relations
    #     'n_rels': 200,
    #     # Number of time windows
    #     'n_tws': 365,

    #     # Patterns
    #     # Distribution over entities, determining which are used to populate pattern templates.
    #     # Should be a function taking a number of entities as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_ents': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Distribution over relations, determining which are used to populate pattern templates
    #     # Should be a function taking a number of relations as input and returning the same number
    #     # of weights. Defaults to uniform distribution
    #     'pat_distr_rels': None,  #lambda x: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x),
    #     # Number of 3-hop patterns
    #     'n_3_hop': 400,
    #     # Time lag for 3-hop patterns
    #     'time_lag_3_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 2-hop patterns
    #     'n_2_hop': 100,
    #     # Time lag for 2-hop patterns
    #     'time_lag_2_hop': [
    #         (0, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Number of 1-hop patterns
    #     'n_1_hop': 25,
    #     # Time lag for 1-hop patterns
    #     'time_lag_1_hop': [
    #         (1, lambda: scipy.stats.poisson(5).rvs(1)[0]),
    #     ],
    #     # Maximum number of times to search for a valid pattern to instantiate
    #     # before moving on
    #     'max_retries': 1,

    #     # Edge list creation
    #     # Density with which we randomly wire entities per window
    #     # Overridden by rnd_avg_density_distr if it is not None
    #     'rnd_avg_density': 1,
    #     # Function that returns an integer to be used for average density per entity
    #     # Setting to None will cause rnd_avg_density to be used instead
    #     'rnd_avg_density_distr': None,  #lambda: scipy.stats.poisson.rvs(1, size=1)[0],
    #     # Probability that we do not apply a given pattern, per valid pattern (with all antecedents
    #     # satisfied in previous time windows)
    #     'p_skip_consequence': 0,
    #     # Probability that we create artificially create edges that validate a given pattern
    #     # (create edges that satisfay all antecedents), per pattern
    #     'n_hops2p_force': {
    #         1: .025,
    #         2: .1,
    #         3: .4,
    #     },
    # },
]