# Base config
import scipy

configs = [
    # Uniform distribution over entities and relations
    {
        # KG/TKG toggle
        'graph_mode': 'tkg',

        # Path information
        # Export directory
        'export_dir': 'opt_runs_icews14_20251022_best',

        # Seed: int for reproducibility or None for random generation
        'seed': 367,

        # Debug: Controls debug print statements
        'debug': False,

        # Train-Valid-Test split
        'split': (.8, .1, .1),
        # For KG setting only: all, consequence_only
        'split_on': 'consequence_only',

        # Skip backtracking to instantiate consequences and emit them directly
        'fast_force_consequence': True,

        # Skips matching for emergent patterns (caused by pattern interactions that
        # could occur if prevent_quad_collision is False)
        'skip_generation_matcher': True,

        # Basic stats
        # Number of runs, each run will have a dedicated directory inside export_dir
        'n_runs': 3,
        # Number of jobs
        'n_jobs': 3,
        # Number of entities
        'n_ents': 7128,
        # Number of relations
        'n_rels': 230,
        # Number of time windows
        'n_tws': 365,

        # Patterns
        # Distribution over entities, determining which are used to populate pattern templates.
        # Should be a function taking a number of entities as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_ents': lambda x, seed: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x, random_state=seed),
        # Distribution over relations, determining which are used to populate pattern templates
        # Should be a function taking a number of relations as input and returning the same number
        # of weights. Defaults to uniform distribution
        'pat_distr_rels': lambda x, seed: scipy.stats.gamma.rvs(1, loc=0, scale=2, size=x, random_state=seed),
        # Whether patterns must be composed of unique triples (i.e., no antecedents or consequence
        # within the same pattern can share the same (head, relation, tail) triple)
        'require_unique_triples': True,
        # Whether patterns are prohibited from including self-connections of the form (e1, r, e1)
        'prohibit_selfconnections': False,
        # Prohibit relation in consequence from being different from all relations in the antecedents
        # True => all consequences will have relations that appear in at least one of their antecedents
        # False => consequences may have relations that never appeared in any of their antecedents
        'prohibit_new_consequence_relations': True,
        # Require sequential-ness of rule, namely any consecutive antecedents must: share at least one
        # entity (with the same restriction on the consequence and the first and last antecedents) and
        # be 2-regular. Functionally, this requires "cycle"-format rules, like TLogic's.
        'require_sequential_rule': False,
        # Prevent quadruples (h, r, o, t) from being attributed to multiple patterns. We do this by preventing
        # instantiations of patterns that would lead to such collisions, so this doesn't bias the TKG generation,
        # just produce simpler, less noisy TKGs.
        'prevent_quad_collisions': True,
        # Maximum number of times to try to resample a pattern for instantiation, to prevent quadruple collision
        'max_instantiation_resamples': 20,
        # Number of 3-hop patterns
        'n_3_hop': 100,
        # Time lag for 3-hop patterns
        'time_lag_3_hop': [
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        # Number of 2-hop patterns
        'n_2_hop': 100,
        # Time lag for 2-hop patterns
        'time_lag_2_hop': [
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
            (1, lambda seed: scipy.stats.poisson(5).rvs(1, random_state=seed)[0]),
        ],
        # Number of 1-hop patterns
        'n_1_hop': 100,
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
        # TODO Takes a *long* time to label if we allow random wiring.
        'rnd_avg_density': .0,
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
            1: .5,
            2: .5,
            3: .5,  # Should be .59 to replicate ICEWS14 average density, but simplifying things here
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
            1: None,  #lambda seed: scipy.stats.binom.rvs(n=1, p=.5, size=1, random_state=seed)[0],
            2: None,  #lambda seed: scipy.stats.binom.rvs(n=1, p=.5, size=1, random_state=seed)[0],
            3: None,  #lambda seed: scipy.stats.binom.rvs(n=1, p=.5, size=1, random_state=seed)[0],
        }
    },
]