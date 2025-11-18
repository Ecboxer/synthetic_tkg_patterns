# Synthetic TKG Patterns

## Requirements
Defined in requirements.txt

## Create a single synthetic TKG
Copy and rename `configs/config.py` (in this example we will call the copy `configs/example.py`).

Enter desired synthetic TKG hyperparameters in `configs/config.py`.

Run `python run.py -c configs/example.py`

See `run.py` for instructions on running multiple configs with a single command line call.

## Optimize for similarity to an existing TKG
Edit the TKG hyperparameter search space in `optimize_tkg.py`'s `objective_obtuna` function.

We assume the existing, reference TKG is located at path `path_ref_tkg`.

Run the following:
```
python3 optimize_tkg.py \
    --ref_tkg_dir path_ref_tkg \
    --budget N \
    --export_root opt_runs
```

This will create a directory `opt_runs` with `N` trial, synthetic TKGs and optimize for the similarity of these synthetic TKGs to the reference TKG. The best synthetic TKG will also be copied to `opt_runs/best_trial`.

## Updated optimization commands (for multi-threading)
DONE (baselines are too low in this version)
'''
export NAS_ROOT="/nas/ckgfs/users/eboxer/synthetic_tkg_patterns"
export OPTUNA_AREA="$NAS_ROOT/_sqlite_optuna"
export OPTUNA_DB="$OPTUNA_AREA/optuna.db"
export STUDY_NAME="opt_runs_icews14_20251111"

python3 init_sqlite_optuna.py "$OPTUNA_DB"
python3 create_study.py "$OPTUNA_DB" "$STUDY_NAME"

TOTAL_TRIALS=800 \
WORKERS=16 \
REF_DIR="/nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/ICEWS14" \
EXPORT_ROOT="$NAS_ROOT/$STUDY_NAME" \
./run_optuna_workers_sqlite.sh
'''

DONE ckg05
'''
export NAS_ROOT="/nas/ckgfs/users/eboxer/synthetic_tkg_patterns"
export OPTUNA_AREA="$NAS_ROOT/_sqlite_optuna"
export OPTUNA_DB="$OPTUNA_AREA/optuna.db"
export STUDY_NAME="opt_runs_icews14_20251112"

python3 init_sqlite_optuna.py "$OPTUNA_DB"
python3 create_study.py "$OPTUNA_DB" "$STUDY_NAME"

TOTAL_TRIALS=800 \
WORKERS=16 \
REF_DIR="/nas/ckgfs/users/eboxer/TKG-Forecasting-Evaluation/data/ICEWS14" \
EXPORT_ROOT="$NAS_ROOT/$STUDY_NAME" \
./run_optuna_workers_sqlite.sh
'''
