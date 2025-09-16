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