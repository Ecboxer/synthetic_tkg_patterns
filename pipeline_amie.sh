#!/usr/bin/env bash
python run_amie_all.py data_static_small_20251028_prepared data_static_sequential_small_20251028_prepared \
  --java-heap 8G --mins 5 --minhc 0.05 --minpca 0.05 --maxad 3 --const --maxadc 1 \
  --datalog --oute --log-help --clobber --sep tab
python run_amie_all.py data_static_medium_20251028_prepared data_static_sequential_medium_20251028_prepared \
  --java-heap 8G --mins 5 --minhc 0.05 --minpca 0.05 --maxad 3 --const --maxadc 1 \
  --datalog --oute --log-help --clobber --sep tab
python run_amie_all.py data_static_large_20251028_prepared data_static_sequential_large_20251028_prepared \
  --java-heap 8G --mins 5 --minhc 0.05 --minpca 0.05 --maxad 3 --const --maxadc 1 \
  --datalog --oute --log-help --clobber --sep tab
