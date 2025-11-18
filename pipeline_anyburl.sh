#!/usr/bin/env bash
python run_anyburl_all.py data_static_small_20251028_prepared \
  --threads 24 --time-sec 300 --java-heap 12G --clobber
python run_anyburl_all.py data_static_sequential_small_20251028_prepared \
  --threads 24 --time-sec 300 --java-heap 12G --clobber
python run_anyburl_all.py data_static_medium_20251028_prepared \
  --threads 24 --time-sec 300 --java-heap 12G --clobber
python run_anyburl_all.py data_static_sequential_medium_20251028_prepared \
  --threads 24 --time-sec 300 --java-heap 12G --clobber
python run_anyburl_all.py data_static_large_20251028_prepared \
  --threads 24 --time-sec 300 --java-heap 12G --clobber
python run_anyburl_all.py data_static_sequential_large_20251028_prepared \
  --threads 24 --time-sec 300 --java-heap 12G --clobber
