# GCG Bulk Attack

Attack model for set of prompts to compare against orthogonalised model's performance.

## Basic run with default settings
python run_gcg_attacks.py

See flags for custom configurations. Some potentially useful options:

## Specify number of GPUs and memory constraints
python run_gcg_attacks.py --num-gpus 4 --model-memory-upper-bound-gb 20 --gpu-memory-lower-bound-gb 90

## Use probe sampling and early stopping
python run_gcg_attacks.py --probe-sampling --early-stop

## Run on just 10 samples
python run_gcg_attacks.py --num-samples 10