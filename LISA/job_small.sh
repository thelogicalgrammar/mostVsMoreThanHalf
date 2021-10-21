#!/bin/bash
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 00:30:00

module load 2019
module load Python/3.6.6-intel-2019b
source ../../venv/bin/activate

script -c 'python -u \
../simulation.py \
-n_chains 1 --artificial_data True -num_participants 2 -num_states 10 -min_picsize 5 -num_trials 2 --cores 1' \
log.txt
