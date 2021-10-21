#!/bin/bash
#SBATCH -n 2
#SBATCH -J with_ANS
#SBATCH -p shared
#SBATCH -t 50:00:00

module load 2019
module load Python/3.6.6-intel-2019b
source ../../venv/bin/activate

# the script -c is done so that the output is printed in real time to log.txt
script -c 'python -u \
../simulation.py -n_chains 1 -l2_minimize_model False -s3_ANS_model True \
-min_picsize 10 --cores 2 -path_data ../data/\*.csv' \
log.txt
