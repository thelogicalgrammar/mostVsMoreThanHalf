#!/bin/bash
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 5:00:00

module load 2019
module load Python/3.6.6-intel-2019b
source ../../venv/bin/activate

# The double quotes in "$1" ensures that the variable is expanded but the asterisk isn't, in case I use glob.
echo "In job_single_participant: $1"
script -c "python -u \
../simulation.py \
-n_chains 2 -min_picsize 10 -path_data $1 --cores 1" \
log.txt
