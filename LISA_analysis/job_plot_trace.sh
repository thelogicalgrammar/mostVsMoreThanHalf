#!/bin/bash
#SBATCH -n 1
#SBATCH -J plot_trace_model
#SBATCH -p shared
#SBATCH -t 1:00:00

module load 2019
module load Python/3.6.6-intel-2019b
source ../../venv/bin/activate

python ../analysis_functions.py \
--path_fitted ../LISA/model_saved/model_output_without_structural_account.pickle \
--analysis_type plot_trace_model \
--path_save ../analysis_plots/hierarchical_fits_without_structural_account \
--name_add without_structural_account \
--InferenceData_path ./InferenceData_without_structural_account.pickle
