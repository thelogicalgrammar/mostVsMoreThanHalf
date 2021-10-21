:: python simulation.py -n_chains 1 -l2_minimize_model False -min_picsize 10 --cores 1 -path_data "data_10_participants/\*.csv"

:: python simulation.py --artificial_data True -num_participants 1 -num_states 10 -num_trials 3 --cores 1 -l2_minimize_artificial True -l2_minimize_model False

:: python simulation.py --artificial_data True -num_participants 3 -num_states 10 -num_trials 3 --cores 1 -s3_ANS_model False -l2_minimize_model False

:: python simulation.py --artificial_data True -num_participants 3 -num_states 10 -num_trials 3 --cores 1 -l2_minimize_model False -alternatives_account False

python simulation.py -n_chains 1 -l2_minimize_model False -min_picsize 10 --cores 1 -path_data data_10_participants/\*.csv
