# An alternatives explanation of 'most' and 'more than half'

This repo contains the code for model fitting of experimental data on quantifier production. 
This code is used in the paper "An alternatives account of 'most' and 'more than half'" by Fausto Carcassi and Jakub Szymanik.

## Repo content

- `run_analysis.cmd`: Run analysis on windows.
- `analysis_functions.py`: Contains various functions for plotting and model comparison. This is meant to be used on the fitted model.
- `simulation.py`: Functions to pre-process data and fit the model.
- `RSA_theano_model.py`: A theano implementation of the RSA model described in the paper. Used in `simulation.py` for model fitting.
- `tests_on_model_fit.cmd`: Run tests of model fitting on windows.
- `data`: Contains the anonymized data of the experiment presented in the paper.
- `LISA`: Contains various bash scripts to run the model fitting on a server.
- `analysis_plots`: Contains various plots of posterior parameters for each participant for the two models.
- `LISA_analysis`: Contains various bash scripts to run the analyses of the fitted models on a server.

A good starting point to undestand the code is to read through `simulation.py`. 

## Authors

- Fausto Carcassi - _First author_
- Jakub Szymanik - _Second author_