import pymc3 as pm
import theano as tt
import theano.tensor as T
import matplotlib.pyplot as plt
from RSA_theano_model import theano_RSA

from glob import glob
import pandas as pd
from pprint import pprint

import seaborn as sns
import numpy as np

import scipy.stats as stats
import argparse
import os
import pickle

from warnings import warn


def produce_experiment_language(num_states, alternatives_account=True):
    """
    Produce various information concerning the language used by participants in experiment
    The important thing about this function is that it depends on data only through states,
    which is the maximum number of states. Conceptually convenient to separate from data.

    Parameters
    ----------
    num_states: int
        Number of states (including 0). basically, the maximum number of objects in any picture + 1
    many_signals: Bool
        If true, then the function considers as many signals as the experiment. 
        Otherwise considers a subset of the signals.
    Returns
    ------
    tuple
        Various information concerning the language
    """

    # proportions_array has shape (# states, # states)
    proportions_arrays = [
        # need to add 1 because there's always 1 state more than there
        # are objects (namely, the state where the proportion is 0)
        np.linspace(0, 1, num_objects+1) 
        for num_objects in np.arange(num_states)
    ]

    #### for each signal, specify 
    # name, array, cost, type
    # make sure that the name is the same as the name recorded in the experimental data
    #### signals in experiment
    # for now, wrt types:
    #     1=upward monotonic, 2=point, 3=downward monotonic, 4=other
    
    # NOTE: when the cost is not used,
    # the types encode the "level of complexity" 
    # rather than monotonicity profile

    # NOTE: the names of the real signals need to be the same
    # as their names in the experimental data CSV.
    # That's how the model knows which signal in the experiment
    # corresponds to which of the signals modelled here
    silence = (
        'Silence',
        [np.ones(shape=p.shape) for p in proportions_arrays],
        0, 0)#4)
    most = (
        'Most',
        [np.full(p.shape,-1) for p in proportions_arrays],
        1, 1)#1)
    most = (
        'Most',
        [p > 0.5 for p in proportions_arrays],
        1, 1)#1)
    mth = (
        'More than half',
        [p > 0.5 for p in proportions_arrays],
        3, 2)#1)
    every = (
        'All',
        [p == 1. for p in proportions_arrays],
        1, 1)#1)
    # approximate "half" for the cases where there is no perfect mid?
    half = (
        'Half',
        [np.abs(p-0.5) == np.min(np.abs(p-0.5)) 
            if p.size!=0 else np.array([]) for p in proportions_arrays],
        1, 1)#1)
    many = (
        'Many',
        [p > 0.4 for p in proportions_arrays],
        1, 1)#1)
    none = (
        'None',
        [p == 0. for p in proportions_arrays],
        1, 1)#3)
    lth = (
        'Less than half',
        [p < 0.5 for p in proportions_arrays],
        3, 2)#3)
    few = (
        'Few',
        [p < 0.2 for p in proportions_arrays],
        1, 1)#3)
    some = (
        'Some',
        [p > 0. for p in proportions_arrays],
        1, 1)#1)

    # signals that I am assuming the speaking thinks of the 
    # listener thinking about, but are not real options in the experiment
    more_than_1_3rds = (
        'more_than_1_3rds',
        [p > 0.33 for p in proportions_arrays],
        3, 2)#1)
    more_than_2_3rds = (
        'more_than_2_3rds',
        [p > 0.66 for p in proportions_arrays],
        3, 2)#1)
    less_than_1_3rds = (
        'less_than_1_3rds',
        [p < 0.33 for p in proportions_arrays],
        3, 2)#3)
    less_than_2_3rds = (
        'less_than_2_3rds',
        [p < 0.66 for p in proportions_arrays],
        3, 2)#3)

    more_than_1_4ths = (
        'more_than_1_4rds',
        [p > 0.25 for p in proportions_arrays],
        3, 2)#1)
    more_than_3_4ths = (
        'more_than_3_4rds',
        [p > 0.75 for p in proportions_arrays],
        3, 2)#1)
    less_than_1_4ths = (
        'less_than_1_4rds',
        [p < 0.25 for p in proportions_arrays],
        3, 2)#1)
    less_than_3_4ths = (
        'less_than_3_4rds',
        [p < 0.75 for p in proportions_arrays],
        3, 2)#1)
    # l
    if alternatives_account:
        signals = [
            most, mth, every, half, many, none, lth, few, some, 
            more_than_1_3rds, more_than_2_3rds,
            less_than_1_3rds, less_than_2_3rds,
            more_than_1_4ths, more_than_3_4ths,
            less_than_1_4ths, less_than_3_4ths
        ]
        types = np.array([s[3] for s in signals])
    else:
        signals = [most, mth, every, half, many, none, lth, few, some]
        # if not alternatives_account, all signals compete with each other
        types = np.array([1 for s in signals])

    real_signals_indices = np.arange(9).astype(int)
    names = [signal[0] for signal in signals]
    # signals_language_array for each picsize has an array 
    # with shape (# signals, picsize)
    # print([a for a in list(zip([s[1] for s in signals]))])
    possible_signals_array = [
        np.array(a).astype(int) 
        for a in list(zip(*[s[1] for s in signals]))
    ]
    costs_possible = np.array([s[2] for s in signals])

    # (real signals)
    costs_real = costs_possible[real_signals_indices]

    # (real signals, possible signals)
    at_most_as_costly = (
            (costs_real.reshape(-1,1) - costs_possible) >= 0).astype(int)

    return possible_signals_array, real_signals_indices, costs_possible, at_most_as_costly, types, names


def create_distance(num_states):
    """
    Parameters
    ----------
    num_states: int
        shape: (states, states)
        
    """
    a = np.tile(np.arange(num_states), reps=(num_states,1))
    distances = np.abs(a - np.arange(num_states).reshape(-1,1))
    return distances


def get_data(path="./data/*.csv", cap_picsize=np.inf, min_picsize=0, **args):
    data_csvs = []
    for participant_idx, single_path in enumerate(glob(path)):
        print(single_path)
        data_csvs.append(
            initial_filter_df(
                pd.read_csv(single_path),
                participant_idx,
                cap_picsize=cap_picsize,
                min_picsize=min_picsize,
                **args
            )
        )
    data = pd.concat(data_csvs)
    return data


def preprocess_data(data, alternatives_account):
    """
    Parameters
    ----------
    data: array
        Experimental data
    Returns
    -------
    Tuple
        A bunch of data containing information needed for model fit
    """

    # breakpoint()
    num_participants = data["id_index"].max() + 1
    num_states = data["total"].max() + 1

    (possible_signals_array, real_signals_indices, costs_possible,
        at_most_as_costly, types, names) = produce_experiment_language(
        num_states,
        alternatives_account
    )

    distances = create_distance(num_states)

    # Create arrays encoding experimental data

    # the index of the signal picked by each participant in each datapoint
    picked_signals_indices = [names.index(n) for n in data["choice"]]

    # all the following arrays have shape (# participants * # trials)

    # contains the participant's index for each datapoint
    participants_indices = data["id_index"].values
    # the # of objects in the picture seen for each datapoint
    picsizes_values = data["total"].values
    # the size of the target set for each datapoint
    states_values = data["target"].values

    return {
        "num_participants": num_participants, 
        "num_states": num_states, 
        "possible_signals_array": possible_signals_array, 
        "real_signals_indices": real_signals_indices, 
        "costs_possible": costs_possible, 
        "at_most_as_costly": at_most_as_costly, 
        "types": types, 
        "distances": distances,
        "picked_signals_indices": picked_signals_indices, 
        "picsizes_values": picsizes_values,
        "participants_indices": participants_indices,
        "states_values": states_values,
        "names": names
    }


def produce_artificial_data(num_participants, num_states, num_trials,
        min_picsize, consider_costs, l2_minimize_artificial, 
        s3_ANS_artificial, alternatives_account,
        context_dependent_most_artificial,
        include_s3=False, include_s3_post_error=False):
    """
    This function produces artificial simulated data, i.e. an imaginary run
    of the experiment. 
    Parameters
    ----------
    num_participants: The number of participants in the experiment
    num_states: the number of states (i.e. num_objects+1)
    num_trials: the number of trials for each participant
    min_picsize: the minimum number of objects that verify the statement
    Returns
    -------
    dict
        Various information about the simulated run
    """
    # produce language information
    (possible_signals_array, real_signals_indices, costs_possible, 
        at_most_as_costly, types, names) = produce_experiment_language(
            num_states, alternatives_account)
    
    # produce participant indices for each datapoint
    # by repeating each part index n_trials times
    participants_indices = np.repeat(
        np.arange(num_participants), 
        repeats=num_trials)

    # size of the picture for each trial
    picsizes_values = np.random.choice(
        np.arange(min_picsize, num_states), 
        size=num_participants*num_trials)

    # the number of objects in the target set (<= picsize)
    states_values = np.apply_along_axis(
        lambda x: np.random.choice(np.arange(x)),
        arr=picsizes_values.reshape(-1,1)+1,
        axis=1)

    distances = create_distance(num_states) 

    # NOTE: picked_signals_indices is not used when 
    # generating data
    # but it still needs some value
    picked_signals_indices = np.full(states_values.shape, -1)

    arguments = {
        'num_participants': num_participants,
        'num_states': num_states,
        'possible_signals_array': possible_signals_array,
        'real_signals_indices': real_signals_indices,
        'costs_possible': costs_possible, 
        'at_most_as_costly': at_most_as_costly, 
        'types': types, 
        'distances': distances, 
        'picked_signals_indices': picked_signals_indices,
        'picsizes_values': picsizes_values, 
        'participants_indices': participants_indices, 
        'states_values': states_values,
        'consider_costs': consider_costs, 
        'names': names, 
        'l2_minimize_model': l2_minimize_artificial, 
        's3_ANS_model': s3_ANS_artificial,
        'include_s3': include_s3,
        'include_s3_post_error': include_s3_post_error, 
        'context_dependent_most_model': context_dependent_most_artificial
    }
     
    model = create_model(**arguments)

    with model:
        print('Test point: ', model.test_point)
        prior_samples = pm.sample_prior_predictive()

    # this functions returns values for these values 
    # even when they are not strictly speaking needed
    # This simplifies things in the analysis
    if not l2_minimize_artificial:
        prior_samples['l2_minimize_artificial'] = None
    if not consider_costs:
        prior_samples['cost_factors'] = None
    if not s3_ANS_artificial:
        prior_samples['w_vector_s3'] = None

    # breakpoint()
    # note that this does not contain context_dependent_most_artificial
    # because that doesn't have to do with how the model
    # should fit the data
    model_input = {
        "num_participants": num_participants, 
        "num_states": num_states, 
        "possible_signals_array": possible_signals_array, 
        "real_signals_indices": real_signals_indices, 
        "costs_possible": costs_possible, 
        "at_most_as_costly": at_most_as_costly, 
        "types": types, 
        "distances": distances,
        # the picked signals are sampled from the prior
        # prior_samples['picked_signals'] has shape (# observations, # samples)
        # only pick one specific sample
        "picked_signals_indices": prior_samples['picked_signals'][-1], 
        "picsizes_values": picsizes_values,
        "participants_indices": participants_indices,
        "states_values": states_values,
        "names": names
    }
    # the prior samples here are effectively the true parameters
    prior_samples.update({
        "l2_minimize_artificial": l2_minimize_artificial,
        "s3_ANS_artificial": s3_ANS_artificial
    })
    # return model_input, real_parameters
    # if l2_minimize is None, distances and choice_alphas are also None
    return model_input, prior_samples


def add_production_error(s3, error_params, library=T):
    """
    Goes from an array with shape 
    (# participants, # real signals, # states)
    where each state is a probability vector
    to an array with the same shape, but
    where the probability vectors might be squished
    """
    # # error_params has shape (# participants), so I have to reshape
    # error_params_extended = error_params[:,np.newaxis,np.newaxis]
    # unnorm_noisy = s3 + (1/(1+library.exp(-error_params_extended))-0.5)*(1-2*s3)
    # # normalize so there is a prob vector for each state
    # return unnorm_noisy / unnorm_noisy.sum(axis=-2, keepdims=True)
    # create uniform array
    uniform_unnorm = library.ones_like(s3)
    uniform = uniform_unnorm / uniform_unnorm.sum(axis=-2, keepdims=True)
    error_params_expanded = error_params[:,np.newaxis,np.newaxis]
    return error_params_expanded*uniform+(1-error_params_expanded)*s3


def define_prior(num_participants,l2_minimize_model, consider_costs,
        s3_ANS_model, context_dependent_most_model):
    """
    Creates the various priors for the model
    """
    parameters_dict = dict()

    pop_alpha_mu = pm.Normal(
        'pop_alpha_mu', mu=3, sigma=2)
    pop_alpha_sigma = pm.HalfNormal(
        'pop_alpha_sigma', sigma=2)

    parameters_dict['alphas'] = pm.TruncatedNormal(
        "alphas",
        mu=pop_alpha_mu,
        sigma=pop_alpha_sigma,
        lower=0,
        shape=num_participants
    )
    alphas_print = tt.printing.Print('alphas')(
        parameters_dict['alphas'])

    # use a prior that essentially encodes the 
    # idea that the population-level distribution
    # over production errors is going to be
    # skewed towards 0, but some participants
    # might noisy. This means alpha small-ish, beta large-ish
    error_coeffs_alpha = pm.HalfNormal(
        'pop_error_coeffs_alpha', sigma=1)
    error_coeffs_beta = pm.TruncatedNormal(
        'pop_error_coeffs_beta', mu=3, sigma=1.5, lower=0)

    parameters_dict['error_coeffs'] = pm.Beta(
        'error_coeffs', 
        alpha=error_coeffs_alpha,
        beta=error_coeffs_beta,
        shape=num_participants
    )

    if l2_minimize_model:

        ### hyperprior for population-level parameters over choice_alphas
        pop_choice_alpha_mu = pm.Normal(
            'pop_choice_alpha_mu', mu=3, sigma=2)
        pop_choice_alpha_sigma = pm.HalfNormal(
            'pop_choice_alpha_sigma', sigma=5)

        parameters_dict['choice_alphas'] = pm.TruncatedNormal(
            "choice_alphas", 
            mu=pop_choice_alpha_mu,
            sigma=pop_choice_alpha_sigma,
            lower=0,
            shape=num_participants
        )
        choice_alphas_print = tt.printing.Print(
            'choice_alphas')(parameters_dict['choice_alphas'])

    if consider_costs:

        ### hyperprior for population-level parameters over cost_factors
        pop_cost_factors_mu = pm.HalfNormal(
            'pop_cost_factors_mu', sigma=5)
        pop_cost_factors_sigma = pm.HalfNormal(
            'pop_cost_factors_sigma', sigma=5)

        parameters_dict['cost_factors'] = pm.TruncatedNormal(
            'cost_factors',
            mu=pop_cost_factors_mu,
            sigma=pop_cost_factors_sigma,
            lower=0,
            shape=num_participants
        )

    if s3_ANS_model:

        pop_s3_ANS_mu = pm.HalfNormal(
            'pop_s3_ANS_mu', sigma=5)
        pop_s3_ANS_sigma = pm.HalfNormal(
            'pop_s3_ANS_sigma', sigma=5)

        parameters_dict['w_vector_s3'] = pm.TruncatedNormal(
            'w_vector_s3',
            mu=pop_s3_ANS_mu,
            sigma=pop_s3_ANS_sigma,
            lower=0,
            shape=num_participants
        )

    if context_dependent_most_model:
        # each participant has one threshold for most
        # which I assume doesn't change throughout the experiment
        # partee et al claims the threshold for 'most' is 0.5<=t<=1.0
        pop_most_threshold_alpha = pm.Gamma(
            'pop_most_threshold_alpha', alpha=3, beta=2)
        pop_most_threshold_beta = pm.Gamma(
            'pop_most_threshold_beta', alpha=3, beta=2)
        most_thresholds_untransformed = pm.Beta(
            'most_thresholds_untransformed', 
            alpha=pop_most_threshold_alpha,
            beta=pop_most_threshold_beta
        )
        most_thresholds = pm.Deterministic(
            'most_thresholds',
            (most_thresholds_untransformed+0.5)/2
        )

    return parameters_dict


def norm_pdf(x, mu, sigma):
    return T.exp(-1/2*T.square((x-mu)/2))*(sigma*T.sqrt(2*np.pi))


def apply_ANS(s3, w_vector, subitation_range=4):
    """
    Parameters
    ----------
    s3: tensor
        Tensor with shape (participants, real_signals, states)
    w_vector: vector
        A vector with the weber values for each participant.
        Shape (n_participants)
    subitation_range: int
        How many elements at the border fall in the subitation range
    Returns
    -------
    tensor
        A tensor with shape (participants, real_signals, states)
    """
    # subitation range can be at most the number of states
    subitation_range = T.clip(subitation_range, 0, s3.shape[2])
 
    # xs contains the row index
    # and it's where the normal dists are evaluated
    # this is because each column corresponds to a real observed state
    # and will contain the probs of perceiving each possible state
    # given that real state
    _,xs,possible_states = T.mgrid[
        0:s3.shape[0],
        0:s3.shape[-1],
        0:s3.shape[-1]]
    # std of distribution is a function of possible state and weber constant
    sigmas = possible_states*w_vector[:,np.newaxis,np.newaxis]
    
    reverse_xs = T.max(xs) - xs
    reverse_possible_states = T.max(possible_states) - possible_states   
    reverse_sigmas = T.max(sigmas) - sigmas

    # crete a partial eye to substitute for the part of
    # noisy observations that concerns the states within
    # the subitation range
    sub_range_array = T.eye(xs.shape[1],subitation_range)
    # when sigma is 0, norm_pdf returns 0
    # however, in the case where x is indeed 0, the density should be 1.
    # this changes the arrays for that value
    noisy_observations = norm_pdf(xs,possible_states,sigmas)
    noisy_observations = T.set_subtensor(
        noisy_observations[:,:,:subitation_range],
        sub_range_array[np.newaxis,:,:])

    # invert the sub_range_array (anti-transpose)
    # the inverse stuff concerns the estimation of 
    # the complement of the target set
    sub_range_array_inverse = sub_range_array[::-1,::-1]
    noisy_observations_inverse = norm_pdf(
        reverse_xs,reverse_possible_states,reverse_sigmas)
    noisy_observations_inverse = T.set_subtensor(
        noisy_observations_inverse[:,:,-subitation_range:], 
        sub_range_array_inverse[np.newaxis,:,:])
    # an array with shape (participants, # possible states, # real states)
    # where real states refers to actually observed and possible to estimates
    # NOTE: they are the same in number!
    ANS_array_unnorm = noisy_observations * noisy_observations_inverse
    # normalize across last dimension so that only 
    # possible states have a non-zero probability
    ANS_array = (
        ANS_array_unnorm / 
        ANS_array_unnorm.sum(axis=-2, keepdims=True)
    )
    # marginalize to get probability of producing
    # each real signal given each real observation, 
    # once the ANS is considered
    ANS_s3 = T.batched_dot(s3, ANS_array)
    # breakpoint()
    return ANS_s3


def construct_s3_in_loop(
        picsizes_values,
        real_signals_indices,
        num_states,
        num_participants,
        possible_signals_array,
        types,
        distances,
        costs_possible,
        at_most_as_costly,
        l2_minimize_model,
        s3_ANS_model,
        consider_costs,
        include_s3,
        include_s3_post_error,
        participants_indices,
        states_values,
        return_prob_accept=True, prior_dict=None, 
        return_variables=True, **kwargs):
    """
    NOTE: this function can be used both as part of the creation of the pm 
    model, and to create a theano function that goes from a bunch of 
    inputs to the desired output, e.g. s3. When the former, prior_dict
    contains some pm variables. When the latter, prior_dict is not passed.

    NOTE: I don't define any pymc3 variable here or in the functions called
    here. It's all calculations.
    """

    min_picsize = min(picsizes_values)

    if return_prob_accept:
        probability_accept = T.zeros(
            shape=(len(picsizes_values), len(real_signals_indices)))

    if include_s3:
        s3_store = T.zeros(shape=(
            num_states, 
            num_participants,
            len(real_signals_indices),
            num_states
        ))

    if include_s3_post_error:
        s3_post_error_store = T.zeros(shape=(
            num_states, 
            num_participants,
            len(real_signals_indices),
            num_states
        ))

    ### RSA model
    # s3: for each picsize (i.e. state),
    # variable with shape (participant, real_signal, state)
    # add the first element as an empty list 
    # (corresponding to the case where the picture has 0 objects,
    # which never happens)
    # so that I can use the dataframe columns directly as indices. 
    # and because of that start loop from 1
    for index_state, state in enumerate(range(min_picsize, num_states)):

        # create the arguments to give to RSA function
        arguments = {
            'possible_signals_array': tt.shared(
                possible_signals_array[state], name="possible_signals_array"), 
            'real_signals_indices': tt.shared(
                real_signals_indices, name="real_signals_indices"), 
            'types': tt.shared(types, name="types"),
            'l2_minimize': l2_minimize_model,
            'consider_costs': consider_costs,
        }
        # if a prior dict was passed which contains a theano variable,
        # then use the alphas of the prior dict (which could be
        # a pm variable if this function is called from pm)
        # if no alpha is specified, create a new variable for it
        if prior_dict is not None:
            arguments['alphas'] = prior_dict['alphas']

        if l2_minimize_model:
            arguments['distances']=tt.shared(
                distances[:state+1,:state+1], name="distances")

            if prior_dict is not None:
                arguments['choice_alphas'] = prior_dict['choice_alphas']

        if consider_costs:
            arguments.update({
                'objective_costs_possible': tt.shared(
                    costs_possible,
                    name="objective_costs_possible"), 
                'at_most_as_costly': tt.shared(
                    at_most_as_costly,
                    name="at_most_as_costly"), 
            })
            if prior_dict is not None:
                arguments['cost_factors'] = prior_dict['cost_factors']

        # NOTE: if prior_dict is None, some variables are created inside
        # theano_RSA, and are contains in 'variables', namely
        # alphas, choice_alphas, and cost_factors
        s3, variables = theano_RSA(
            **arguments, return_symbolic=True, 
            return_variables=True)

        if s3_ANS_model:
            if prior_dict is not None:
                w_vector_s3 = T.dvector('w_vector_s3')
            else:
                w_vector_s3 = prior_dict['w_vector_s3']

            s3 = apply_ANS(s3, w_vector_s3)

        if include_s3:
            s3_store = T.set_subtensor(
                s3_store[state,:,:,:state+1], s3)

        if prior_dict is None:
            error_coeffs = T.dvector('error_coeffs')
        else:
            error_coeffs = prior_dict['error_coeffs']
        
        ### apply the error to the production probability
        s3 = add_production_error(s3, error_coeffs)

        if include_s3_post_error:
            s3_post_error_store = T.set_subtensor(
                s3_post_error_store[state,:,:,:state+1], s3)

        if return_prob_accept:
            # calculate the probability of production for that
            # combination of participants and picsize
            relevant_indices = (picsizes_values == state).nonzero()[0]
            subtensor = s3[
                participants_indices[relevant_indices],:,
                states_values[relevant_indices]]
            # has shape (# judgments, # real signals)
            probability_accept = T.set_subtensor(
                probability_accept[relevant_indices],
                subtensor)
        
    output_dict = dict()
    if return_prob_accept:
        output_dict['probability_accept'] = probability_accept
    if include_s3:
        output_dict['s3'] = s3_store
    if include_s3_post_error:
        output_dict['s3_post_error'] = s3_post_error_store

    if return_variables:
        return output_dict
    else:
        # up to this point, I have only dealt with theano variables
        # this is where I create a function from those variables to
        # some output (NOTE: some of the input variables are contained
        # in prior_dict)
        
        arguments = [variables['alphas'], error_coeffs]
        if l2_minimize_model:
            arguments.append(variables['choice_alphas'])
        if consider_costs:
            arguments.append(variables['cost_factors'])
        if s3_ANS_model:
            arguments.append(w_vector_s3)
        return tt.function(arguments, output_dict)


def create_model(num_participants, num_states, possible_signals_array,
            real_signals_indices, costs_possible, at_most_as_costly, 
            types, distances, picked_signals_indices,
            picsizes_values, participants_indices, states_values,
            consider_costs, names, l2_minimize_model, s3_ANS_model,
            context_dependent_most_model,
            include_s3=False, include_s3_post_error=False):

    with pm.Model() as model:

        prior_dict = define_prior(
            num_participants,
            l2_minimize_model,
            consider_costs,
            s3_ANS_model,
            context_dependent_most_model
        )

        input_dict = {
            'prior_dict': prior_dict,
            'picsizes_values': picsizes_values,
            'real_signals_indices': real_signals_indices,
            'num_states': num_states,
            'num_participants': num_participants,
            'possible_signals_array': possible_signals_array,
            'types': types,
            'distances': distances,
            'costs_possible': costs_possible,
            'at_most_as_costly': at_most_as_costly,
            'l2_minimize_model': l2_minimize_model,
            's3_ANS_model': s3_ANS_model,
            'consider_costs': consider_costs,
            'include_s3': include_s3,
            'include_s3_post_error': include_s3_post_error,
            'participants_indices': participants_indices,
            'states_values': states_values
        }
        construct_output = construct_s3_in_loop(**input_dict)

        probability_accept = construct_output['probability_accept']

        # save the probability of acceptance
        pm.Deterministic("probability_accept", probability_accept)
        
        if include_s3:
            pm.Deterministic('s3', construct_output['s3'])
        if include_s3_post_error:
            pm.Deterministic('s3_post_error', 
                construct_output['s3_post_error'])

        ### observed
        obs = pm.Categorical(
            "picked_signals", 
            p=probability_accept, 
            shape=len(picsizes_values), 
            observed=picked_signals_indices)
        # print(model.test_point)

        # for var in model.vars:
        #     pm.Deterministic(f'print_{var.name}', 
        #         tt.printing.Print(var.name)(var))
        #     pm.Deterministic(f'dlogp_{var.name}',
        #         tt.printing.Print(f'dlogp_wrt_{var.name}')(
        #             tt.grad(model.logpt, var, disconnected_inputs='warn')
        #         )
        #     )

    return model


def model_initialization_feedback(step, model):
    q0 = step._logp_dlogp_func.dict_to_array(model.test_point)
    p0 = step.potential.random()
    # make sure the potentials are all finite
    print("p0: ", p0)

    start = step.integrator.compute_state(q0, p0)
    print("Start energy: ", start.energy)

    # make sure model logp and its gradients are finite
    logp, dlogp = step.integrator._logp_dlogp_func(q0)
    print("logp: ", logp)
    print("dlogp: ", dlogp)

    # make sure velocity is finite
    v = step.integrator._potential.velocity(p0)
    print("v: ", v)
    kinetic = step.integrator._potential.energy(p0, velocity=v)
    print("kinetic: ", kinetic)


def plot_part(df):
    fig, ax = plt.subplots()

    restricted_df = df# df[df["total"] >=10]

    for quant in restricted_df["choice"].unique():
        props = restricted_df.loc[restricted_df["choice"] == quant, "prop"]
        sns.kdeplot(props, label=quant, ax=ax)
    ax.set_xlim(0,1)

    ax.set_title(restricted_df.iloc[0,:]["id"])
    plt.legend()
    plt.show()


def initial_filter_df(df_unfiltered, idx, cap_picsize=np.inf, 
        min_picsize=0, include_gender=False):
    """
    """
    # helps exclude rows where there's not data - they're all over 
    indices_selection_not_nan = ~(df_unfiltered["select_answer.clicked_name"].isnull())
    # helps exclude first few rows, which are just training
    indices_pic_not_nan = ~(df_unfiltered["Pic"].isnull())
    df = df_unfiltered.loc[indices_selection_not_nan & indices_pic_not_nan,:].reset_index()

    selection_indices = df["select_answer.clicked_name"].str[-1].astype(int)
    quantifiers = df.filter(like="Q").values[range(len(df)), selection_indices-1]
    # pprint(quantifiers)

    relevant = (df["Pic"]
                .str.replace("ratio", "")
                .str.replace(".png", "")
                .str.split("_", expand=True)
                .iloc[:,1:3]
                .astype(int))
    relevant.columns = ["target", "total"]
    relevant["prop"] = relevant["target"] / relevant["total"]
    relevant["choice"] = quantifiers
    # relevant["id"] = df["Prolific ID*"]
    relevant["id_index"] = idx
    if include_gender:
        relevant['gender'] = df['Gender*']

    relevant = relevant[
        (relevant["total"] <= cap_picsize) &
        (relevant["total"] >= min_picsize)
    ]
    relevant.reset_index()

    return relevant


def run_model(model, cores, draws=1000, tune=1000, n_chains=1):
    """
    
    """
    # this is done so that the progress bar prints an '\n' at the end of each step
    pm.sampling.progress_bar.end = '\n'
    with model:
        step = pm.NUTS(target_accept=0.99)
        model_initialization_feedback(step, model)
        trace=pm.sample(
            cores=cores,
            chains=n_chains,
            step=step,
            draws=draws,
            tune=tune,
            return_inferencedata=True
        )
    return trace


def pickle_model(output_path, **kwargs):
    """
    Pickles PyMC3 model and trace and whatever else is included in kwargs
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if len(list(glob(output_path+'*.pickle'))) == 0:
        new_index = 0
    else:
        new_index = max([
            int(os.path.basename(os.path.splitext(path)[0]).split("_")[-1])
            for path in list(glob(output_path+'*.pickle'))
        ]) + 1

    with open(output_path+f"model_output_{new_index}.pickle", "wb") as buff:
        pickle.dump(kwargs, buff)


def str2bool(string):
    assert type(string)==str, "Input is not a string"
    if string.lower() == 'false':
        return False
    elif string.lower() == 'true':
        return True
    else:
        raise argparse.ArgumentTypeError('Boolean value unexpected. True or False')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Inputs for the simulation")

    parser.add_argument(
        '-consider_costs', type=str2bool, default='False',
        help='Wether to consider the utterance cost (False for ACL paper)'
    )

    parser.add_argument(
        '--artificial_data', type=str2bool, default="False",
        help='Whether to produce artificial data'
    )
    # these parameters are only used if artificial_data is true
    # and they are needed to know how to simulate the data
    parser.add_argument(
        '-num_participants', type=int, default=0,
        help=''
    )
    parser.add_argument(
        '-num_states', type=int, default=0,
        help='Maximum number of objects in any pics + 1 (because it includes 0)'
    )
    parser.add_argument(
        '-num_trials', type=int, default=0,
        help='Number of trials for each participant'
    )
    parser.add_argument(
        '-l2_minimize_artificial', type=str2bool, default='True',
        help=('whether l2 tries to minimize the distance'
            'or just calculates posterior WHEN SIMULATING DATA')
    )
    parser.add_argument(
        '-s3_ANS_artificial', type=str2bool, default='False',
        help='Whether s3 perceived with ANS IN ARTIFICIAL DATA'
    )
    parser.add_argument(
        '-context_dependent_most_artificial', type=str2bool, default='False',
        help='Whether most is context dependent IN ARTIFICIAL DATA'
    )

    # these parameters describe how to run the model fit
    parser.add_argument(
        '-n_chains', type=int, default=1,
        help='Number of chains to run (if on windows, set 1)'
    )
    parser.add_argument(
        '-path_data', type=str, default='./data/*.csv',
        help='Path in which to find the data in csv format'
    )
    parser.add_argument(
        '--cores', type=int, 
        help='Number of cores for pymc3 to use'
    )
    parser.add_argument(
        '-output_path', type=str, default='./model_saved/',
        help='Path in which to save the pickled results'
    )
    parser.add_argument(
        '-draws', type=int, default=2000,
        help='Number of (non burn-in) samples to take'
    )
    parser.add_argument(
        '-tune', type=int, default=3000,
        help='Number of tuning samples to take'
    )
    parser.add_argument(
        '-max_picsize', type=int, default=100,
        help='(ONLY FOR REAL DATA) Maximum size of pics to consider'
    )
    parser.add_argument(
        '-l2_minimize_model', type=str2bool, default='True',
        help=('whether l2 tries to minimize the distance'
            'or just calculates posterior IN THE MODEL')
    )
    parser.add_argument(
        '-s3_ANS_model', type=str2bool, default='False',
        help='Whether s3 perceived with ANS IN MODEL'
    )
    parser.add_argument(
        '-context_dependent_most_model', type=str2bool, default='False',
        help='Whether most is context dependent IN MODEL'
    )

    # this argument is useful both when producing simulated data 
    # and when using real data 
    parser.add_argument(
        '-min_picsize', type=int, default=1,
        help=(
            '(FOR BOTH REAL AND SIMULATED DATA) '
            'Minimimum size of pic to consider '
            '(When 0, consider all data)')
    )
    parser.add_argument(
        '-alternatives_account', type=str2bool, default='True',
        help=('Regulates the set of alternative utterances.')
    )

    args = parser.parse_args()

    if args.artificial_data:
        assert 0 not in [
            args.num_participants, args.num_states, args.num_trials], (
            "If artificial data, specify parameters")
        model_input, real_parameters = produce_artificial_data(
            args.num_participants,
            args.num_states,
            args.num_trials,
            args.min_picsize,
            args.consider_costs,
            args.l2_minimize_artificial,
            args.s3_ANS_artificial,
            args.alternatives_account,
            args.context_dependent_most_artificial
        )

    else:
        data = get_data(
            args.path_data, 
            cap_picsize=args.max_picsize, 
            min_picsize=args.min_picsize
        )
        model_input = preprocess_data(
            data,
            args.alternatives_account,
        )
        # we don't know the real parameters if we use real data. 
        real_parameters = None

    # I'm adding to the dict rather than as arguments
    # because I am saving model_input further down
    model_input.update({
        'l2_minimize_model': args.l2_minimize_model,
        's3_ANS_model':args.s3_ANS_model,
        'context_dependent_most_model': args.context_dependent_most_model,
        'consider_costs': args.consider_costs
    })
    model = create_model(**model_input)
     
    trace = run_model(model, args.cores, args.draws, args.tune, args.n_chains)
    # If the data comes from a directory, add the directory to the stuff to save
    if not args.artificial_data:
        model_input['data'] = list(glob(args.path_data)) 

    pickle_model(args.output_path,
        model=model,
        trace=trace,
        model_input=model_input,
        real_parameters=real_parameters,
        args=vars(args)
    )
