import numpy as np
import theano as tt
import theano.tensor as T


def simplify_compatibility_array(considered_signals):
    """
    This function solves the problem with redundant information when
    there are multiple signals with the same set of alternatives.
    Rather than calculating the arrays independently for signals
    with the same comparison set, only calculate the needed ones
    and then index within the reduced array to get the behaviour
    of each real signal. 
    Example: 'all' and 'none' have the same set of alternatives. 
    Therefore, rather than calculating the arrays for 'all' and 'none'
    independently, calculate one array for each alternatives profile, 
    and then get the behaviour of 'all' and 'none' by indexing within 
    the array of the corresponding type.

    Parameters
    ----------
    considered_signals: array
        Shape (# possible signals, # possible signals)
    Returns
    -------
    tuple
        unique_alt_profiles: boolean array 
            with shape (# unique profiles, # signals)
            which says for each signal whether it belongs
            to each profile.
        index_signals_profile: len==# possible signals
            Says for each signal to which profile it belongs

        Intended usage is for an array with shape
        (# participants, # alt profiles, # possible signals, # states)
        to do array[:, index_signals_profile, T.arange(), ,:]
        to get (# participants, # possible signals, # states)
        and then index the resulting array with real_signals_indices
        to get the (# participants, # real signals, # states)
    """


    # calculate for each signal the index of the alt profile of that signal
    # i.e. all the signals with the first alt profile get 0, 
    # second get 1 etc.
    unique_alt_profiles, index_signals_profile = T.extra_ops.Unique(
        axis=0, return_inverse=True)(considered_signals)
    # cons_print = tt.printing.Print('considered_signals')(considered_signals)
    # uniquealtprint = tt.printing.Print('unique alt')(unique_alt_profiles)
    # insigprint = tt.printing.Print('index_signals_profile')(index_signals_profile)

    return (unique_alt_profiles, index_signals_profile)


def from_profiles_shape_to_real_signals_shape(array, real_signals_indices,
        index_signals_profile):
    """
    for each participant and real signal, select the 
    distribution over states that corresponds to that real signal 
    in its own comparison set 

    Parameters
    ----------
    array: tensor
        Has shape (participant, unique alternatives profiles, possible signals, state)
    Returns
    -------
    tensor
        Shape (participants, real signals, state)
    """
    # array (e.g. l2) has shape 
    # (participant, unique alternatives profiles, possible signals, state)
    # For each participant, state, and real signal, select 
    # the possible signal corresponding to that real signal
    # Resulting shape (participants, possible signals, states)
    language_possible = array[:,
        index_signals_profile,
        T.arange(array.shape[2]),
        :]
    
    # select only the real signals
    language = language_possible[:,real_signals_indices,:]

    # cons_print = tt.printing.Print('array')(array)
    # cons_print = tt.printing.Print('language poss')(language_possible)
    # cons_print = tt.printing.Print('indsigprof')(index_signals_profile)
    # cons_print = tt.printing.Print('lang')(language)
    return language


def calculate_L0(possible_signals_array, distances, choice_alphas, 
        minimize_distance=False):
    """
    l0 has shape (# participant, # trial, # signals, # states)
    And needs not consider different alternative sets
    Need to include trials because the listener 0 is going to be 
    different depending on the number of objects in the picture
    
    This function is the same for sophisticated and base RSA model
    """
    
    # find for each (possible_signal, state) the expected distance 
    # between the guess based on the signal and the state. 
    # this is identical for all participants because it only depends on 
    # the literal listener, the literal meanings (no parameters involved), 
    # and the # of objects in the picture, encoded in the picsize dimension
    language_l = possible_signals_array / possible_signals_array.sum(
        axis=-1, keepdims=True)

    if minimize_distance:
        expected_dist_l0 = T.dot(language_l, distances)
    
        # create array with shape (participant, signal, state) with 
        # the probability of guessing each state given each signal
        # this depends on the choice_alpha parameter, so specific to a participant
        unnorm_l0 = T.exp(
            choice_alphas[:,np.newaxis,np.newaxis]*-expected_dist_l0)
        l0 = unnorm_l0 / T.sum(unnorm_l0, axis=-1, keepdims=True)
    
    else:
        # add the participant dimension. In the version where l0 was 
        # a distance minimizer the l0 for each participant was difference
        # because it depended on the choice_alpha param
        # however, now it's the same for all participants
        l0 = language_l[np.newaxis,:,:]
    
        # l0_print = tt.printing.Print('l0')(l0)
    return l0


def calculate_s1(l0_extended, consider_costs, cost_factors,
        objective_costs_possible, alphas, unique_alt_profiles_expanded):
    """
    Calculate the pragmatic speaker s1
    """

    if consider_costs:
        # calculate the utility for the literal listener of each 
        # utterance given each state. utility_l0 has shape 
        # (participant, unique alt profiles, possible_signal, state)

        # NOTE: when consider_cost is true, calculate_l0 has to be
        # distance-minimizing, because otherwise it might be 0
        # in some entries, meaning that the log will be -inf,
        # which breaks the gradient
        utility_l0 = T.log(l0_extended) 
        
        # calculate the actual costs of each signal for each participant
        # shape (# participants, # possible signals)
        # by multiplying the cost factors with the signals' "objective" costs
        costs_possible = T.outer(cost_factors, objective_costs_possible)
        # if costs are considered, subtract the cost from the utility
        utility_l0 -= costs_possible[:,np.newaxis,:,np.newaxis]

        unnorm_s1 = T.exp(
            alphas[:,np.newaxis, np.newaxis, np.newaxis]*utility_l0
        )
    else:
        # if there is no cost involved, 
        # np.exp(alpha*np.log(l0)) = np.exp(np.log(l0))**alpha =
        # l0 ** alpha
        unnorm_s1 = l0_extended ** alphas[:,np.newaxis, np.newaxis, np.newaxis]
    
    # NOTE: it is important that for each set of alternative utterances,
    # there is at least one signal for each state.
    
    # NOTE: if a certain signal is not included in the comparison set 
    # for a real signal,then the speaker s1 will never produce it, 
    # for any possible state, given that real signal index
    # Therefore, if for that real signal the listener gets that 
    # impossible signal (impossible because it's not in the comparison set)
    # the behaviour is not defined. 
    # however, since NaNs break the gradient, here I create a tensor
    # with all 0s and give it the values of unnorm_s1 for those
    # signals who are compatible with the alternatives profile
    # So that within a profile, the speaker will never use 
    # a signal incompatible with the profile.
    # NOTE: unnorm_s1 has shape (# participants, # alternative profiles, # possible signals, # states)

    s1_unnorm = T.switch(
        unique_alt_profiles_expanded,
        unnorm_s1, 0
    )

    # l0_print = tt.printing.Print('unique art prof exp')(unique_alt_profiles_expanded)
    # l0_print = tt.printing.Print('s1 unnorm')(unnorm_s1.shape)

    s1 = s1_unnorm / s1_unnorm.sum(axis=-2, keepdims=True)
    # s1_print = tt.printing.Print('s1')(s1)
    return s1


def calculate_l2(s1, distances, choice_alphas, 
        unique_alt_profiles_expanded, minimize_distance=False):
    """
    Calculates pragmatic listener l2 (called l1 in usual RSA models)
    """
    # at the moment, in s1 in the possible signal dimension,
    # the signals that are not compatible with the alternatives
    # profile dimension are all 0. Therefore, dividing
    # by the sum produced NaNs. However, since all the operations 
    # in calculating s2 are within signal, and s3 excludes the
    # signals not compatible with a profile, I can just fill s1
    # with 1s for those signals not compatible with the alternatives 
    # profile

    s1_filled = T.switch(
        unique_alt_profiles_expanded,
        s1, 1)

    # normalize across the state dimension, so that each 
    # possible_signal encodes a distribution vector
    l2 = s1_filled / s1_filled.sum(axis=-1, keepdims=True)

    # do a distance-minimizing listener with the
    # more complex shape (participant, alt profiles, possible_signal, state)
    if minimize_distance:
        # shape (participants, alt profiles, possible_signals, state)
        # with the expected distance of the listener's guess to each 
        # possible real state, for each combination of participant and real signal
        # l2_print = tt.printing.Print('l2')(l2)
        # distances_print = tt.printing.Print('distances')(distances)
        expected_dist_l2 = T.tensordot(l2, distances, axes=[[3],[0]])

        unnorm_l2 = T.exp(
            choice_alphas[:,np.newaxis,np.newaxis,np.newaxis]*
            -expected_dist_l2
        )

        # renormalize
        l2 = unnorm_l2 / T.sum(unnorm_l2, axis=-1, keepdims=True)

    # l2_print = tt.printing.Print('l2')(l2)
    return l2


def calculate_s3(l2_language, real_signals_indices, cost_factors, alphas,
        consider_costs, index_signals_profile):
    """
    do what I did above for s1, but with real signals instead of 
    the imagined signals
    """ 

    # for each participant for each trial, calculate probability of 
    # them producing each of the available signals for each state
    # then the probability of the total observed behaviour can be 
    # calculated from this

    if consider_costs:
        # calculate the utility for l2 of each real signal given each state
        # utility_l2 has shape (participant, real_signal, state)
        utility_l2 = T.log(l2_language)
        objective_costs_real = objective_costs_possible[real_signals_indices]
        # array with shape (# participants, # real signals)
        costs_real = T.outer(cost_factors, objective_costs_real)
        utility_l2 -= costs_real[:,:,np.newaxis]

        unnorm_s3 = T.exp(alphas[:,np.newaxis, np.newaxis]*utility_l2)
    else:
        # see l0 for an explanation of this condition
        unnorm_s3 = l2_language**alphas[:,np.newaxis,np.newaxis]

    # shape (participant, real signal, state)
    s3 = unnorm_s3 / unnorm_s3.sum(axis=-2, keepdims=True)
    # s3_print = tt.printing.Print('s3')(s3)
    return s3


def theano_RSA(
        possible_signals_array=T.lmatrix("possible_signals_array"), 
        real_signals_indices=T.lvector("real_signals_indices"), 
        alphas=T.dvector("alphas"), 
        choice_alphas=None, 
        cost_factors=T.dvector("cost_factors"), 
        objective_costs_possible=T.dvector("objective_costs_possible"), 
        at_most_as_costly=T.lmatrix("at_most_as_costly"), 
        types=T.lvector("types"),
        distances=None, 
        l2_minimize=True,
        consider_costs=False,
        return_symbolic=True,
        return_gradient=False,
        return_variables=False,
        return_language=['s3'],
        flatten_if_single_lang=True):
    """
    Parameters
    ----------
    possible_signals_array: 
        Shape (# possible signals, # states)
        The meaning of each possible signal
    real_signals_indices: 
        the indices of the real signals in the possible_signals_array. 
        shape (# real signals)
    alphas: 
        shape (# participants)
    choice_alphas: 
        shape (# participants)
    cost_factors: 
        If it's not considered:
            (1) they aren't considered by the speaker
            (2) they don't determine compatibility between signals
        How much each participant weight the cost of the messages in production.
        shape (# participants)
    objective_costs_possible: 
        The "objective" cost of the signals
        shape (# signals)
    at_most_as_costly:
        for each real signal, whether each possible signal is 
        at most as costly as the real signal
        shape (# real signals available to participant,
        # signals considered by listener). 
        i,j is 1 iff cost(i) >= cost(j)
    types:
        The type of each signal (they are used together with at_most_as_costly
        to calculate the alternatives set of each signal)
    distances:
        shape (# states, # states)
    return_symbolic: bool
        If true, returns a theano object that can be used in PyMC3. 
        Otherwise, returns a function that can be evaluated on numpy arrays.
    Returns
    -------
    theano variable or theano function
        Depending on return_symbolic
        s3 contains (participant, real signal, state)
    """
    # if l2 minimizes and no value was specified,
    # use the choice_alphas
    # otherwise keep it None
    if l2_minimize and (choice_alphas is None):
        choice_alphas = T.dvector("choice_alphas")
    if l2_minimize and (distances is None):
        distances = T.dmatrix("distances")
    if not l2_minimize and (choice_alphas is not None):
        print("l2 does not minimize but a choice_alphas was specified!")
    if not l2_minimize and (distances is not None):
        print("l2 does not minimize but distances were specified!")

    # the language array for each trial, but only for the real signals
    # shape (trials, possible signals, states)
    real_signals_array = possible_signals_array[real_signals_indices]

    if consider_costs:
        # shape (# real signal, # possible signals)
        considered_signals = at_most_as_costly & types.dimshuffle("x", 0)
    else:
        # considered_signals has shape (# real signals, # possible signals)
        # element (i,j) says whether signal j is compatible with signal i
        # in other words: each row is the set of alternatives
        # when cost_factors is None, j is compatible with i iff type(j)<=type(i)
        considered_signals = types.dimshuffle(0, 'x') >= types.dimshuffle('x', 0)
        # considered_print = tt.printing.Print('considered')(considered_signals)

    # use this function to help with the simplification where
    # instead of recalculating the alternatives set for every signal,
    # I only calculate for different alternatives sets
    (unique_alt_profiles, index_signals_profile) = simplify_compatibility_array(
                considered_signals)
    # tile to shape (num participants, alt profs, possible sigs, states)
    unique_alt_profiles_expanded = T.tile(
        unique_alt_profiles[np.newaxis, :,:, np.newaxis],
        reps=(alphas.shape[0],1,1,possible_signals_array.shape[-1])
    )

    # shape (participants, possible signals, states)
    l0 = calculate_L0(possible_signals_array, distances, choice_alphas)

    # note that the way that the picked signal restricts the possible signals 
    # did not matter until this point, because the literal listener does not 
    # consider the alternative signals when calculating its posterior 
    # given the signal. so I just calculate l0 for 
    # all possible signals indistinctly
    # however, since the restriction by signal matters for s1, 
    # I tile this here into shape 
    # (participant, alternative_profiles, possible_signal, state)
    l0_extended = l0[:,np.newaxis,:,:]
    l0_extended = T.tile(
        l0_extended,
        reps=(1, unique_alt_profiles.shape[0],1,1)
    )
    
    # (participant, unique alternatives profiles, possible signals, state)
    s1 = calculate_s1(l0_extended, consider_costs, cost_factors,
        objective_costs_possible, alphas, unique_alt_profiles_expanded)

    # (participant, unique alternatives profiles, possible signals, state)
    l2 = calculate_l2(s1, distances, choice_alphas,
        unique_alt_profiles_expanded, minimize_distance=l2_minimize)

    # (participants, real signals, states)   
    l2_language = from_profiles_shape_to_real_signals_shape(
        l2, real_signals_indices, index_signals_profile)
    
    # (participants, real signals, states)
    s3 = calculate_s3(l2_language, real_signals_indices, cost_factors,
        alphas, consider_costs, index_signals_profile)

    # For printing specific languages
    # s1_language = from_profiles_shape_to_real_signals_shape(
    #     s1, real_signals_indices, index_signals_profile)
    # cons_print = tt.printing.Print('s1')(s1)
    
    # make sure the shape is uniform, namely
    # (# participants, # real signals, # states)
    languages_to_return = []
    if 'l0' in return_language:
        l0_language = l0[:,real_signals_indices,:]
        languages_to_return.append(l0_language)
    if 's1' in return_language:
        s1_language = from_profiles_shape_to_real_signals_shape(
            s1, real_signals_indices, index_signals_profile)
        languages_to_return.append(s1_language)
    if 'l2' in return_language:
        languages_to_return.append(l2_language)
    if 's3' in return_language: 
        languages_to_return.append(s3)
    
    if len(languages_to_return)==1 and flatten_if_single_lang:
        output_lang = languages_to_return[0]
    else:
        output_lang = languages_to_return

    if return_symbolic:
        if return_variables:
            variables = {                
                "possible_signals_array": possible_signals_array, 
                "real_signals_indices": real_signals_indices, 
                "alphas": alphas, 
                "types": types,
                "distances": distances,
            }
            if l2_minimize:
                variables['choice_alphas'] = choice_alphas
            if consider_costs:
                variables.update({
                    "cost_factors": cost_factors, 
                    "objective_costs_possible": objective_costs_possible, 
                    "at_most_as_costly": at_most_as_costly
                })
            return output_lang, variables
        else:
            return output_lang
    else:
        arguments = [
                possible_signals_array, 
                real_signals_indices, 
                alphas, 
                types,
            ]
        if consider_costs:
            arguments += [
                cost_factors, 
                objective_costs_possible, 
                at_most_as_costly, 
            ]
        if l2_minimize:
            arguments += [
                choice_alphas,
                distances
            ]

        # if return_gradient, defines the jacobian as
        # the return value, otherwise whichever
        # language was specified
        output = tt.gradient.jacobian(
            expression=output_lang.flatten(), 
            wrt=alphas
        ) if return_gradient else output_lang

        return tt.function(
            arguments, 
            output,
            on_unused_input='warn'
        )
