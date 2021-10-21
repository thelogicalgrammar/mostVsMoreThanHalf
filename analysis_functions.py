from copy import deepcopy
from os.path import join, basename, splitext
from os import mkdir
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.special import gamma
import simulation
import theano as tt
import pymc3 as pm
import RSA_theano_model
import pickle
from pprint import pprint
import arviz as az
import argparse


def plot_kde_data(df, figname='aggregated_data', ylim=None, 
                  rightpos=1., leftpos=0., savefig=True, sharey=True):
    
    # name of each signal in order and x-axis position of label 
    name_data = [
        ['None', rightpos],
        ['Few', rightpos],
        ['Some', rightpos],
        ['Less than half', rightpos],
        ['Half', leftpos],
        ['More than half', leftpos],
        ['Most', leftpos],
        ['Many', leftpos],
        ['All', leftpos]
    ]

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(
        df, 
        row="signals", 
        hue='signals',
        row_order=[x[0] for x in name_data], 
        hue_order=[x[0] for x in name_data], 
        aspect=15, 
        height=0.3,
        sharey=sharey
    )

    g.map(
        sns.kdeplot, 
        "proportions",
        bw_adjust=1, 
        clip_on=False,
        fill=True, 
        alpha=1, 
        linewidth=1.5,
        clip=[0,1]
    )

    g.map(
        sns.kdeplot, 
        "proportions", 
        clip_on=False, 
        color="w", 
        lw=2, 
        bw_adjust=1
    )
    
    g.map(plt.axhline, y=0, lw=1, clip_on=False)

    # g.map(plot_label, "signals")
    print(g.axes)
    g.fig.subplots_adjust(hspace=-.6, bottom=0.2)
    g.set_titles("")
    g.set(yticks=[])
    g.set(xlim=[0,1])
    if ylim is not None:
        g.set(ylim=[0,ylim])
    g.despine(bottom=True, left=True)

    for index, (signame, xpos) in enumerate(name_data):
        if xpos<0.5:
            ha='left'
        else:
            ha='right'
        g.axes[index,0].text(
            xpos, 
            0.4, 
            signame, 
    #         fontweight="bold", 
            style='italic',
            horizontalalignment=ha
        )

    g.axes.flatten()[-1].set_xlabel('Proportion')

    if savefig:
        plt.savefig(f'{figname}.png', dpi=300)
    
    return g

    
def get_fit(path):
    with open(path, 'rb') as openfile:
        results = pickle.load(openfile)

    model = results["model"]
    trace = results["trace"] 
    model_input = results["model_input"] 
    real_parameters = results["real_parameters"] 
    
    return model, trace, model_input, real_parameters


def posterior_predictions():
    # TODO: implement
    pass


def model_comparison(*args):
    """
    args are a bunch of paths
    Each path contains the data stored for one model fit
    """
    models_dict = dict()
    for path in args:
        model, trace, model_input, real_parameters = get_fit(path)
        model_name = splitext(basename(path))
        print('Done with model: ', model_name)
        model.name=model_name
        models_dict[model] = trace
    WAIC_comparison = pm.compare(models_dict, ic='WAIC')
    pm.compareplot(WAIC_comparison)
    plt.show()
    print(WAIC_comparison)


def plot_aggregated_participants(model_input, names_to_plot=None, path_save=''):
    names = np.array(model_input['names'])
    real_signals_indices = np.array(model_input['real_signals_indices'])
    picked_signals_indices = np.array(model_input['picked_signals_indices'])
    picsizes_values = np.array(model_input['picsizes_values'])
    participants_indices = np.array(model_input['participants_indices'])
    states_values = np.array(model_input['states_values'])
    num_states = np.array(model_input['num_states'])
    
    names_real = names[real_signals_indices]
    if names_to_plot is None:
        names_to_plot = names_real
    unique_picsizes_values = np.unique(picsizes_values)
    colors = cm.get_cmap('Set1', len(real_signals_indices))(
        np.linspace(0, 1, len(real_signals_indices)))
    # one row for each picsize value
    fig, axes = plt.subplots(
        len(unique_picsizes_values),
        sharex=True, figsize=(5,4))
    # breakpoint()
    # one ax for each unique picsize
    for unique_picsize, ax in zip(unique_picsizes_values, axes):
        for unique_signal_index, signal_name in zip(
                real_signals_indices, names_real):
            mask = ((picsizes_values==unique_picsize) &
                    (picked_signals_indices==unique_signal_index))
            subset_states_values = states_values[mask]
            # don't plot the absolute numbers, plot the proportion
            subset_states_values_prop = subset_states_values/unique_picsize
            if signal_name in names_to_plot:
                # for part_index in np.unique(participants_indices):
                #     signal_states_part_mask = mask & \
                #         (participants_indices==part_index)
                #     print("part index: ", part_index, " signal: ", signal_name)
                #     print('states_values: ', 
                #             states_values[signal_states_part_mask])
                #     print("\n")
                sns.kdeplot(
                    subset_states_values_prop,
                    ax=ax, label=signal_name,
                    color=colors[unique_signal_index],
                    legend=False)
        ax.set_ylabel(f'{unique_picsize}')
        ax.set_xlim(0,1)
    axes[-1].legend(
        bbox_to_anchor=(0.5,1),
        loc='lower center', 
        fontsize='x-small',
        ncol=2
    )    
    # fig.tight_layout()
    fig.savefig(f'{path_save}/aggregated_participants.png', dpi=300)


def gamma_pdf(x, mu, sigma):
    alpha = mu**2 / sigma**2
    beta = mu / sigma**2
    return ((beta**alpha)*(x**(alpha-1))*np.exp(-beta*x)) / gamma(alpha)


def prior_predictions(alternatives_account=True, return_data=False):
    """

    """

    # it causes an error if num_participants = 1
    num_participants = 2
    num_states = 21
    min_picsize = 10
    consider_costs = False
    l2_minimize_artificial = True
    s3_ANS_artificial = False
    num_trials = 20

    # these inputs don't matter 
    s3_ANS_model = False
    context_dependent_most_artificial = False

    model_input, real_parameters = simulation.produce_artificial_data(
        num_participants,
        num_states,
        num_trials,
        min_picsize,
        consider_costs,
        l2_minimize_artificial,
        s3_ANS_artificial,
        alternatives_account,
        context_dependent_most_artificial,
        include_s3=True,
        include_s3_post_error=True
    )

    # a bunch of picked_signals_indices were sampled from the prior
    # but I am not going to plot them
    model_input['picked_signals_indices'] = model_input[
            'picked_signals_indices'][0]

    tt.config.compute_test_value = 'ignore'
    if return_data:
        return {
            'trace': real_parameters,
            'model_input': model_input
        }
    else:
        plot_participants_of_hierarchical_model(
            trace=real_parameters,
            model_input=model_input,
            path_append='prior_',
            # argument for plot_participant
            plot_judgments=False,
        )


def from_model_input_to_lang(model_input, trace, lang=['s3'], 
        include_error=False):
    """
    Goes from the model input and a trace to the specified
    languages (i.e. l0, s1, etc.)
    Returns
    -------
    list of lists of arrays
        Top list: languages
        Nest list: state
        Inner array: (participants, real signals, states)
    """
    if include_error:
        assert lang==['s3'], "If you include error, language has to be s3"

    min_picsize = model_input["picsizes_values"].min()
    num_states = model_input["picsizes_values"].max()+1

    lang_func = RSA_theano_model.theano_RSA(
        return_symbolic=False,
        return_language=lang
    )    

    # for each language, initialize a list to add the arrays to
    lang_lists = [[[]]*len(lang)]*min_picsize
    if include_error:
        lang_lists_with_errors = [[[]]*len(lang)]*min_picsize
    for state in np.arange(min_picsize, num_states):
        # langs is a list with len len(lang)
        # and for each element has an array with shape
        # (participants, real signal, state)
        langs = lang_func(
            possible_signals_array=model_input['possible_signals_array'][state],
            real_signals_indices=model_input['real_signals_indices'],
            types=model_input['types'],
            distances=model_input['distances'][:state+1,:state+1],
            alphas=trace['alphas'].flatten(),
            choice_alphas=trace['choice_alphas'].flatten(),
        )
        # append the language array for that state
        lang_lists.append(langs)
        if include_error:
            s3_with_error = simulation.add_production_error(
                langs,
                trace['error_coeffs'].flatten(),
                library=np
            )
            lang_lists_with_errors.append(s3_with_error)
    if len(lang)>1:
        lang_lists = list(zip(*lang_lists))
    if include_error:
        return lang_lists, lang_lists_with_errors
    else:
        return lang_lists


def plot_single_parameter(alpha, choice_alpha, model_input):
    trace = {
        'alphas': np.array([alpha]),
        'choice_alphas': np.array([choice_alpha])
    }
    # lang_names = ['l0', 's1', 'l2', 's3']
    lang_names = ['s3']
    # lang_names = ['s1']
    languages = from_model_input_to_lang(
        model_input, trace, lang=lang_names)

    names_real = np.array(names)[np.array(model_input['real_signals_indices'])]

    for language_name, language in zip(lang_names, languages):
        # eliminate those initial empty lists
        language_reduced = [j[0] for j in language if type(j)!=list]

        fig, axes = plt.subplots(
            len(language_reduced),
            len(names_real),
            sharex=True, sharey=True
        )
        for j, signame in enumerate(names_real):
            for i, state in enumerate(language_reduced):
                axes[i,j].plot(state[j])
            axes[0,j].set_title(signame, fontsize='x-small')
        plt.savefig((f'{language_name}_'
            f'alpha={alpha}_choice={choice_alpha}.png'), dpi=300)

        
def plot_participant(s3, names_real, picsizes_values,
        alphas, error_coeffs=None, choice_alphas=None, w_vector_s3=None,
        picked_signals_indices=None, states_values=None, s3_with_error=None,
        path='.', participant_index=0, plot_judgments=True, name_append=""):
    """
    Parameters
    ----------
    """
    if plot_judgments:
        assert (
            (picked_signals_indices is not None) and (states_values is not None)), "can't plot judgs!"

    min_picsize = picsizes_values.min()
    num_states = picsizes_values.max()+1

    # print("Should be 4: ", len(langs_list))
    # print("Should be # states: ", len(langs_list[0]))
    # print("should be #MCMC sample, 9, 10: ", langs_list[0][0].shape)

    fig, axes = plt.subplots(
        len(names_real)+1, # the additional one is for the trace plots
        num_states-min_picsize,
        sharey=True
    )
    # plot the trace
    gs = axes[0,0].get_gridspec()
    for ax in axes[0,:]:
        ax.remove()

    alphas_trace_ax = fig.add_subplot(gs[0,0:3])
    sns.kdeplot(alphas, ax=alphas_trace_ax)
    alphas_trace_ax.set_xlim(0,4)
    alphas_trace_ax.set_title(r'$\alpha$')
    alphas_trace_ax.xaxis.tick_top()
    alphas_trace_ax.tick_params(axis='both', which='major', labelsize='x-small')
    alphas_trace_ax.set_ylabel('')

    if choice_alphas is not None:
        choice_alphas_trace_ax = fig.add_subplot(gs[0,4:7])
        sns.kdeplot(choice_alphas, ax=choice_alphas_trace_ax)
        choice_alphas_trace_ax.set_xlim(0,4)
        choice_alphas_trace_ax.set_title(r'$\rho$')
        choice_alphas_trace_ax.xaxis.tick_top()
        choice_alphas_trace_ax.tick_params(
            axis='both', which='major', labelsize='x-small')
        choice_alphas_trace_ax.set_ylabel('')

    if error_coeffs is not None:
        error_coeffs_trace_ax = fig.add_subplot(gs[0,8:])
        sns.kdeplot(error_coeffs, ax=error_coeffs_trace_ax)
        error_coeffs_trace_ax.set_xlim(0,1)
        error_coeffs_trace_ax.set_title(r'$\epsilon$')
        error_coeffs_trace_ax.xaxis.tick_top()
        error_coeffs_trace_ax.tick_params(
            axis='both', which='major', labelsize='x-small')
        error_coeffs_trace_ax.set_ylabel('')

    if w_vector_s3 is not None:
        # plot the trace for the ANS
        pass

    for picsize_index, picsize in enumerate(range(min_picsize, num_states)):
        # s3 has shape (MCMC samples, max_picsize, real signals, picsizes) 
        # for the participnat, get the s3 for the specified picsize
        # in the local models, I accumulate the data on a list
        # in the actual model, I have a zero-padded array
        if type(s3)==np.array:
            s3_sub = s3[picsize,:,:,:picsize+1]
        elif type(s3)==list:
            s3_sub = s3[picsize]
        # add a dimension so it has shape (1, #samples,#realsigs,#states) 
        # which returns shape e.g. (1, 4000, 9, 16)
        # i.e. 1, # samples, # real signals, # states
        # the initial 1 is because hpd is usually applied to a trace with
        # multiple variables, so the first dimension is for the variables.
        # finally, reshape to (2, # real signals, # states)
        hpd_mins, hpd_maxs = np.transpose(pm.hpd(s3_sub[None]),(2,0,1))

        if s3_with_error is not None:
            if type(s3_with_error)==np.array:
                s3_with_error_sub = s3_with_error[picsize,:,:,:picsize+1]
            elif type(s3)==list:
                s3_with_error_sub = s3_with_error[picsize]
            hpd_mins_with_error, hpd_maxs_with_error = pm.hpd(
                s3_with_error_sub[None]).transpose(2,0,1)

        # loop over the rows of the plot, 
        # i.e. the real signals
        for signal_index, name in enumerate(names_real):
            participant_ax = axes[signal_index+1, picsize_index]
            if plot_judgments: 
                picked_signals_indices = np.array(picked_signals_indices)
                # find the indices for which the participant has used that signal
                indices_where_signal = picked_signals_indices==signal_index
                indices_where_picsize = picsizes_values==picsize
                states_signal_selected = states_values[
                    indices_where_picsize & indices_where_signal]
                for value, count in zip(
                        *np.unique(states_signal_selected, return_counts=True)):
                    stepsize=0.03
                    participant_ax.scatter(
                        [value]*count,
                        np.linspace(0.9,0.9-(stepsize*(count-1)),count),
                        s=0.5, color='blue', alpha=0.6)

            lower = hpd_mins[signal_index]
            upper = hpd_maxs[signal_index]
            participant_ax.fill_between(
                range(len(lower)), lower, upper,
                alpha=1., label='s3', color='blue'
            )
            if s3_with_error is not None:
                lower_with_error = hpd_mins_with_error[signal_index]
                upper_with_error = hpd_maxs_with_error[signal_index]
                participant_ax.fill_between(
                    range(len(lower_with_error)),
                    lower_with_error, upper_with_error,
                    alpha=0.5, label='s3_with_error', color='red'
                )
            participant_ax.set_xticks([])
            participant_ax.tick_params(
                axis='both', which='major', labelsize='x-small')
            participant_ax.set_ylim(0,1)

    for signal_index, name in enumerate(names_real):
        axes[signal_index+1, 0].set_ylabel(
            name.replace(" ", "\n"), fontsize="x-small")
    for picsize_index, picsize in enumerate(range(min_picsize, num_states)):
        axes[-1, picsize_index].set_xticks([0, picsize])
    fig.savefig(
        join(path, f'{name_append}{participant_index}.png'),
        dpi=300,
        transparent=True
    )
    plt.close()


def plot_trace(trace, model):
    with model:
        pm.traceplot(
            trace, 
            var_names=["alphas", "choice_alphas"]
        ) 
    plt.savefig('traceplot.png', dpi=300)


def from_trace_to_single_participant(trace, part_index):
    single_part_trace = dict()
    for key, value in trace.items():
        if type(value) == np.ndarray:
            try:
                single_part_trace[key] = value[:,part_index]
            except KeyError:
                pass
    return single_part_trace

    
def plot_participants_of_hierarchical_model(trace, model_input,
        path_append='', **kwargs):
    """
    Loops over the participants of a hierarchical model 
    and for each participant plot the data
    """
    try:
        s3 = trace['s3']
        s3_with_error = trace['s3_post_error']
    except KeyError:
        s3 = None
        s3_with_error = None
    
    participants_indices = model_input['participants_indices']
    names_real = np.array(model_input['names'])[
        np.array(model_input['real_signals_indices'])]
    try:
        picked_signals_indices = np.array(model_input['picked_signals_indices'])
        states_values = np.array(model_input['states_values'])
    except KeyError:
        picked_signals_indices = None
        states_values = None

    # save one plot for each participant
    # when doing prior checks, there's only one participant
    for index in range(model_input["num_participants"]):

        folder_name = './analysis_plots/hierarchical_fits_'+path_append+'/'
        try:
            mkdir(folder_name)
        except FileExistsError:
            pass

        # calculate the s3 and s3 with error if they aren't already there
        # (They're precalculated when doing prior checks)
        if s3 is None or s3_with_error is None:
            trace_reduced = {
                'alphas': trace['alphas'][:,index].flatten(),
                'choice_alphas': trace['choice_alphas'][:,index].flatten(),
                'error_coeffs': trace['error_coeffs'][:,index].flatten(),
            }
            s3_index, s3_index_post_error = from_model_input_to_lang(
                model_input, trace_reduced, include_error=True)
        else:
            # shape of s3_index: (MCMC samples, picsizes, real signals, picsizes)
            # but reshape to (picsizes, samples, real_signals, states)
            #isolate the s3 relative to this participant
            s3_index = np.transpose(s3[:,:,index,:,:], (1,0,2,3))
            s3_index_post_error = np.transpose(
                s3_with_error[:,:,index,:,:], (1,0,2,3))

        this_participant_indices = participants_indices==index
        if picked_signals_indices is not None:
            participant_picked_sig_ind = picked_signals_indices[
                    this_participant_indices]
            participant_states_values = states_values[this_participant_indices]
        # reduce to only one participant's data
        args = {
            's3': s3_index, 
            's3_with_error': s3_index_post_error,
            'picsizes_values': model_input['picsizes_values'][
                this_participant_indices],
            'alphas': trace['alphas'][:,index], 
            'names_real': names_real, 
            'error_coeffs': trace['error_coeffs'][:,index],
            'picked_signals_indices': participant_picked_sig_ind,
            'states_values': participant_states_values
        }
        try:
            if trace['choice_alphas'] is not None:
                args['choice_alphas'] = trace['choice_alphas'][:,index]
        except KeyError:
            args['choice_alphas'] = None
        try:
            if trace['w_vector_s3'] is not None:
                args['w_vector_s3'] = trace['w_vector_s3'][:,index]
        except KeyError:
            args['w_vector_s3'] = None

        plot_participant(
            **args,
            participant_index=index, 
            path=folder_name,
            **kwargs
        )


def save_inf_data(inf_data, path):
    # save arviz data in the plotting folder    
    with open(path, 'wb') as openfile:
        pickle.dump(inf_data, openfile)


def plot_trace_hierarchical_model(folder_add, model=None, trace=None,
        model_input=None, inf_data_path=None, folder_retrieve=None, name_add=''):
    
    assert not ((model==None) and (path_retrieve==None)), (
        'Specify at least one of model and path_retrieve')

    # check if there is a pickle object with the inference data already
    # stored at the specified location.
    if inf_data_path is None:
        inf_data = az.from_pymc3(trace=trace, model=model)
    else:
        try:
            with open(inf_data_path, 'rb') as openfile:
                inf_data = pickle.load(openfile)
        except FileNotFoundError:
            # arviz returns InferenceData object
            inf_data = az.from_pymc3(trace=trace, model=model)
            print('Transformed into arviz InferenceData') 
            save_inf_data(inf_data, inf_data_path)
            print('and saved')

    for var in inf_data.posterior:
        # specify which variables to plot
        if var!='probability_accept':
            if 'pop' in var:
                az.plot_trace(
                    data=inf_data, 
                    var_names=var
                )
            else:
                az.plot_forest(
                    data=inf_data,
                    var_names=var
                )
            plt.savefig(f'{folder_add}/traceplot_{name_add}_{var}.png', dpi=300)
            plt.close()


def get_waic(inf_data, return_variances=True):
    """
    Calculate waic and possibly return variances, which are
    useful for diagnostics (if some of them are >0.4, 
    the WAIC might be unrealiable).
    """
    data_waic = az.waic(inf_data, pointwise=True)
    
    # these three lines to calculate vars are copied from 
    # arviz-devs.github.io/arviz/_modules/arviz/stats/stats.html#waic
    log_lik = az.stats.stats_utils.get_log_likelihood(inf_data)
    log_lik = log_lik.stack(sample=('chain', 'draw'))
    vars_lpd = log_likelihood.var(dim='sample')

    return (data_waic, vars_lpd) if return_variances else data_waic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inputs for the simulation")

    parser.add_argument(
        '--path_fitted', type=str, default=None,
        help='The path to the fitted model'
    )
    parser.add_argument(
        '--path_fitted_2', type=str, default=None,
        help='Path to second fitted model for e.g. model comparison'
    )
    parser.add_argument(
        '--analysis_type', type=str, action='append',
        help=('What function to run on the model'
            '(Can provide more than one, but use flag repeatedly)')
    )
    parser.add_argument(
        '--path_save', type=str,
        help='Where to save the resulting pictures'
    )
    parser.add_argument(
        '--name_add', type=str, default='',
        help='What name to add to the produced files'
    )
    parser.add_argument(
        '--InferenceData_path', type=str, default=None,
        help='The path to the inference data object (Optional)'
    )

    args = parser.parse_args()
    analysis_type = args.analysis_type

    if args.path_fitted is not None:
        model, infdata, model_input, real_parameters = get_fit(
            args.path_fitted)
        # '../model_saved/model_output_0.pickle'
        # 'C:/Users/faust/Desktop/amsterdam_postdoc_local/model_output_3.pickle'

        # calculate the trace from the infdata 
        # note that I need to flatten the first two dimensions
        # which correspond to chain,draw in the infdata
        trace = {
            key:v.values.reshape(-1,*v.values.shape[2:])
            for key, v in infdata.posterior.items()
        }
        # pprint(trace.varnames)
        pprint(model_input.keys())

    # plot_participant(model_input, trace)
    # plot_trace(trace, model)
    # plot_single_parameter(3, 3, model_input)

    if 'plot_individual_participants' in analysis_type:
        plot_participants_of_hierarchical_model(
            trace, 
            model_input,
            path_append=args.name_add
        )

    if 'prior_prediction' in analysis_type:
        prior_predictions()

    if 'plot_aggregated' in analysis_type:
        plot_aggregated_participants(
            model_input,
            names_to_plot=['Most', 'More than half'],
            path_save=args.path_save
        )

    if 'model_comparison' in analysis_type:
        # in this case the results are simply printed
        # in the log file
        model_comparison(args.path_fitted, args.path_fitted_2)
        # ('C:/Users/faust/Desktop/amsterdam_postdoc_local/'
        #     'model_output_with_fourths.pickle'),
        # ('C:/Users/faust/Desktop/amsterdam_postdoc_local/'
        #     'model_output_with_thirds.pickle')

    if 'plot_trace_model' in analysis_type:
        plot_trace_hierarchical_model(
            folder_add=args.path_save,
            trace=trace,
            model=model,
            model_input=model_input,
            folder_retrieve=args.path_fitted,
            name_add=args.name_add,
            inf_data_path=args.InferenceData_path
        )

