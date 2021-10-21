python ./analysis_functions.py ^
--path_fitted C:/Users/faust/Desktop/amsterdam_postdoc_local/model_output_without_structural_account.pickle ^
--analysis_type plot_individual_participants ^
--path_save ./analysis_plots ^
--name_add without_structural_account 

:: python ./analysis_functions.py ^
:: --path_fitted C:/Users/faust/Desktop/amsterdam_postdoc_local/model_output_both_mechanisms.pickle ^
:: --analysis_type plot_individual_participants ^
:: --path_save ./analysis_plots ^
:: --name_add both_mechanisms 

:: python ./analysis_functions.py ^
:: --path_fitted C:/Users/faust/Desktop/amsterdam_postdoc_local/model_output_both_mechanisms.pickle ^
:: --analysis_type plot_trace_model ^
:: --path_save ./analysis_plots ^
:: --name_add both_mechanisms ^
:: --InferenceData_path C:/Users/faust/Desktop/amsterdam_postdoc_local/InferenceData_both_mechanisms.pickle
