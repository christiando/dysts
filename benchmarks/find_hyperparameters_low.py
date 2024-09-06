#!/usr/bin/python
# CUDA_VISIBLE_DEVICES=0 python find_hyperparameters.py 

import sys
import os
import json

# HS using local dysts
sys.path.append(os.path.join(os.path.dirname(__file__),'../../dysts/'))
import dysts
from dysts.datasets import *

import pandas as pd
import numpy as np
np.random.seed(0)

# Original
# import darts
# from darts.models import *
# from darts import TimeSeries
# import darts.models

# HS using local darts
sys.path.append(os.path.join(os.path.dirname(__file__),'../../darts/'))
import darts
from darts.models import *
from darts import TimeSeries
import darts.models


cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()


# input_path = os.path.dirname(cwd)  + "/dysts/data/train_univariate__pts_per_period_15__periods_12.json"
# pts_per_period = 15
# network_inputs = [5, 10, int(0.5 * pts_per_period), pts_per_period] # can't have kernel less than 5

input_path = os.path.dirname(cwd)  + "/dysts/data/train_univariate__pts_per_period_100__periods_12.json"
pts_per_period = 100
network_inputs = [5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period]

SKIP_EXISTING = True
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE, 
                 darts.utils.utils.SeasonalityMode.NONE, 
                 darts.utils.utils.SeasonalityMode.MULTIPLICATIVE]
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE, 
                 darts.utils.utils.SeasonalityMode.NONE
                ]
time_delays = [3, 5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period, int(1.5 * pts_per_period)]
time_delays = [3, 5, 10, int(0.25 * pts_per_period)]
network_outputs = [1, 4]
network_outputs = [1]


import torch
has_gpu = torch.cuda.is_available()
if not has_gpu:
    warnings.warn("No GPU found.")
else:
    warnings.warn("GPU working.")



dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/hyperparameters/221025_hyperparameters_" + dataname + ".json"
#221025_hyperparameters_ # noise level 0.8, not including RBF
#220812_hyperparameters_ # noise level 0.2, not including RBF
#220618_hyperparameters_ # includes RBF kernel
#220428_hyperparameters_ # for 2022 6 presentation before Christian's error correction

equation_data = load_file(input_path)

#print(equation_data)
try:
    with open(output_path, "r") as file:
        all_hyperparameters = json.load(file)
        
except FileNotFoundError:
    all_hyperparameters = dict()

parameter_candidates = dict()

parameter_candidates["ARIMA"] = {"p": time_delays}
parameter_candidates["LinearRegressionModel"] = {"lags": time_delays}
#parameter_candidates["RandomForest"] = {"lags": time_delays, "lags_exog": [None]} #HS lags_exog returns an error. Turned it off.
parameter_candidates["RandomForest"] = {"lags": time_delays}
parameter_candidates["NBEATSModel"] = {"input_chunk_length": network_inputs, "output_chunk_length": network_outputs}
parameter_candidates["TCNModel"] = {"input_chunk_length": network_inputs, "output_chunk_length": network_outputs}
parameter_candidates["TransformerModel"] = {"input_chunk_length": network_inputs, "output_chunk_length": network_outputs}
parameter_candidates["RNNModel"] = {
    "input_chunk_length" : network_inputs,
    "output_chunk_length" : network_outputs,
    "model" : ["LSTM"],
    "n_rnn_layers" : [2],
    "n_epochs" : [200]
}
parameter_candidates["ExponentialSmoothing"] = {"seasonal": season_values}
parameter_candidates["FourTheta"] = {"season_mode": season_values}
parameter_candidates["Theta"] = {"season_mode": season_values}
#for model_name in ["AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", "NaiveSeasonal", "Prophet"]: #Prohet does not work. Removed it.
for model_name in ["AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", "NaiveSeasonal","KalmanForecaster"]:
        parameter_candidates[model_name] = {}


# ##########################    
# HS: Here we include the LSS, NLSS
# LSS
#220415
# latent_dims = [5, 10]
# parameter_candidates['LSS'] = {
#     "Dz" : latent_dims
# }

# LSS_Takens
#220415
latent_dims_takens = [5, 10]
parameter_candidates['LSS_Takens'] = {
    "Dz" : latent_dims_takens
}

# NLSS
#220415
# latent_dims = [5, 10]
# kernel_numbers = [5, 30]
# parameter_candidates['NLSS'] = {
#     "Dz" : latent_dims,
#     "Dk" : kernel_numbers
# }

# NLSS_Takens
#220415
latent_dims_takens = [5, 10]
kernel_numbers_takens = [5, 30]
parameter_candidates['NLSS_Takens'] = {
    "Dz" : latent_dims_takens,
    "Dk" : kernel_numbers_takens
}

# # NLSS_Noiseless
# parameter_candidates['NLSS_Noiseless'] = {
#     "Dz" : latent_dims,
#     "Dk" : kernel_numbers
# }

# RBF_Takens
#220722
# latent_dims_takens = [5, 10]
# kernel_numbers_takens = [5, 30]
# parameter_candidates['RBF_Takens'] = {
#     "Dz" : latent_dims_takens,
#     "Dk" : kernel_numbers_takens
# }

# NLSS_Takens
#220415
# latent_dims_takens = [5, 10]
# kernel_numbers_takens = [5, 30]
# 220312
# latent_dims_takens = [5, 10]
# kernel_numbers_takens = [5, 20]
# 220315
# latent_dims_takens = [3]
# kernel_numbers_takens = [30]
# 220407
# latent_dims_takens = [5, 10, 20]
# kernel_numbers_takens = [5, 30]
#220415
#latent_dims_takens = [5, 10]
#kernel_numbers_takens = [5, 30]

##########################    

#     
for equation_name in equation_data.dataset: #Chao data
    # The following models does not work in the existing models (e.g., ARIMA). Excluded.
    if equation_name in ['GenesioTesi','Hadley','SprottD','StickSlipOscillator']:
       continue
    
    print(equation_name, flush=True)
    
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))
    noise_scale = np.std(train_data[:int(5/6 * len(train_data))]) # prevent leakage
    # noiseoriginal 0.2 
    #220812_results_ #for the third presentation Small noise 0.2
    #train_data += 0.2 * np.std(train_data) * np.random.normal(size=train_data.shape[0])

    #221025_results_ #for the third presentation Large noise 0.8
    train_data += 0.8 * np.std(train_data) * np.random.normal(size=train_data.shape[0])

    # no noise
    #train_data += 0 * np.std(train_data) * np.random.normal(size=train_data.shape[0])

    if equation_name not in all_hyperparameters.keys():
        all_hyperparameters[equation_name] = dict()
    
    split_point = int(5/6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
    
    for model_name in parameter_candidates.keys():
        print(equation_name + " " + model_name)
        if SKIP_EXISTING and model_name in all_hyperparameters[equation_name].keys():
            if model_name in ['']: #['NLSS_Takens','LSS_Takens','RBF_Takens']: #,,'NLSS','LSS','KalmanForecaster'
                #print(f"{model_name} exists, but forced to re-fit")
                
                #Turn on the following if we want to skip
                print(f"{model_name} exists, skipped")
                #continue
            else:
                print(f"Entry for {equation_name} - {model_name} found, skipping it.")
                #continue
        
        model = getattr(darts.models, model_name)
        model_best = model.gridsearch(parameter_candidates[model_name], y_train_ts, val_series=y_test_ts,
                                     metric=darts.metrics.mse)
        
        best_hyperparameters = model_best[1].copy()
        
        # Write season object to string
        for hyperparameter_name in best_hyperparameters:
            if "season" in hyperparameter_name:
                best_hyperparameters[hyperparameter_name] = best_hyperparameters[hyperparameter_name].name
        
        all_hyperparameters[equation_name][model_name] = best_hyperparameters

    with open(output_path, 'w') as f:
        json.dump(all_hyperparameters, f, indent=4)
        print('wrote')
    # data is written in hyperparameters_train_univariate__pts_per_period_15__periods_12.json    
    