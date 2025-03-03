#!/usr/bin/python
# To run this code
# CUDA_VISIBLE_DEVICES=0 python compute_benchmarks_noise_fine.py 

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
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--pnlss_only', action='store_true', help='Run pnlss models only')
args = parser.parse_args()


cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
## link to TEST data

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/220812_results_" + dataname + "_noise.json"
#240605_results_ #for the third presentation Large noise 0.8 old hyperparam
#240606_results_ #for the third presentation Large noise 0.2 old hyperparam

#221025_results_ #for the third presentation Large noise 0.8
#220812_results_ #for the third presentation Small noise 0.2
#220428_results_ #for the second presentation
dataname = dataname.replace("test", "train" )
hyperparameter_path = cwd + "/hyperparameters/220812_hyperparameters_" + dataname + ".json"
#221025_results_ #for the third presentation Large noise 0.8
#220812_results_ #for the third presentation Small noise 0.2
#220428_hyperparameters_ #for the second presentation

metric_list = [
    'coefficient_of_variation',
    'mae',
    'mape',
    'marre',
    #'mase', # requires scaling with train partition; difficult to report accurately
    'mse',
    #'ope', # runs into issues with zero handling
    'r2_score',
    'rmse',
    #'rmsle', # requires positive only time series
    'smape'
]

equation_data = load_file(input_path)

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
    

#for equation_name in list(equation_data.dataset.keys())[::-1]:
for equation_name in list(equation_data.dataset.keys())[::+1]:
    # The following models does not work in the existing models (e.g., ARIMA). Excluded.
    if equation_name in ['GenesioTesi','Hadley','MacArthur','SprottD','StickSlipOscillator']:
        continue
    
    # if equation_name != 'BeerRNN':
    #     continue

    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))
    noise_scale = np.std(train_data[:int(5/6 * len(train_data))]) # prevent leakage
    
    #220812_results_ #for the third presentation Small noise 0.2
    train_data += 0.2 * np.std(train_data) * np.random.normal(size=train_data.shape[0])
    
    #221025_results_ #for the third presentation Large noise 0.8
    #240605_results_ #for the third presentation Large noise 0.8
    #train_data += 0.8 * np.std(train_data) * np.random.normal(size=train_data.shape[0])

    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    
    split_point = int(5/6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
    
    all_results[equation_name]["values"] = np.squeeze(y_val)[:-1].tolist()
    
    for model_name in all_hyperparameters[equation_name].keys():
        if model_name in all_results[equation_name].keys():
            #continue
            if not args.pnlss_only:
                if model_name in ['LSS_Takens','NLSS_Takens']:
                    print(f"{model_name} exists, but forced to re-fit")
                else:
                    print(equation_name + " " + model_name, flush=True)
                    continue


        # if model_name == "NLSS_Sampling": #"NLSS":
        #     print(f"{model_name} exists in hyperparam search, but skips")
        #     continue

        all_results[equation_name][model_name] = dict()
        
        print(equation_name + " " + model_name, flush=True)
        
        # look up season object from string
        for hyperparameter_name in all_hyperparameters[equation_name][model_name]:
            if "season" in hyperparameter_name:
                old_val = all_hyperparameters[equation_name][model_name][hyperparameter_name]
                all_hyperparameters[equation_name][model_name][hyperparameter_name] = getattr(darts.utils.utils.SeasonalityMode, old_val)
    
        model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])
        #model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])
        if hasattr(model, "force_reset"):
            all_hyperparameters[equation_name][model_name]["force_reset"] = True
            model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])

        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val))
        pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))
        
        all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred.values()).tolist()
        
        for metric_name in metric_list:
            print(metric_name)
            
            metric_func = getattr(darts.metrics.metrics, metric_name)

            print(metric_func(true_y, pred_y))
            all_results[equation_name][model_name][metric_name] = metric_func(true_y, pred_y)
        
        with open(output_path, 'w') as f:
            print(f)
            json.dump(all_results, f, indent=4, sort_keys=True)   
        



