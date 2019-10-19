import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from clean_dataframe import X_Y_split, clean_dataframe
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from functions import XGBoost_Model
from datetime import datetime
import json, pickle, codecs


import sys, os, shutil

# some house cleaning before starting

WORKING_FOLDER = os.getcwd()
PLOTS = WORKING_FOLDER + "/PLOTS"
SUPERCEDED_PATH = WORKING_FOLDER +"/SUPERCEDED"
today  = datetime.now()

if not os.path.isdir(PLOTS):
    os.mkdir(PLOTS)
    print("directory for plots created")

day = today.strftime("%d%h")
hour = str(today.hour)
minute = str(today.minute)

plot_save_path = PLOTS + "/%s_%s_%s" %(day, hour, minute)
if os.path.isdir(plot_save_path):
    print("Directory for plot exists, moving to superceded directory")
    if not os.path.isdir(SUPERCEDED_PATH):
        os.mkdir(SUPERCEDED_PATH)
    shutil.move(plot_save_path, SUPERCEDED_PATH+"/%s_%s_%s" %(day, hour, minute))

#make new path
os.mkdir(plot_save_path)

print("Plots will be saved in %s "%plot_save_path)
try:
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
except FileNotFoundError:
    print("Unable to get main file for training, check path!")
    sys.exit(1)

print("Preparing Dataframe...\n")

#for testing
raw_file = pd.read_csv("./Experiments/exp_1/test_data/test_1/test_1.csv")

# read file into dataframe
#raw_file = pd.read_csv(data_file)

# clean dataframe
clean_data = clean_dataframe(raw_file)

# Define parameter search space
parameters = {
    'learning_rate': np.linspace(0.01,0.6,10),
    'max_depth' : np.arange(3,len(clean_data.columns),2),
    'subsample' : np.linspace(0.7,1.0,4),
    'colsample_by_tree' : np.linspace(0.7,1.0,4),
    'reg_alpha' : np.linspace(0,100,num=10),
    'reg_lambda' : np.linspace(0,100, num = 10),
    'gamma' : np.linspace(0,100, num = 10),
}

default_params = dict(max_depth=5,
              colsample_bytree=1,
              subsample = 1,
              learning_rate=0.08,
              objective="reg:squarederror",
              n_estimators =500,
              silent=0,
              reg_alpha=0,
              reg_lambda=10,
              booster='gbtree')

model_test = XGBoost_Model(default_params, clean_data )

# ****************** INDIVIDUAL PARAMETER SEARCHES***************************#
print("Starting Individual parameter search...")

print("RESULTS:\n")
individual_results = {}
for tuning_param in list(parameters.keys()):
    #n_iter = len(parameters.get(tuning_param))
    n_iter = 2
    results = model_test.tune_model_parameter(tuning_param, parameters.get(tuning_param), save_plot=True,
                                    save_path=plot_save_path, randomized=True,n_splits=2, data_size=0.01, n_iter=n_iter)
    individual_results[tuning_param] = results.cv_results_

# ****************** COMBINED PARAMETER SEARCHES***************************#

print("\nStarting combined parameter search...")

print("RESULTS:\n")
model_test.tune_all_parameters(parameters, n_iter=2, cv=4, data_size=0.01)
