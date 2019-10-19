
#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib, time
from clean_dataframe import X_Y_split, clean_dataframe
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from functions import XGBoost_Model
from datetime import datetime

import sys, os, shutil

try:
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
except FileNotFoundError:
    print("Unable to get main file for training, check path!")
    sys.exit(1)

# some house cleaning before starting

WORKING_FOLDER = os.getcwd()
MODELS = WORKING_FOLDER + "/DATA_MODELS"
SUPERCEDED_PATH = WORKING_FOLDER +"/SUPERCEDED"

if not os.path.isdir(MODELS):
    os.mkdir(MODELS)
#raw_file = pd.read_csv("./Experiments/exp_1/test_data/test_1/test_1.csv")
raw_file = pd.read_csv(data_file)
clean_data = clean_dataframe(raw_file)

# default parameters
default_params = dict(max_depth=7,
                      colsample_bytree=0.9,
                      subsample = 0.9,
                      learning_rate=0.4,
                      objective="reg:squarederror",
                      n_estimators =1000,
                      silent=0,
                      reg_alpha=10,
                      reg_lambda=10,
                      gamma=10,
                      booster='gbtree')

# data ranges
ranges = [0.01, 0.05, 0.1]

for subset in ranges:
    subset_name = "subset_%s"%subset
    # create directory for details of subset
    subset_path = MODELS + "/%s"%subset_name
    if not os.path.isdir(subset_path):
        os.mkdir(subset_path)
    #create subset by sampling main data
    subset_data = clean_data.sample(frac=subset)

    # confirm shape of subset dataset
    print("Shape of data being used for training is :")
    print(subset_data.shape)
    #create xgboost model object
    model_test = XGBoost_Model(default_params, subset_data )

    #train model
    start_time = time.time()
    print("Training Model for %s\n" %subset_name)
    fitted_model = model_test.fit(n_estimators=20, save_plot=True,
                                  save_path=subset_path)
    results = fitted_model.evals_result_
    #save model and results
    print("\nSaving Model and evaluation results for %s\n" %subset_name)
    joblib.dump(fitted_model, subset_path+"/%s.mdl"%subset_name)
    joblib.dump(results, subset_path+"/results_%s.rsl"%subset_name)
    print("Completed in %d" %(time.time() - start_time))
    print("\n===================================================\n")


print("TRAINING COMPLETED")
