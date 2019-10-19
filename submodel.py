# script to :
    # run through all the dataset (snapshots of the flow)
    # build a model and train it
    # save model

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib, glob
from clean_dataframe import X_Y_split, clean_dataframe
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from functions import XGBoost_Model

import sys, os, shutil
from datetime import datetime
import time

default_params = dict(max_depth=7,
                      colsample_bytree=0.9,
                      subsample = 0.9,
                      learning_rate=0.1,
                      objective="reg:squarederror",
                      n_estimators =1000,
                      silent=0,
                      reg_alpha=10,
                      reg_lambda=10,
                      gamma=10,
                      booster='gbtree')

# some house cleaning before starting

WORKING_FOLDER = os.getcwd()
SOURCE_FILES = ""
MODELS = WORKING_FOLDER +"/MODELS"

if not os.path.isdir(MODELS):
    os.mkdir(MODELS)
    print("directory for models created")

# get path to files
try:
    path = sys.argv[1]
except:
    print("Specify source location!")
    sys.exit(1)

files = glob.glob(path +"/*.csv")
num_files = len(files)
print("Number of files found: %d" %num_files)

file_count = 0

# iterate through all files
for file in files:
    filename = os.path.split(file)[1]
    # set up save space for file
    save_path = MODELS +"/%s" %filename
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    #get start time
    file_start_time = datetime.now()
    print("===========================================================")
    print("Total Files\t:\t %d "%num_files)
    print("Completed\t:\t %d" %file_count)
    print("Remaining\t:\t %d" %(num_files - file_count))
    print("===========================================================\n")
    print("Current file\t: \t%s" %filename)
    print("start time\t: \t%s\n" %file_start_time.strftime("%D - %H:%M:%S"))
    print("Creating DataFrame object...\n")
    dataframe = pd.read_csv(file)

    print("Cleaning Dataframe and keeping only required columns...\n")
    clean_data = clean_dataframe(dataframe)
    # for test
    clean_data = clean_data.sample(frac=0.01)
    # create xgboost model
    model_test = XGBoost_Model(default_params, clean_data)

    # train_model
    train_start_time = datetime.now()
    print("->Starting Model Training at %s...\n" %train_start_time.strftime("%D - %H:%M:%S"))
    fitted_model = model_test.fit(n_estimators=10, save_plot=True, save_path=save_path)
    results = fitted_model.evals_result_

    train_end_time = datetime.now()
    print("\n->Completed Model Training at %s...\n" %train_end_time.strftime("%D - %H:%M:%S"))
    print("Total time for training : %d" %(train_end_time - train_start_time).total_seconds())

    #save model and results
    print("\nSaving Model and evaluation results for %s\n" %filename)
    joblib.dump(fitted_model, save_path+"/%s.mdl"%filename)
    joblib.dump(results, save_path+"/results_%s.rsl"%filename)


    #increment file count
    file_count += 1
