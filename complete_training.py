#add script_utils folder to sys to import custom modules
import sys, os, shutil
sys.path.insert(0,"/Users/Lawrence/Documents/MACHINE_LEARNING/INTERNSHIP/Turbu/SANDBOX/scripts_utils")

import pandas as pd
import numpy as np
from clean_dataframe import clean_dataframe, X_Y_split
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from visualize_results import display_results

#** DEFINE STORAGE LOCATIONS
WORKING_FOLDER = os.getcwd()
MODEL_FOLDER = WORKING_FOLDER + "/MODELS"
TRAIN_FOLDER = WORKING_FOLDER + "/TRAINING_DATA "

# check if folders exits
if not os.path.isdir(MODEL_FOLDER):
    print("Creating directory to save Models in...")
    os.mkdir(MODEL_FOLDER)
if not os.path.isdir(TRAIN_FOLDER):
    print("Creating directory to save training data...")
    os.mkdir(TRAIN_FOLDER)

train_file = " "

try:
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
        print("Using file provided at command line.")
    else:
        train_file = train_file
        print("Using file provided in source file.")
except FileNotFoundError:
    print("Unable to get main file for taining, check path!")
    sys.exit(1)

# read file into dataframe
raw_file = pd.read_csv(train_file)

raw_file = raw_file.sample(frac=0.01)

# *************** PREPARE DATAFRAME ********************************
print("Preparing dataframe...")

#clean file and keep only required columns for building model
clean_file = clean_dataframe(raw_file)

# split train file into train and validation
train_data, validation_data = train_test_split(clean_file, test_size=0.3, random_state=10)

# get mini sample to used for tuning
mini_train_data = train_data.sample(frac=0.1)

# split data into input and response
train_x, train_y = X_Y_split(train_data)
mini_train_x, mini_train_y = X_Y_split(mini_train_data)
validation_x, validation_y = X_Y_split(validation_data)

# keep copy of full data for final fitting before saving model
full_x, full_y = X_Y_split(clean_file)

"""
# convert to DMatrix for use with xgboost
train_mat = xgb.DMatrix(train_x, label=train_y,feature_names=train_x.columns)
validation_mat = xgb.DMatrix(validation_x, label=validation_y,feature_names=validation_x.columns)
"""
##*********** PERFORM INITIAL TRAINING ********************************
print("Preparing to commence training...")

# specify default parameters for model
default_params = dict(max_depth=len(train_x.columns),
              colsample_bytree=0.8,
              subsample = 0.8,
              learning_rate=0.4,
              objective="reg:squarederror",
              n_estimators =100,
              silent=0,
              reg_alpha=0,
              reg_lambda=0,
              booster='gbtree')

# define evaluation set 
eval_set = [(train_x, train_y), (validation_x, validation_y)]
#eval_mat = [(train_mat, "train"), (validation_mat, "validation")]

# Set up Model
model = xgb.XGBRegressor(**default_params)
initial_model  = model.fit(train_x, train_y, eval_set=eval_set, eval_metric="rmse", early_stopping_rounds=100,
                          verbose=False)

# TEST save model
print("Fitting trained model on full dataset...\n")
initial_model = initial_model.fit(full_x, full_y,)

# save model
print("saving model...\n")
joblib.dump(initial_model, f"{MODEL_FOLDER}/initial_model")

print(f"model saved at {MODEL_FOLDER}!")

## *** TUNING MODEL ****
# define parameter space to be used for tuning
# Tuning parameters
parameters = {
    'learning_rate': np.linspace(0.01,0.6,10),
    'max_depth' : np.arange(5,len(train_x.columns),1),
    'subsample' : np.linspace(0.8,1.0,3),
    'colsample_by_tree' : np.linspace(0.8 ,1.0,3),
    'n_estimators' : np.arange(100, 1000,200),
    'reg_alpha' : np.linspace(0,100,num=5),
    'reg_lambda' : np.linspace(0,100, num = 5),
    'gamma' : np.linspace(0,40, num = 5),
}

# set up randomizedsearch for good parameters with tuning
print("starting model tuning...")
tuned_model = RandomizedSearchCV(initial_model,parameters,n_iter=100, n_jobs=1,
                        scoring="neg_mean_squared_error", cv=5)
tuned_model.fit(mini_train_x, mini_train_y, verbose=False)

# write grid results to file
print("Model Tuning completed, writing grid results to file...\n")
grid_results = pd.DataFrame(tuned_model.cv_results_)
grid_results = grid_results.sort_values(by="mean_test_score", ascending=False)
grid_results.to_csv(f"{TRAIN_FOLDER}/tuning_grid_results.csv", header=True, index=False)

# evaluate tuned model
print("Fitting tuned model to train data and evaluating with validation data")
final_estimator = tuned_model.best_estimator_
final_model = final_estimator.fit(train_x, train_y, eval_set=eval_set, eval_metric="rmse",early_stopping_rounds=50, verbose=False)

# save tuned model
print("saving tuned  model...\n")
joblib.dump(tuned_model, f"{MODEL_FOLDER}/tuned_model")

print(f"model saved at {MODEL_FOLDER}!")

# fit final model to entire dataset
print("Fitting final model to entire dataset...")
final_model = final_model.fit(full_x, full_y,)

# save model
print("saving final model...")
joblib.dump(final_model, f"{MODEL_FOLDER}/final_model")

