"""
Functions for preparing dataframe for models
"""

#import libraries
import pandas as pd
import numpy as np

### ---------- GLOBAL VARIABLES -----------------###
# choose which variables to include in final dataset

INST_VEL = ['VELOC:0','VELOC:1','VELOC:2']
AVG_VEL = ['AVVEL:0','AVVEL:1','AVVEL:2']
POSITIONS = ['Points:0', 'Points:1', 'Points:2', 'delta']
PRESSURE  = ['PRESS']
GRADIENTS = ['Gradients:0','Gradients:1','Gradients:2','Gradients:3',
             'Gradients:4','Gradients:5','Gradients:6','Gradients:7',
             'Gradients:8']
TO_PREDICT = ['u_plus']
REYNOLDS = ['viscosity','Local_Re','Local_Re_Avg','Local_Re_y',
        'Local_Re_log', 'Log_delta', 'y_delta']
OTHERS = ['wall_shear','u_tau','y_plus', 'TURBU']

#!!! URGENT
# Specify viscosity of flow. This is used if dataset does not contain viscosity
visco_180 = 3.547e-4
visco_1000 = 5.3566e-5

FLOW_VISCOSITY = visco_1000

# y_plus include to avoid error in plotting. Ensure it is not in the final set
#REQUIRED_COLUMNS = INST_VEL + GRADIENTS + AVG_VEL + POSITIONS + TO_PREDICT + REYNOLDS + ['y_plus']
#REQUIRED_COLUMNS = ['VELOC:0', 'AVVEL:0','Points:0', 'Points:1', 'Points:2','u_plus','Local_Re','delta', 'Local_Re_log','y_plus', 'Log_delta','Local_Re_Avg','PRESS']

#For tests
REQUIRED_COLUMNS = ['u_plus','y_plus','delta','Points:1','AVVEL:0','VELOC:0','Local_Re','Local_u_tau','Local_Re_log']
# function to correct 0's when calculating logs
"""
def calc_log(x):
    if x == 0:
        return np.log(0.001)
    else:
        return np.log(x)
"""

def calc_log(x):
    result = np.log(x)
    if result < 0:
        return 0
    else:
        return result


# Function to prepare dataframe and return clean dataframe ready for model
def clean_dataframe(data, min_vars = True):
    if "viscosity" not in data.columns:
        data['viscosity'] = FLOW_VISCOSITY
    assert "viscosity" in data.columns, "viscosity not found in dataframe"
    data['delta_squared'] = data['delta']**2
    #data['delta'] = data.delta + 0.001
    data['Local_Re'] = data['VELOC:0'] * data['delta'] / data.viscosity
    data['Local_u_tau'] = np.sqrt(abs(data.wall_shear))
    data['Local_u_plus'] = data['VELOC:0'] / data['Local_u_tau'] 
    #data['Local_Re'] = data.Local_Re + 0.001
    data['Local_Re_Avg'] = data['AVVEL:0'] * data['delta'] / data.viscosity
    data['Local_Re_y'] = data['VELOC:0'] * data['Points:1'] / data.viscosity
    data['Local_Re_log'] = data.Local_Re.apply(calc_log)
    data['Log_delta'] = data.delta.apply(calc_log)
    data['y_delta'] = data['Points:1'] / data.delta
    #data['pu_tau'] = np.sqrt(abs(data.wall_shear))
    if min_vars:
        data = data[REQUIRED_COLUMNS]
        msg = "Error in subsetting required columns"
        assert list(data.columns)== REQUIRED_COLUMNS, msg
    # just to drop delta. can be changed
    #data = data.drop(columns=['delta'])
    return data

# Function to get inputs and response from clean-dataframe
# drop delta if not required as input
def X_Y_split(dataframe):
    return dataframe.drop(columns=TO_PREDICT + ['y_plus']), dataframe[TO_PREDICT]
