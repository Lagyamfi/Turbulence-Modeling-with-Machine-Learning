"""
script to generate mixture of data based on ranges of viscosity and deltas
"""

import pandas as pd
import numpy as np
#from clean_dataframe import clean_dataframe
from datafile import DataFile
import datafile
import matplotlib.pyplot as plt
import joblib, glob
import sys, os, shutil
from datetime import datetime
import time

# function to add dimensionless values to dataset
def get_off_wall_point(dataframe, height=1):
    grid_height = len(dataframe.groupby('Points:1'))
    plane = grid_height**2
    sorted_dataframe = dataframe.sort_values(["Points:1", "Points:2","Points:0"])
    offWall_height = float(np.unique(sorted_dataframe.iloc[plane*height:plane*(height+1)]["Points:1"]))
    offWall_velocity = np.mean(sorted_dataframe.iloc[plane*height:plane*(height+1)]["VELOC:0"])
    return offWall_height, offWall_velocity

def add_data(data,off_wall_height=1):
    off_h, off_vel = get_off_wall_point(data, height=off_wall_height)
    off_h, off_vel = get_off_wall_point(data, height=off_wall_height)
    data['dim_y'] = data["Points:1"] / off_h
    data['dim_delta'] = data['delta']/ off_h
    data['dim_veloc'] = data["VELOC:0"] / off_vel
    return data



WORKING_FOLDER = os.getcwd()
# get path to files
try:
    path = sys.argv[1]
except:
    print("Specify source location!")
    sys.exit(1)

if len(sys.argv) > 2:
    sample_size = float(sys.argv[2])
else:
    sample_size = 0.1


# get base_file to be modified
file=path
filename = os.path.split(file)[1]

#get start time
file_start_time = datetime.now()


print("===========================================================\n")
print("Basefile\t: \t%s" %filename)
print("%d %% of each generated dataset will be sampled for final dataset! \n" %
        (sample_size * 100))
print("start time\t: \t%s\n" %file_start_time.strftime("%D - %H:%M:%S"))
print("Opening basefile...\n")
base_file = pd.read_csv(file)

base_file['viscosity'] = 5.3566e-5
#base_file['viscosity'] = 3.547e-4

print("Making delta modifications...\n")
old_delta = np.max(base_file['delta'])
# delta ranges
delta_range = [1e-1,2,5,1e1]
new_deltas = []
for delta in delta_range:
    print("current delta : %d\n" %delta)
    dataframe = datafile.modify_delta(base_file, old_delta, delta)
    dataframe = add_data(dataframe)
    new_deltas.append(dataframe)
print("Delta modifications completed. Concatenating files...\n")
complete_delta = pd.concat([i.sample(frac=sample_size) for i in new_deltas], ignore_index=True)

print("Shape of complete delta dataset is %d %d\n" %(complete_delta.shape))

print("Making viscosity modifications...\n")
old_viscosity = base_file.viscosity[0]
# viscosity ranges
viscosity_range = [1e-5,1e-3,1e1,1e4]
new_viscos = []
for visco in viscosity_range:
    print("current viscosity : %d\n" %visco)
    dataframe = datafile.modify_viscosity(base_file, old_viscosity, visco)
    dataframe = add_data(dataframe)
    new_viscos.append(dataframe)
print("Viscosity modifications completed. Concatenating files...\n")
complete_visco = pd.concat([i.sample(frac=sample_size) for i in new_viscos], ignore_index=True)
print("Shape of complete viscosity dataset is %d %d\n" %(complete_visco.shape))

print("Concatenating all files to make final dataset...\n")
complete_file = pd.concat([base_file.sample(frac=0.2),complete_visco, complete_delta], ignore_index=True)

print("Shape of complete dataset is %d %d\n" %(complete_file.shape))

print("Saving complete dataset...\n")

complete_file.to_csv("train_complete_1000.csv",  header=True, index=False)
