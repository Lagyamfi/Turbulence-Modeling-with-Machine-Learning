"""
function for working with dataframe and making modifications
"""
# libraries
import pandas as pd
import numpy as np
#import clean_dataframe as cd


import warnings
warnings.filterwarnings("ignore")

### GLOBAL PARAMETERS #####

# new parameters to be modified
deltas = [1e-3,0.1, 100, 1e3]
viscos = [1e-4,0.1, 100, 1e5]

class DataFile(object):
    """
    Class for a datafile giving allowing to perform additional functions directly.
    Default input: 
    Note: 
    (V)iscosity is one value for the whole datafile
    (v)iscosity is the corresponding viscosity value for each row
    """
    def __init__(self,dataframe=None, file_path=None, viscosity=None):
        if not(dataframe is None):
            self.dataframe = dataframe
        elif file_path != None:
            self.dataframe = pd.read_csv(file_path)
        # fill viscosity column in dataframe
        if "viscosity" in self.dataframe.columns:
            if viscosity != None:
                msg = "Viscosity provided different from viscosity value in datafile provided!"
                assert dataframe.viscosity[0] == viscosity, msg
            else:
                # check that one unique value of viscosity present for all dataset
                assert len(np.unique(dataframe.viscosity)) == 1, "more than 1 value of viscosity found."
            self.Viscosity = self.dataframe.viscosity
        else:
            assert viscosity != None, "viscosity value required for datafile"
            self.Viscosity = viscosity
            self.dataframe['viscosity'] = self.Viscosity
        
        # fill height of channel
        assert "delta" in self.dataframe.columns, "delta value missing"
        self.delta = np.max(dataframe.delta)
            
    def __call__(self):
        return self.dataframe
        
    def __str__(self):
        pass
    
    def __repr__(self):
        pass
    
    def modify_viscosity(self, new_viscosity, inplace=False):
        """
        Modify viscosity of dataset for new provided viscosity.
        Specify inplace to get dataset modified without a return, or no
        """
        temp_df = modify_viscosity(self.dataframe,self.Viscosity, new_viscosity)
        if inplace:
            self.Viscosity = new_viscosity
            self.dataframe = temp_df
        else:
            return temp_df
        
    def modify_delta(self,new_delta, inplace=False):
        temp_df = modify_delta(self.dataframe, self.delta, new_delta)
        if inplace:
            self.delta = new_delta
            self.dataframe = temp_df
        else:
            return temp_df
    
    def get_modified_sample(self, sample_size=10000, new_delta=None, new_viscosity=None, keep_original=False, inclusive=True):
        modified_sample = return_modified_sample(self, sample_size=sample_size, new_delta=new_delta,
                                                 new_viscosity=new_viscosity,keep_original=keep_original, inclusive=inclusive)
        return modified_sample



## ------- helper functions -----------------

def modify_viscosity(dataframe_main, viscosity_old, viscosity_new):
    """
    Given a new viscosity value, replace old viscosity with this value in a dataframe.
    Scale velocity and u_tau accordingly using helper functions.
    """
    dataframe = dataframe_main.copy()
    dataframe['viscosity'] = viscosity_old
    ratio = viscosity_new / viscosity_old
    scale_velocity(dataframe, ratio, inplace=True)
    scale_utau(dataframe, ratio, inplace=True)
    #dataframe['u_tau'] = dataframe['u_tau'] * ratio
    dataframe['viscosity'] = viscosity_new
    dataframe['u_plus'] = dataframe['VELOC:0'] /  dataframe['u_tau']
    dataframe['y_plus'] = dataframe['delta'] * dataframe['u_tau'] / dataframe['viscosity']
    return dataframe

# function to rescale velocities
def scale_velocity(dataframe, ratio, inplace=True):
    """
    Cycle through all velocity parameters and scale them based on a ratio of old and new
    values of any parameter (most likely delta and viscosity).
    """
    veloc_list = ['VELOC:0','VELOC:1','VELOC:2','AVVEL:0','AVVEL:1','AVVEL:2']
    for veloc in veloc_list:
        if veloc in dataframe.columns:
            dataframe[veloc] = dataframe[veloc] * ratio
    if inplace == False:
        return dataframe

def scale_utau(dataframe, ratio , inplace=True):
    """
    scale u_tau based on a ratio of old and new values of any parameter
    (most likely delta and viscosity).
    """
    dataframe['u_tau'] = dataframe['u_tau'] * ratio
    if inplace == False:
        return dataframe


#function to get correct delta based on symetry of setup
def new_y(old_y,delta_new):
    if old_y > delta_new:
        return 2*delta_new - old_y
    else:
        return old_y

def modify_delta(dataframe_main, delta_old, delta_new):
    """
    Given a new delta (half channel) value, replace old delta with this value in a dataframe.
    Scale velocity and u_tau accordingly using helper functions.
    Recalculate new delta.
    """
    dataframe = dataframe_main.copy()
    ratio = delta_old / delta_new
    #dataframe['viscosity'] = viscosity_original
    scale_velocity(dataframe, ratio, inplace=True)
    dataframe['Points:1'] = dataframe['Points:1'] * (delta_new/delta_old)
    scale_utau(dataframe, ratio, inplace=True)
    dataframe['u_plus'] = dataframe['VELOC:0'] / dataframe['u_tau']
    dataframe['delta'] = dataframe['Points:1'].apply(new_y, args=(delta_new,))
    dataframe['y_plus'] = dataframe['delta'] * dataframe['u_tau'] / dataframe['viscosity']
    return dataframe


#Function to get samples
def get_complete_sample(path=None, FILE=None):
    """
    Given a path of csv files, return a single csv file containing a sample of each the individual files.
    Usually these individual files will be files of the same flow but with some parameters modified.
    This new complete sample is what will be used for training models.

    If single file is provided, only a modified sample of the file is returned.
    """
    if path:
        files = glob.glob(path + "/*.csv")
    else:
        files = [FILE]
    data_list = []
    #iterate through all data files and generate sampled and modified data
    for file in files:
        dataframe = pd.read_csv(file)
        data_list.append(return_modified_sample(dataframe))
    final_df = pd.concat(data_list, axis=0, ignore_index=True)
    return final_df


# function to get sample from dataframe
def return_modified_sample(dataframe, sample_size=10000, new_delta=None, new_viscosity=None,keep_original=False, inclusive=True):
    """
    Given a dataframe of custom Datafile type, iterate through a "previously defined list of new deltas and 
    viscosities(this to be modified in future to be added as an argument)" and return
    a final dataframe comprising samples (default sample size=10000) for each of the newly
    generated data.
    Specify inclusive to include sample drawn from original data as part of the final return dataset
    """
    final = []

    data = dataframe.dataframe.sample(sample_size)
    #data['viscosity'] = viscosity_original
    if keep_original:
        final.append(dataframe.dataframe)
    if inclusive:
        final.append(data)
    # iterate through list of new deltas and viscosity and modify sampled data
    if not(new_delta is None):
        for delta in new_delta:
            mod_data = modify_delta(data, dataframe.delta, delta)
            final.append(mod_data)
    if not(new_viscosity is None):
        for viscosity in new_viscosity:
            mod_data = modify_viscosity(data, dataframe.Viscosity, viscosity)
            final.append(mod_data)
    #combine original and modified data    
    final_df = pd.concat(final, axis=0, ignore_index=True)
    return final_df
