from dataset.constants import * 
import pandas as pd 
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import simps
import os

def first_peak_vector(Y):
    return np.array([np.abs(hilbert(y)).max() for y in Y])

def trim_signal(first_peak_real,Y,start=START_SIGNAL):
    return np.array([Y[i][start:start+LENGTH_SIGNAL]/first_peak_real[i] for i in range(len(Y))])

def X_rnn_builder(X):
    return X[:,:].reshape(len(X),24,-1)

def trim_X(X):
    return np.array(X)[:,:-1]

def merge_data(array_1,array_2):
    array_1,array_2 = pd.DataFrame(array_1),pd.DataFrame(array_2)
    return np.array(array_1.append(array_2))

def energy_X(X):
    return np.array([simps(X[i]) for i in range(len(X))])

def extract_folders(file_loc):
    folder = file_loc
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    data_folders = []
    for sub_folder in sub_folders:
        if len(sub_folder.split('.'))==1:
            data_folders.append(sub_folder)
    return data_folders

def merge_datasets(dataset_1,dataset_2):
    key_data = dataset_1.keys()
    merged_dataset = []
    for k in key_data:
        data_2 = dataset_2[k]
        data_1 = dataset_1[k]
        merged_dataset.append(np.vstack((data_1[k],data_2[k])))
    res = {key_data[i]: merged_dataset[i] for i in range(len(key_data))}
    return res


        