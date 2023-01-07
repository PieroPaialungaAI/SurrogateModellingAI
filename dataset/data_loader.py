from dataset.utils import * 
from dataset.constants import * 
import pandas as pd 
import numpy as np
import os
from scipy.signal import hilbert
import warnings
warnings.filterwarnings("ignore")

def Dataset(start_signal =START_SIGNAL, data_path=DATA_PATH,augment=True):
    print('Importing the dataset...\n')
    X_right = np.array(pd.read_csv(data_path+'/'+'right_profile_defects.csv').drop('Unnamed: 0',axis=1))
    X_left = np.array(pd.read_csv(data_path+'/'+'left_profile_defects.csv').drop('Unnamed: 0',axis=1))
    Y = np.array(pd.read_csv(data_path+'/'+'target_scans.csv').drop('Unnamed: 0',axis=1))
    main_peak = first_peak_vector(Y)
    print('Preprocessing the dataset...\n')
    X_right = trim_X(X_right)
    X_energy = energy_X(X_right)
    X_left = trim_X(X_left)
    Y = trim_signal(main_peak,Y,start_signal)
    if augment==True:
        print('Using augmented data... \n')
        X_right_bonus = np.array(pd.read_csv(data_path+'/bonus_right_profiles.csv').drop('Unnamed: 0',axis=1))
        X_left_bonus = np.array(pd.read_csv(data_path+'/bonus_left_profiles.csv').drop('Unnamed: 0',axis=1))
        Y_bonus = np.array(pd.read_csv(data_path+'/bonus_target_scans.csv').drop('Unnamed: 0',axis=1))
        X_energy_bonus = energy_X(X_right_bonus)
        print('Preprocessing the dataset...\n')
        X_right_bonus = trim_X(X_right_bonus)
        X_left_bonus = trim_X(X_left_bonus)
        main_peak_bonus = first_peak_vector(Y_bonus)
        Y_bonus = trim_signal(main_peak_bonus,Y_bonus,start_signal)
        print('Merging the dataset...\n')
        X_right = merge_data(X_right,X_right_bonus)
        X_left = merge_data(X_left,X_left_bonus)
        X_energy = merge_data(X_energy,X_energy_bonus)
        Y = merge_data(Y,Y_bonus)
        main_peak = merge_data(main_peak,main_peak_bonus)
    X_rnn = X_rnn_builder(X_right)
    last_string = data_path.split('/')[-1]
    angle = float(last_string.split('_')[0])
    angle_data = np.array([angle]*len(Y))
    try:
        energy_and_angle = np.vstack((angle_data,X_energy[:,0])).T
    except:
        energy_and_angle = np.vstack((angle_data,X_energy)).T
    res = {'RNN Data':X_rnn, 'Full Model Target': Y, 'Right Profile Data': X_right, 'Left Profile Data': X_left,
           'Main Peak Data':main_peak,'Energy and Angle Data':energy_and_angle,'Augmented Data': augment}
    return res

def FullDataset(full_data_path=DATA_PATH):
    folders = extract_folders(DATA_PATH)
    try:
        first_dataset = Dataset(data_path = DATA_PATH+'/'+folders[0],augment=False)
    except:
        print('No augmented data found...\n')
        first_dataset = Dataset(data_path = DATA_PATH+'/'+folders[0],augment=False)
    print('Building whole dataset...\n')
    for f in folders[1::]:
        print('Extracting data from '+f)
        try:
            print('Found bonus data!...\n')
            second_dataset = Dataset(data_path = DATA_PATH+'/'+f,augment=False)
            first_dataset = merge_datasets(first_dataset,second_dataset)
        except:
            second_dataset = Dataset(data_path = DATA_PATH+'/'+f,augment=False)
            first_dataset = merge_datasets(first_dataset,second_dataset)
    return first_dataset

    
