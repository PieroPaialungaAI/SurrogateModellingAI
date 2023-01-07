import keras 
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
from dataset.constants import * 
from dataset.utils import * 
from dataset.data_loader import *
warnings.filterwarnings("ignore")
from metrics import * 


def make_predictions():
    print('Loading the dataset...\n')
    CNN_data = np.load('CNN_data.npy')
    RNN_data =np.load('RNN_data.npy')
    Y = np.load('Y_data.npy')
    main_peak = np.load('main_peak_data.npy')
    Energy_angle_data = np.load('energy_and_angle_data.npy')
    print('Preparing full scan data...\n')
    full_scan_data = [CNN_data,RNN_data,Energy_angle_data]
    print('Importing trained model...\n')
    full_peak_model = keras.models.load_model('fullscan_saved_model.h5')
    print('Running the model for the full scan...\n')
    Y_pred = full_peak_model.predict(full_scan_data)
    print('Importing the model for the main peak ...\n')
    main_peak_model = keras.models.load_model('mainpeak_saved_model.h5')
    print('Preparing the main peak data ...\n')
    main_peak_data = [RNN_data, Energy_angle_data]
    print('Running the model for the main peak...\n')
    main_peak_pred = main_peak_model.predict(main_peak_data)
    train_test_list = np.load('training_test_index.npy',allow_pickle=True).item()
    print('Exporting the training set and test set predictions...\n')
    train_list,test_list = train_test_list['Train'],train_test_list['Test']
    result= {'Y':Y,'Y_pred':Y_pred,'main_peak':main_peak,'main_peak_pred':main_peak_pred,'train_list':train_list,'test_list':test_list}
    np.save('result.npy',result,allow_pickle=True)
    return result
def extract_result():
    try:
        print('Result not found, generating new one... \n')
        result = np.load('result.npy',allow_pickle=True).item()
    except:
        print('Using previous results... \n')
        result = make_predictions()
    return result
    

if __name__=='__main__':
    results = extract_result()
    print('Plotting the results of the second peak...\n')
    second_peak_plot(results['Y'],results['Y_pred'],results['train_list'],results['test_list'])
    energy_angle_data = np.load('energy_and_angle_data.npy')
    X = np.load('CNN_data.npy')
    angle_data = energy_angle_data[:,0]
    print('Plotting 10 random examples...\n')
    plot_random_predictions(angle_data,X,results['Y'],results['Y_pred'],results['test_list'])
    print('Plotting 10 best examples...\n')
    plot_best_predictions(angle_data,X,results['Y'],results['Y_pred'],results['test_list'])
