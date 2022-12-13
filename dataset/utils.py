#from sklearn import preprocessing
#from sklearn.metrics import mean_squared_error as mse
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
#import scipy.stats
#from scipy.signal import find_peaks
#import statsmodels.api as sm

#def find_lagmin(defect):
#    autocorr_i = sm.tsa.acf(defect,nlags=len(defect))/np.var(defect)
    #autocorr_i = autocorr_i/autocorr_i[0]
#    try:
#        autocorr_i = autocorr_i[:np.where(autocorr_i<0)[0][0]]
#    except:
#        continue
#    return np.argmin(np.abs(autocorr_i-1/np.exp(1)))

#rms = lambda x_seq: (sum(x*x for x in x_seq)/len(x_seq))**(1/2)
#def autocorr_vector(X):
#    autocorr_values = np.array([find_lagmin(X[i]) for i in range(len(X))])
from constants import * 
import pandas as pd 

import numpy as np
from scipy.signal import hilbert

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



        