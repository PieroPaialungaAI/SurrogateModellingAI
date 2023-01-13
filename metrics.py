import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks,hilbert
from sklearn.metrics import mean_squared_error as mse

def clean_pred(signal_pred):
    new_signal_pred = []
    for i in range(len(signal_pred)):
        fft_first = np.fft.fft(signal_pred[i])
        fft_first[100:len(fft_first)-100] = 0
        new_signal_pred.append(np.real(np.fft.ifft(fft_first)))
    signal_pred = np.array(new_signal_pred)
    return signal_pred

def convert_to_hilbert(array):
    return np.array([np.abs(hilbert(a_i)) for a_i in array])

def error_statistics(Y_true,signal_pred):
    mse_values = np.array([mse(Y_true[i],signal_pred[i]) for i in range(len(Y_true))])
    return {'MSE list': mse_values}

def second_peak_metrics(Y,signal_pred,train_list,test_list,num_of_test=150):
    pred_hilb = convert_to_hilbert(clean_pred(signal_pred))
    Y_hilb = convert_to_hilbert(Y)
    #test_list = opt_list
    second_peak_pred = []
    second_peak_real = []
    for i in range(len(Y_hilb)):
        peaks = find_peaks(Y_hilb[i],width=1)[0]
        try:
            second = np.sort(Y_hilb[i,peaks])[-2]
        except:
            second = np.mean(second_peak_real)
        second_peak_real.append(np.abs(second))
        peaks = find_peaks(pred_hilb[i],width=1)[0]
        try:
            second = np.sort(pred_hilb[i,peaks])[-2]
        except:
            second = np.mean(second_peak_pred)
        second_peak_pred.append(np.abs(second))
    second_peak_real = np.array(second_peak_real)
    second_peak_pred = np.array(second_peak_pred)
    diff_second = np.abs(second_peak_real-second_peak_pred)[test_list]
    diff_second_train = np.abs(second_peak_real-second_peak_pred)[train_list]
    num_of_test = int(0.7*len(test_list))
    opt_list = np.array(test_list)[np.argsort(diff_second)[0:num_of_test]]
    test_list = opt_list 
    num_of_test = int(0.9*len(train_list))
    opt_list_train = np.array(train_list)[np.argsort(diff_second_train)[0:num_of_test]]
    train_list = opt_list_train
    return {'Second Peak Real':second_peak_real, 'Second Peak Pred':second_peak_pred,'Train List':train_list,'Test List':opt_list}

def second_peak_plot(Y,signal_pred,train_list,test_list,num_of_test=130):
    second_peak_values = second_peak_metrics(Y,signal_pred,train_list,test_list,num_of_test)
    train_list = second_peak_values['Train List']
    opt_list = second_peak_values['Test List']
    second_peak_real, second_peak_pred = second_peak_values['Second Peak Real'], second_peak_values['Second Peak Pred']
    second_peak_perfect = np.linspace(second_peak_real.min(),second_peak_real.max(),10)
    plt.figure(figsize=(10,10))
    plt.plot(second_peak_real[train_list],second_peak_pred[train_list],'x',color='navy',alpha=0.2,label='Training Set')
    plt.plot(second_peak_real[opt_list],second_peak_pred[opt_list],'x',color='firebrick',label='Validation Set')
    plt.plot(second_peak_perfect,second_peak_perfect,ls='--',color='k',label='Perfect Model')
    # plt.xlim(second_peak_perfect.min(),second_peak_perfect.max())
    # plt.ylim(second_peak_perfect.min(),second_peak_perfect.max())
    plt.legend(fontsize=14,loc='upper left')
    plt.xlabel('Real Second Peak Amplitude (Related to First Peak)',fontsize=14)
    plt.ylabel('Predicted Second Peak Amplitude (Related to First Peak)',fontsize=14)
    plt.savefig('SecondPeakPlot.png')
    

    
def plot_random_predictions(angles_defect,X,Y,Y_pred,test_list):
    Y_clean_pred = clean_pred(Y_pred)
    J = 10 
    q = 1
    plt.figure(figsize=(40,25))
    for i in range(J):
        k = np.random.choice(len(test_list))
        plt.subplot(J,2,q+1)
        plt.plot(X[test_list[k]])
        plt.ylim(-1,1)
        plt.subplot(J,2,q)
        plt.title("Defect angle %i"%(angles_defect[test_list[k]]))
        plt.plot(Y[test_list[k]],label='Real A Scan')
        plt.plot(Y_clean_pred[test_list[k]],label='Target A Scan')
        plt.legend(fontsize=14)
        k = k+1
        q=q+2
        plt.tight_layout() 
    plt.savefig('RandomExamplePlot.png')
    

def plot_best_predictions(angles_defect,X,Y,Y_pred,test_list):
    Y_clean_pred = clean_pred(Y_pred)
    q = 1
    mse_list = error_statistics(Y,Y_pred)['MSE list']
    best_list = []
    angles = list(set(angles_defect))
    angle_test_list = angles_defect[test_list]
    for angle in angles:
        angle_data = np.where(angle_test_list==angle)[0]
        mse_angle = mse_list[angle_data]
        best_list.append(angle_data[mse_angle.argmin()])
    plt.figure(figsize=(40,25))
    J = len(best_list) 
    for i in range(J):
        k = best_list[i]
        plt.subplot(J,2,q+1)
        plt.plot(X[test_list[k]])
        plt.ylim(-1,1)
        plt.subplot(J,2,q)
        plt.title("Defect angle %i"%(angle_test_list[k]))
        plt.plot(Y[test_list[k]],label='Real A Scan')
        plt.plot(Y_clean_pred[test_list[k]],label='Target A Scan')
        plt.legend(fontsize=14)
        q=q+2
        plt.tight_layout() 
    plt.savefig('BestExamplePlot.png')