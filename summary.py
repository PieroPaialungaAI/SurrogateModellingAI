import numpy as np
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
from analyzer import *

if __name__=='__main__':
    results = np.load('result.npy',allow_pickle=True).item()
    Y, Y_pred = results['Y'],results['Y_pred']
    train_list, test_list = results['train_list'],results['test_list']
    X = results['X']
    angle_data = results['Angle']
    print('Plotting the results of the second peak...\n')
    second_peak_plot(Y,Y_pred,train_list,test_list)
    print('Plotting 10 random examples...\n')
    plot_random_predictions(angle_data,X,Y,Y_pred,train_list)
    print('Plotting 10 best examples...\n')
    plot_best_predictions(angle_data,X,Y,Y_pred,test_list)
    print('Smoothing predictions...\n')
    Y_pred = clean_pred(Y_pred)
    print('Exporting MSE statistics...\n')
    mse_stat = error_statistics(Y,Y_pred)['MSE list']
    pd.DataFrame(mse_stat[test_list]).describe().to_csv('MSE_summary.csv')
    print('Extracting second peak results...\n')
    second_peak_data = second_peak_metrics(Y,Y_pred,train_list,test_list)
    print('Formatting the results dataset...\n')
    dataset = build_dataset(second_peak_data,angle_data)
    print('Plotting raw angles results...\n')
    second_peak_vs_angle_raw(second_peak_data,angle_data)
    print('Plotting boxplot angles results...\n')
    second_peak_vs_angle_boxplot(dataset)
    print('Plotting violinplot angles results...\n')
    second_peak_vs_angle_violinplot(dataset)
    print('Extracting angle data...\n')    
    angle_data = angle_dataset(dataset)    
    print('Plotting Amplitdue vs Angle data...\n')    
    angle_plot(angle_data)