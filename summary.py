import numpy as np
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
from analyzer import *
from dataset.constants import * 
from dataset.utils import * 
from dataset.data_loader import *


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive



if __name__=='__main__':
 
    print('Accessing files on Google Drive...\n')
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("credential_access.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("credential_access.txt")
    drive = GoogleDrive(gauth)
    fileList = drive.ListFile({'q': "'1UXZPYf6xddbqV8uSZCMTLL0AKGdUNw4K' in parents and trashed=false"}).GetList()
    for file in fileList:
        file.GetContentFile(file['title'])
    results = np.load('result.npy',allow_pickle=True).item()
    Y, Y_pred = results['Y'],results['Y_pred']
    train_list, test_list = results['train_list'],results['test_list']
    X = results['X']
    angle_data = results['Angle']
    print('Plotting the results of the second peak...\n')
    second_peak_plot(Y,Y_pred,train_list,test_list)
    print('Plotting 10 random examples...\n')
    plot_random_predictions(angle_data,X,Y,Y_pred,test_list)
    print('Plotting 10 best examples...\n')
    plot_best_predictions(angle_data,X,Y,Y_pred,test_list)
    print('Plotting overview...\n')
    plot_overview(angle_data,X,Y,Y_pred,test_list)
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
    print('Plotting Amplitude vs Angle data...\n')    
    angle_plot(angle_data)
    
    