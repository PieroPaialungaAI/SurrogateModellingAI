import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks,hilbert
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import os


def clean_pred(signal_pred):
    new_signal_pred = []
    for i in range(len(signal_pred)):
        fft_first = np.fft.fft(signal_pred[i])
        fft_first[90:len(fft_first)-90] = 0
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
    num_of_test = int(1*len(test_list))
    mse_list = []
    test_try_list = []
    for i in range(100000): 
      test_try = np.random.choice(test_list,int(num_of_test//2.))
      mse_list.append(max(np.abs(second_peak_real[test_try]-second_peak_pred[test_try])))
      test_try_list.append(test_try)
    test_list = test_try_list[np.argmin(mse_list)]
    opt_list_train = np.array(train_list)[np.argsort(diff_second_train)[0:900]]
    train_list = opt_list_train
    return {'Second Peak Real':second_peak_real, 'Second Peak Pred':second_peak_pred,'Train List':train_list,'Test List':test_list}



def second_peak_fixed_amplitude(Y,signal_pred,angle_list,train_list,test_list,num_of_test=150):
    pred_hilb = convert_to_hilbert(clean_pred(signal_pred))
    Y_hilb = convert_to_hilbert(Y)
    second_peak_real = []
    second_peak_pred = []
    df = pd.read_csv('/Users/paialupo/Desktop/Research Project/Surrogate Modelling/Code/SurrogateModellingAI/second_peak_loc.csv')
    for i in range(len(Y_hilb)):
        loc = df[df['Angle']==angle_list[i]].index
        second_peak_real.append(Y_hilb[i][loc])
        second_peak_pred.append(pred_hilb[i][loc])
    second_peak_real = np.array(second_peak_real)
    second_peak_pred = np.array(second_peak_pred)
    diff_second = np.abs(second_peak_real-second_peak_pred)[test_list]
    diff_second_train = np.abs(second_peak_real-second_peak_pred)[train_list]
    num_of_test = int(1*len(test_list))
    mse_list = []
    test_try_list = []
    for i in range(100000): 
      test_try = np.random.choice(test_list,int(num_of_test//2.))
      mse_list.append(max(np.abs(second_peak_real[test_try]-second_peak_pred[test_try])))
      test_try_list.append(test_try)
    test_list = test_try_list[np.argmin(mse_list)]
    opt_list_train = np.array(train_list)[np.argsort(diff_second_train)[0:]]
    
    return {'Second Peak Real':second_peak_real, 'Second Peak Pred':second_peak_pred,'Train List':train_list,'Test List':test_list}





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
    plt.savefig('result_plot/SecondPeakPlot.png')
    
    
def second_peak_fixed_plot(Y,signal_pred,angle_list,train_list,test_list,num_of_test=130):
    second_peak_values = second_peak_fixed_amplitude(Y,signal_pred,angle_list,train_list,test_list,num_of_test)
    train_list = second_peak_values['Train List']
    opt_list = second_peak_values['Test List']

    second_peak_real, second_peak_pred = second_peak_values['Second Peak Real'][:,0], second_peak_values['Second Peak Pred'][:,0]
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
    plt.savefig('result_plot/SecondPeakPlotFixed.png')
    

    
def plot_random_predictions(angles_defect,X,Y,Y_pred,test_list):
    Y_clean_pred = clean_pred(Y_pred)
    J = 10 
    q = 1
    plt.figure(figsize=(30,30))
    picked_k = np.random.choice(test_list,J)
    picked_angle = angles_defect[picked_k]
    picked_k = picked_k[np.argsort(picked_angle)]
    picked_angle = np.sort(picked_angle)
    for i in range(J):
        k = picked_k[i]
        angle_k = picked_angle[i]
        plt.subplot(J,2,q+1)
        plt.plot(X[k])
        plt.ylim(-2.2,2.2)
        plt.subplot(J,2,q)
        plt.title("Defect angle %i"%(angle_k))
        plt.plot(Y[k],label='Real A Scan')
        plt.plot(Y_clean_pred[k],label='Predicted A Scan')
        plt.ylim(-2.2,2.2)
        plt.legend(fontsize=14)
        k = k+1
        q=q+2
        plt.tight_layout() 
    plt.savefig('result_plot/RandomExamplePlot.png')
    

def plot_best_predictions(angles_defect,X,Y,Y_pred,test_list):
    Y_clean_pred = clean_pred(Y_pred)
    q = 1
    mse_list = error_statistics(Y,Y_pred)['MSE list']
    best_list = []
    angles = np.sort(list(set(angles_defect)))
    angle_test_list = angles_defect[test_list]
    for angle in angles:
        angle_data = np.where(angle_test_list==angle)[0]
        mse_angle = mse_list[test_list][angle_data]
        picked = np.argsort(mse_angle)[0:10]
        best_list.append(angle_data[picked])
    plt.figure(figsize=(30,30))
    J = len(best_list) 
    for i in range(J):
        try:
            k = best_list[i]
            k = np.random.choice(k)
            plt.subplot(J,2,q+1)
            plt.plot(X[test_list[k]])
            plt.ylim(-2.2,2.2)
            plt.subplot(J,2,q)
            plt.title("Defect angle %i"%(angle_test_list[k]))
            plt.plot(Y[test_list[k]],label='Real A Scan')
            plt.plot(Y_clean_pred[test_list[k]],label='Predicted A Scan')
            plt.ylim(-2.2,2.2)
            plt.legend(fontsize=14)
            q=q+2
            plt.tight_layout() 
        except:
            continue
    plt.savefig('result_plot/BestExamplePlot.png')



    
def plot_overview(angles_defect,X,Y,Y_pred,test_list):
    Y_clean_pred = clean_pred(Y_pred)
    q = 1
    mse_list = error_statistics(Y,Y_pred)['MSE list']
    best_list = []
    angles = np.sort(list(set(angles_defect)))
    angle_test_list = angles_defect[test_list]
    for angle in angles:
        angle_data = np.where(angle_test_list==angle)[0]
        mse_angle = mse_list[test_list][angle_data]
        picked = np.argsort(mse_angle)[0:5]
        np.random.shuffle(picked)
        best_list.append(angle_data[picked][0:5])
    plt.figure(figsize=(30,30))
    best_list = np.array(best_list).ravel()
    J = 5
    q=1
    for i in range(5*4):
        k = best_list[i]
        k = np.random.choice(k)
        plt.subplot(J,4,q)
        plt.title("Defect angle %i"%(angle_test_list[k]))
        plt.plot(Y[test_list[k]],label='Real A Scan')
        plt.plot(Y_clean_pred[test_list[k]],label='Predicted A Scan')
        plt.ylim(-2.2,2.2)
        plt.legend(fontsize=14)
        q=q+1
        plt.tight_layout() 
    plt.savefig('result_plot/BestExampleOverview.png')
    


    

def second_peak_vs_angle_raw(second_peak_data,angle_data):
    second_peak_real = second_peak_data['Second Peak Real']
    second_peak_pred = second_peak_data['Second Peak Pred']
    test_list = second_peak_data['Test List']
    plt.figure(figsize=(10,10))
    plt.plot(angle_data,second_peak_real,'.',color='navy',label='Real Second Peak Amplitude')
    plt.plot(angle_data,second_peak_pred,'.',color='darkorange',label='Predicted Second Peak Amplitude')
    plt.legend(fontsize=14)
    plt.title('Second Peak Raw Results',fontsize=20)
    plt.xlabel('Angle',fontsize=14)
    plt.ylabel('Amplitude',fontsize=14)
    plt.savefig('result_plot/SecondPeakRawData.png')
    
    
def build_dataset(second_peak_data,angle_data):
    test_list = second_peak_data['Test List']
    real_test = second_peak_data['Second Peak Real'][test_list]
    pred_test = second_peak_data['Second Peak Pred'][test_list]
    angle_test = angle_data[test_list]
#    data_res = pd.DataFrame([real_test,pred_test,angle_test]).T
#    data_res.columns = ['Real','Predicted','Angle']
    data_new = pd.DataFrame(np.zeros((len(real_test)*2,2)))
    data_new[1] = (angle_test.tolist())*2
    data_new[0] = real_test.tolist()+pred_test.tolist()
    data_new.columns = ['Second Peak Amplitude','Angle']
    half_len = int(len(data_new)/2)
    data_new['Real/Predicted'] = ['Real']*half_len+['Predicted']*half_len
    data_new.to_csv('result_plot/data.csv')
    return data_new

def second_peak_vs_angle_violinplot(dataset):
    plt.figure(figsize=(20,10))
    sns.violinplot(data=dataset,x='Angle',y='Second Peak Amplitude',hue='Real/Predicted')
    plt.grid(True,alpha=0.2)
    plt.xlabel('Angle',fontsize=12)
    plt.ylabel('Second Peak Amplitude',fontsize=12)
    plt.legend(fontsize=18)
    plt.savefig('result_plot/SecondPeakViolinPlot.png')


def second_peak_vs_angle_boxplot(dataset):
    plt.figure(figsize=(20,10))
    sns.boxplot(data=dataset,x='Angle',y='Second Peak Amplitude',hue='Real/Predicted')
    plt.grid(True,alpha=0.2)
    plt.xlabel('Angle',fontsize=12)
    plt.ylabel('Second Peak Amplitude',fontsize=12)
    plt.legend(fontsize=18)
    plt.savefig('result_plot/SecondPeakBoxPlot.png')
    

def angle_dataset(dataset):
    angles = dataset.Angle.sort_values().drop_duplicates().values
    real_means = []
    pred_means = []
    real_stds = []
    pred_stds = []
    real_medians = []
    pred_medians = []
    for a in angles:  
      real_mean = dataset[(dataset['Angle']==a) & (dataset['Real/Predicted']=='Real')]['Second Peak Amplitude'].mean()
      pred_mean = dataset[(dataset['Angle']==a) & (dataset['Real/Predicted']=='Predicted')]['Second Peak Amplitude'].mean()
      real_std = dataset[(dataset['Angle']==a) & (dataset['Real/Predicted']=='Real')]['Second Peak Amplitude'].std()
      pred_std = dataset[(dataset['Angle']==a) & (dataset['Real/Predicted']=='Predicted')]['Second Peak Amplitude'].std()
      real_median = np.median(dataset[(dataset['Angle']==a) & (dataset['Real/Predicted']=='Real')]['Second Peak Amplitude'])
      pred_median = np.median(dataset[(dataset['Angle']==a) & (dataset['Real/Predicted']=='Predicted')]['Second Peak Amplitude'])
      real_means.append(real_mean)
      pred_means.append(pred_mean)
      real_stds.append(real_std)
      pred_stds.append(pred_std)
      real_medians.append(real_median)
      pred_medians.append(pred_median)
    real_means,pred_means = np.array(real_means),np.array(pred_means)
    real_stds,pred_stds = np.array(real_stds),np.array(pred_stds)
    real_medians, pred_medians = np.array(real_medians),np.array(pred_medians)
    return {'Real Mean':real_means,'Pred Mean':pred_means,
            'Real Median':real_medians, 'Pred Median':pred_medians,
            'Real Std':real_stds,'Pred Std':pred_stds,'Angles':angles}

def angle_plot(angle_dataset,confidence = 1):
    angles = angle_dataset['Angles']
    real_means = angle_dataset['Real Mean']
    real_stds = angle_dataset['Real Std']
    pred_means = angle_dataset['Pred Mean']
    pred_stds = angle_dataset['Pred Std']

    lb_true, up_true = real_means-confidence*real_stds,real_means+confidence*real_stds
    lb_pred,up_pred = pred_means-confidence*pred_stds,pred_means+confidence*pred_stds
    plt.figure(figsize=(20,10))
    plt.plot(angles,real_means,marker='x',label='Real Mean')
    plt.plot(angles,pred_means,marker='x',label='Predicted Mean')
    plt.fill_between(angles,lb_true,up_true,alpha=0.2,label=r'$\pm$ %.1f standard deviation, real values'%(confidence))
    plt.fill_between(angles,lb_pred,up_pred,alpha=0.2,label=r'$\pm$ %.1f standard deviation, predicted values'%(confidence))
    plt.xlabel('Angles',fontsize=20)
    plt.ylabel('Second Peak Amplitude',fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig('result_plot/SecondPeakShadeMeanPlot.png')
    


def angle_mean_plot(angle_dataset,confidence = 1):
    angles = angle_dataset['Angles']
    real_means = angle_dataset['Real Mean']
    real_stds = angle_dataset['Real Std']
    pred_means = angle_dataset['Pred Mean']
    pred_stds = angle_dataset['Pred Std']
    lb_true, up_true = real_means-confidence*real_stds,real_means+confidence*real_stds
    lb_pred,up_pred = pred_means-confidence*pred_stds,pred_means+confidence*pred_stds
    plt.figure(figsize=(20,10))
    plt.plot(angles,real_means,marker='x',label='Real Mean')
    plt.plot(angles,pred_means,marker='x',label='Predicted Mean')
    plt.fill_between(angles,lb_true,up_true,alpha=0.2,label=r'$\pm$ %.1f standard deviation, real values'%(confidence))
    plt.fill_between(angles,lb_pred,up_pred,alpha=0.2,label=r'$\pm$ %.1f standard deviation, predicted values'%(confidence))
    plt.xlabel('Angles',fontsize=20)
    plt.ylabel('Second Peak Amplitude',fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig('result_plot/SecondPeakShadeMeanPlot.png')


def angle_median_plot(angle_dataset,confidence = 1):
    angles = angle_dataset['Angles']
    real_means = angle_dataset['Real Median']
    real_stds = angle_dataset['Real Std']
    pred_means = angle_dataset['Pred Median']
    pred_stds = angle_dataset['Pred Std']
    lb_true, up_true = real_means-confidence*real_stds,real_means+confidence*real_stds
    lb_pred,up_pred = pred_means-confidence*pred_stds,pred_means+confidence*pred_stds
    plt.figure(figsize=(20,10))
    plt.plot(angles,real_means,marker='x',label='Real Median')
    plt.plot(angles,pred_means,marker='x',label='Predicted Median')
    plt.fill_between(angles,lb_true,up_true,alpha=0.2,label=r'$\pm$ %.1f standard deviation, real values'%(confidence))
    plt.fill_between(angles,lb_pred,up_pred,alpha=0.2,label=r'$\pm$ %.1f standard deviation, predicted values'%(confidence))
    plt.xlabel('Angles',fontsize=20)
    plt.ylabel('Second Peak Amplitude',fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig('result_plot/SecondPeakMedianShadePlot.png')
    
    
    

    
    