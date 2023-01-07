from model import model 
from dataset.constants import * 
from dataset.utils import * 
from dataset.data_loader import *
from model.model_parameters import * 
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")


def train_test_index(split_size=0.9):
    index_list = np.arange(0,len(Y))
    np.random.shuffle(index_list)
    split_size = int(len(Y)*split_size)
    train_list = index_list[0:split_size]
    test_list = index_list[split_size:]
    final_test_list = index_list[split_size:]
    test_list_copy = test_list.copy()
    train_list_copy = train_list.copy()
    final_test_list_copy = final_test_list.copy()
    return {'Train':np.array(train_list),'Test':np.array(test_list)}



if __name__=='__main__':
    data = FullDataset()
    print('Ready for ML!')
    CNN_data = data['Right Profile Data'][0:700]
    RNN_data = data['RNN Data'][0:700]
    Y = data['Full Model Target'][0:700]
    main_peak = data['Main Peak Data'][0:700]
    Energy_angle_data = data['Energy and Angle Data'][0:700]
    print('Exporting the data')
    np.save('CNN_data.npy',CNN_data)
    np.save('RNN_data.npy',RNN_data)
    np.save('Y_data.npy',Y)
    np.save('main_peak_data.npy',main_peak)
    np.save('energy_and_angle_data.npy',Energy_angle_data)
    print('Training/Test split')
    train_test = train_test_index()
    train_list = train_test['Train']
    test_list = train_test['Test']
    CNN_data_train = CNN_data[train_list]
    CNN_data_test = CNN_data[test_list]
    RNN_data_train = RNN_data[train_list]
    RNN_data_test = RNN_data[test_list]
    Energy_angle_data_train = Energy_angle_data[train_list]
    Energy_angle_data_test = Energy_angle_data[test_list]
    train_data = [CNN_data_train,RNN_data_train,Energy_angle_data_train]
    test_data = [CNN_data_test,RNN_data_test,Energy_angle_data_test]
    Y_train = Y[train_list]
    Y_test = Y[test_list]
    print('Training the full model A scan...\n')
    full_scan_model = model.FullScanModel()
    full_scan_model.fit(train_data,Y_train, batch_size=BATCH_SIZE,epochs=EPOCHS_FULLMODEL,
              validation_data=(test_data,Y_test))
    print('Saving the model...\n')
    full_scan_model.save('fullscan_saved_model.h5')
    main_peak_model = model.MainPeakModel()
    Y_train = main_peak[train_list]
    Y_test = main_peak[test_list]
    train_data = [RNN_data_train, Energy_angle_data_train]
    test_data = [RNN_data_test, Energy_angle_data_test]
    print('Training the main peak model (amplitude)...\n')
    main_peak_model.fit(train_data,Y_train,validation_data=(test_data,Y_test),epochs = EPOCHS_MAINPEAKMODEL)
    print('Saving the model...\n')
    main_peak_model.save('mainpeak_saved_model.h5')
    np.save('training_test_index.npy', train_test) 
    print('Deleting previous results...\n')
    try:
        os.remove("result.npy")
    except:
        print('No other results found...\n')
    

