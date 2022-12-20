from model import model 
from dataset.constants import * 
from dataset.utils import * 
from dataset.data_loader import *
from model.model_parameters import * 
import tensorflow as tf


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
    data = Dataset()
    CNN_data = data['Right Profile Data']
    RNN_data = data['RNN data']
    Y = data['Full Model Target']
    main_peak = data['Main Peak Data']
    train_test = train_test_index()
    train_list = train_test['Train']
    test_list = train_test['Test']
    CNN_data_train = CNN_data[train_list]
    CNN_data_test = CNN_data[test_list]
    RNN_data_train = RNN_data[train_list]
    RNN_data_test = RNN_data[test_list]
    Y_train = Y[train_list]
    Y_test = Y[test_list]
    print('Training the full model A scan...\n')
    model = model.FullScanModel()
    model.fit([CNN_data_train,RNN_data_train],Y_train, batch_size=100,epochs=EPOCHS_FULLMODEL,
              validation_data=([CNN_data_test,RNN_data_test],Y_test))
    print('Saving the model...\n')
    model.save('fullscan_saved_model.h5')
    model = model.MainPeakModel()
    model.build(RNN_data.shape,)
    opt = tf.keras.optimizers.Adam()
    Y_train = main_peak[train_list]
    Y_test = main_peak[test_list]
    model.compile(optimizer=opt,loss = LOSS_MODEL)
    print('Training the main peak model (amplitude)...\n')
    model.fit(RNN_data_train,Y_train,validation_data=(RNN_data_test,Y_test),epochs = EPOCHS_MAINPEAKMODEL)
    print('Saving the model...\n')
    model.save('mainpeak_saved_model.h5')
    np.save('training_test_index.npy', train_test) 

