import tensorflow as tf
import keras 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from keras.layers import Dense, Conv1D,MaxPooling1D,Flatten,AvgPool1D,Dropout,Input,concatenate,LSTM,GRU,SimpleRNN,BatchNormalization
from keras.callbacks import  EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.models import Sequential
from model.model_parameters import * 

def FullScanModel(feat_shape):
    func = 'sigmoid'
    opt = tf.keras.optimizers.Adam()
    inputA = Input(shape=(CNN_INPUT_SHAPE,1))
    inputB = Input(shape=(RNN_INPUT_SHAPE,10))
    inputC = Input(shape=(feat_shape))
    x1 = Conv1D(filters=7, kernel_size=KERNEL_SIZE)(inputA)
    x1 = MaxPooling1D(pool_size=POOL_SIZE)(x1)
    x1 = Conv1D(filters=14, kernel_size=KERNEL_SIZE)(x1)
    x1 = MaxPooling1D(pool_size=POOL_SIZE)(x1)
    x1= Conv1D(filters=34, kernel_size=KERNEL_SIZE)(x1)
    x1 = MaxPooling1D(pool_size=POOL_SIZE)(x1)
    x1= Flatten()(x1)
    x1 = BatchNormalization()(x1)

    x1 = Model(inputs=inputA, outputs=x1)
    y1 = GRU(246,activation='relu')(inputB)
    y1 = Dense(100,activation = func)(y1)
    y1 = BatchNormalization()(y1)
    y1 = Flatten()(y1)
    y1 = Model(inputs=inputB, outputs=y1)
    x2 = Model(inputs=inputC,outputs=inputC)
    combined = concatenate([x1.output, y1.output])
    combined = concatenate([combined,x2.output])
    func = 'tanh'
#    z = Dense(20,activation = func)(combined)
    z = Dense(120,activation = func)(combined)
    z = Dense(OUTPUT_FULL_MODEL_SHAPE)(z)
    model = Model(inputs=[x1.input,y1.input,x2.input], outputs=z)
    model.compile(loss=LOSS_MODEL, optimizer=opt)
    return model


#def MainPeakModel():
#    opt = tf.keras.optimizers.Adam()
#    model = Sequential()
#    func = 'relu'
#    model.add(LSTM(164))
#    model.add(Dense(84,activation=func))
#    model.add(Dense(1))
#    return model


def MainPeakModel(feat_shape):
    func = 'relu'
    opt = tf.keras.optimizers.Adam()
    inputA = Input(shape=(RNN_INPUT_SHAPE,10))
    inputB = Input(shape=(feat_shape))
    x1 = LSTM(164)(inputA)
    x1= Dense(84,activation = func)(x1)
    x1 = Flatten()(x1)
    x1 = Model(inputs=inputA,outputs=x1)
    x2 = Model(inputs=inputB,outputs=inputB)
    combined = concatenate([x1.output,x2.output])
    z = Dense(20,activation = func)(combined)
    z = Dense(1)(z)
    model = Model(inputs =[x1.input,x2.input],outputs=z)
    model.compile(loss=LOSS_MODEL,optimizer=opt)
    return model