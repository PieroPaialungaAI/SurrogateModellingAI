import tensorflow as tf
import keras 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from keras.layers import Dense, Conv1D,MaxPooling1D,Flatten,AvgPool1D,Dropout,Input,concatenate,LSTM,GRU,SimpleRNN
from keras.callbacks import  EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.models import Sequential
from model_parameters import * 

class FullModel(tf.keras.Model):

    def __init__(self):
    super().__init__()
    self.conv1 = Conv1D(7, kernels = KERNEL_SIZE)
    self.maxpool = MaxPooling1D(pool_size = POOL_SIZE)
    self.conv2 = Conv1D(14, kernels = KERNEL_SIZE)
    self.conv3 = Conv1D(34, kernels = KERNEL_SIZE)
    self.GRU = GRU(246,activation=ACTIVATION_FUNC_1)
    self.dense1 = Dense(100,activation = ACTIVATION_FUNC_2)
    self.dense2 = Dense(600,activation = ACTIVATION_FUNC_3)
    self.dense3 = Dense(OUTPUT_FULL_MODEL_SHAPE)

    def call(self, inputA, inputB):
    opt = tf.keras.optimizers.Adam()
    inputB = Input(shape=(len(INPUT_SHAPE),10))
    x = self.conv1(inputA)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = self.maxpool(x)
    x = Flatten()(x)
    x = Model(inputs=inputA, outputs = x)
    y = self.GRU(inputB)
    y = self.dense1(y)
    y = Flatten()(y)
    y = Model(inputs=inputB, outputs = y)
    combined = concatenate([x.output, y.output])
    z = self.dense2(600,activation = ACTIVATION_FUNC_3)(combined)

    return self.dense3(z)

class MainPeakModel(tf.keras.Model):

    def __init__(self):
    super().__init__()
    self.LSTM = LSTM(64)
    self.dense1 = Dense(46,activation = ACTIVATION_FUNC_1)
    selfe.dense2 = Dense(OUTPUT_MAIN_PEAK_MODEL_SHAPE)

    def call(self, inputA, inputB):
    opt = tf.keras.optimizers.Adam()
    inputA = Input(shape=(len(INPUT_SHAPE),10))
    x =self.LSTM(inputA)
    x = self.dense1(x)
    return self.dense2(x)





