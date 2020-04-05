#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:33:08 2020

@author: tanmay
"""

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D


K.clear_session()


def get_model(labels):
    
    inputs  =  Input(shape = (8000, 1))

    conv = Conv1D(8, 13, padding = 'valid', activation = 'relu', strides = 1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    conv = Conv1D(16, 11, padding = 'valid', activation = 'relu', strides = 1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    conv = Conv1D(32, 9, padding = 'valid', activation = 'relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    conv = Conv1D(64, 7, padding = 'valid', activation = 'relu', strides = 1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    conv = Flatten()(conv)
    
    conv = Dense(256, activation = 'relu')(conv)
    conv = Dropout(0.3)(conv)
    
    conv = Dense(128, activation = 'relu')(conv)
    conv = Dropout(0.3)(conv)
    
    outputs = Dense(len(labels), activation = 'softmax')(conv)
    
    model = Model(inputs, outputs)
    
    return model