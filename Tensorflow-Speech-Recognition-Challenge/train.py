#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:21:47 2020

@author: tanmay
"""

import os
import pickle
import librosa

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import get_model
from tensorflow.compat.v1 import ConfigProto
from sklearn.preprocessing import LabelEncoder
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

train_audio_path = "train/audio"

# Reduce training set to clean samples
labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples) == 8000) : 
            all_wave.append(samples)
            all_label.append(label)
            
le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

all_wave = np.array(all_wave).reshape(-1,8000,1)

x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), \
                                            stratify = y, test_size = 0.2, random_state = 42, shuffle = True)
    
model = get_model(labels)

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.001), \
              metrics = ['accuracy'])
    
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10, min_delta = 0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')

history = model.fit(x_tr, y_tr, epochs = 100, callbacks = [es,mc], \
                    batch_size = 32, validation_data = (x_val, y_val))
    
# Sace Model
model.save("asr.h5")

# Save Datasets for future use
pickle_out = open("wav.pickle","wb")
pickle.dump(all_wave, pickle_out)
pickle_out.close()

pickle_out = open("labels_clases.pickle","wb")
pickle.dump((y, classes), pickle_out)
pickle_out.close()

# Plot training metrics
plt.plot(history.history['loss'], label = 'train') 
plt.plot(history.history['val_loss'], label = 'test') 
plt.legend() 
plt.show()