#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:21:47 2020

@author: tanmay
"""

import os
import librosa

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


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