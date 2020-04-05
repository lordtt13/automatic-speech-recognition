#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:58:28 2020

@author: tanmay
"""

import librosa

import IPython.display as ipd

from custom_predict import predict


filepath = 'path_to_recorded_file'

#reading the voice commands

samples, sample_rate = librosa.load(filepath + '/' + 'stop.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples,rate = 8000)  

predict(samples)