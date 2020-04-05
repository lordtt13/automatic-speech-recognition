#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:57:57 2020

@author: tanmay
"""

import sounddevice as sd
import soundfile as sf


samplerate = 16000  
duration = 1 
filename = 'yes.wav'

print("start")

mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)

print("end")

sd.wait()
sf.write(filename, mydata, samplerate)