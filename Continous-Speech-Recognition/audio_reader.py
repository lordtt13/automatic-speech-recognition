#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:14:52 2020

@author: tanmay
"""

import os
import dill
import librosa

from glob import glob
from time import time
from random import shuffle


SENTENCE_ID = 'sentence_id'
SPEAKER_ID = 'speaker_id'
FILENAME = 'filename'


def find_files(directory, pattern = '**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return sorted(glob(os.path.join(directory, pattern), recursive = True))


def read_audio_from_filename(filename, sample_rate):

    audio, _ = librosa.load(filename, sr = sample_rate, mono = True)
    audio = audio.reshape(-1, 1)
    return audio


def extract_speaker_id(filename):
    return filename.split('/')[-2]


def extract_sentence_id(filename):
    return filename.split('/')[-1].split('_')[1].split('.')[0]


