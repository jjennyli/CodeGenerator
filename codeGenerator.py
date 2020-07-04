"""Evan, please check in the latest version, so that I can start from it"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = 'tmp'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_file = '..\outputFiles\WordCountBytecodeHex.txt'
