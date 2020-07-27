from __future__ import print_function

import collections

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

import numpy as np
import tensorflow as tf 
import random
import time

import codecs

import datetime #evan
import random #evan&robert

from mysql.connector import Error #evan
from mysql.connector import errorcode#evan

## --------------------------------------- Database Callback evan & robert
"""
class Database(tf.keras.callbacks.Callback):
  def __init__(self, model, x, y, rdict):
    self.model = model
    self.x = x
    self.y = y
    self.rdict = rdict

  def on_epoch_end(self, epoch, logs=None): #loss and accuracy over epoch
    iterations = epoch*1000
    submission_date = datetime.date.today()
    loss = logs['loss']
    accuracy = logs['sparse_categorical_accuracy']
    
    xindex = r.choice(range(0,len(x))
    x1 = self.x[xindex]
    prediction = self.model.predict(x1)  
    y1 = self.y[xindex]
    
    max = 0
    for i in range(0,len(prediction)):
        if prediction[i] > prediction[max]:
            max = i
    
    symbols_in = [x1[i] for i in range(0, n_input)]
    symbols_out = rdict[y1]
    symbols_out_pred = rdict[max] 
    
    """

## --------------------------------------- Preprocessing
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
#writer = tf.summary.FileWriter(logs_path)#this line was change to the line below by YKU on 05/18/20
#writer = tf.train.SummaryWriter(logs_path)
# Text file containing words for training
#training_file = '..\outputFiles\WordCountBytecodeHex.txt' the line was changed to the following line by YKU on 05/18/20
#training_file = 'outputFiles/WordCountBytecodeHex.txt'
training_file = 'inputFiles/The_D_of_I.txt'
#training_file = 'inputFiles/QuickBrownFox.txt'
#training_file = 'inputFiles/TwoSentences.txt'
#training_file = 'inputFiles/russian_text.txt'
#training_file = 'inputFiles/ES_Sample.txt'
#training_file = 'inputFiles/rus_sample.txt'
#training_file = 'inputFiles/hebrew_sample.txt'
#training_file = 'inputFiles/korean_sample.txt'
#training_file = 'inputFiles/arabic_sample.txt'
#training_file = 'inputFiles/chinese_sample.txt'
#training_file = 'inputFiles/german_sample.txt'
#training_file = 'inputFiles/greek_sample.txt'
#training_file = 'inputFiles/thai_sample.txt'
#training_file = 'inputFiles/telugu_sample.txt'
#training_file = 'inputFiles/yoruba_sample.txt'
#training_file = 'inputFiles/Raven_eng.txt'
#training_file = 'inputFiles/Raven_es.txt'
#training_file = 'inputFiles/Raven_ru.txt'
#training_file = 'inputFiles/JavaByteCodeClean2.txt'
def read_data(fname):
    with open(fname, encoding="ISO-8859-1") as f: # use this for languages with a latin alphabet EVAN
    #with open(fname, encoding="utf16", errors='ignore') as f: # use this for russian, chinese, greek, thai, telugu & arabic EVAN
    #with open(fname, encoding="cp1361", errors='ignore') as f: #use this for korean EVAN DOESN'T WORK
    #with open(fname, encoding="iso8859_6", errors='ignore') as f: #use this for hebrew EVAN DOESN'T WORK
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
# public static --> void main...
# basic words to be used for prediction, coming soon......
n_input = 3

# number of units in RNN cell
n_hidden = 512

## ---------------------------------------- 2020-06-03 MY OWN PREPROCESSING
def encode(data, dictt):
    return [dictt[data[x]] for x in range(0,len(data))]

def group(data, num):
    result = []
    for i in range(0,len(data)-num+1):
        result.append([data[i+x] for x in range(0,num)])
    return result

def expand(groups, size):
    result = []
    while len(result) < size:
        for x in range(0,len(groups)):
            result.append(groups[x])
            if len(result) >= size:
                break
    return result

def seperate(groups, index):
    x = []
    y = []
    for a in range(0,len(groups)):
        x.append(groups[a][:index])
        y.append(groups[a][index:])
    return x, y

def batches(data, dictt, input_size, batch_size):
    x, y = seperate(expand(group(encode(data,dictt),input_size+1),batch_size),input_size)
    return np.array(x), np.array(y)

## --------------------------------------- 2020-06-03 KERAS RNN (TF2)
model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(vocab_size, 64, input_length=n_input))

# Add a LSTM layer with 128 internal units.
# model.add(layers.LSTM(128))
rnn_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(n_hidden),tf.keras.layers.LSTMCell(n_hidden)])
model.add(layers.RNN(rnn_cell))

# Add a Dense layer with 10 units.
model.add(layers.Dense(len(dictionary)))

model.summary()

## --------------------------------------- 2020-06-03 
epochs = 50
batch_size_param = 1
batch_size = 1000*batch_size_param

dictt, reverse_dictt = build_dataset(training_data)
x_train, y_train = batches(training_data, dictt, n_input, batch_size)
(x_val, y_val) = batches(training_data, dictt, n_input, batch_size*0.25)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # List of metrics to monitor
              metrics=['sparse_categorical_accuracy'])

print('# Fit model on training data')
history = model.fit(x_train, y_train,
                    batch_size=batch_size_param,
                    epochs=epochs,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(x_val, y_val),
                    callbacks=[])

## --------------------------------------- 2020-06-03 

predictions = model.predict(x_val)

for x in range(0,len(predictions)):
    prediction = predictions[x]
    max = 0
    for i in range(0,len(prediction)):
        if prediction[i] > prediction[max]:
            max = i

    print("input: ",[reverse_dictt[x_val[x][i]] for i in range(0,len(x_val[x]))], "pred: ", reverse_dictt[max], " --- actual: ", [reverse_dictt[y_val[x][i]] for i in range(0,len(y_val[x]))])
    
## ----------------------------------------- 2020-06-04 PRINT OUT DURATION 
## ----------------------------------------- Moved before graph for time reasons EVAN
end_time = time.time()
print("duration: ",elapsed(end_time - start_time))

#print('\nhistory dict:', history.history)
## ----------------------------------------- 2020-06-04 MATPLOT LIB
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
