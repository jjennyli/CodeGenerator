'''
Li: A TensorFlow RNN sample.
Program coding prediction after n_input words learned from java code.
A complete program is automatically generated if the predicted code
is fed back as input to contiue the generation step.
Note: This is part of Dr. Li's research project. Please don't distribute.
Note: Please only distribute under Dr Li's permission
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf 
#from tensorflow.contrib import rnn#YKU commented this line on 5/18/2020 
import random
import collections
import time
#import csv
import matplotlib.pyplot as plt
import mysql.connector #must install mysql connector #YKU 20200601

from mysql.connector import Error#YKU 20200601

from mysql.connector import errorcode#YKU 20200601
import datetime #YKU 20200603
acc_history = []
#inside = []

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
writer = tf.train.SummaryWriter(logs_path)
# Text file containing words for training
#training_file = '..\outputFiles\WordCountBytecodeHex.txt' the line was changed to the following line by YKU on 05/18/20
#training_file = 'outputFiles/WordCountBytecodeHex.txt'
training_file = 'inputFiles/The_D_of_I.txt'
#training_file = 'inputFiles/QuickBrownFox.txt'
#training_file = 'inputFiles/TwoSentences.txt'
#training_file = 'inputFiles/russian_text.txt'
def read_data(fname):
    with open(fname) as f:#this line was commented out by YKU on 20200603
    #with open('fname', encoding = "ISO-8859-1")as f:#this line was added by YKU on 20200603
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(training_file)
#print("Test...")#YKU
#print(training_data)#YKU
print("Loading training data...")


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
n_input =8

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN_LiTmp(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    #x = tf.split(x,n_input,1)#this code was changed to the following line by YKU on 05/18/20
    x = tf.split(1,n_input,x)
    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    # rnn_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(n_hidden),tf.keras.layers.LSTMCell(n_hidden)])
    #rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)]) # this line was changed to the following line by YKU on 05/18/20
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(512)#this line was added by YKU on 5/18/20

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    # outputs = tf.keras.layers.RNN(rnn_cell, unroll=True)
    #outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32) #this line was commented out by YKU on 05/18/20
    outputs, states = tf.nn.rnn(rnn_cell, x, dtype=tf.float32)#this line was added by YKU on 5/18/20
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN_LiTmp(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
#init = tf.global_variables_initializer()#this line was commented out by YKU on 05/18/20
init = tf.initialize_all_variables() #this line was added by YKU on 05/18/20
# Launch the graphical session
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    #writer.add_graph(session.graph) #this line was commented out by YKU on 05/19/2020
    writer.add_graph(session.graph.as_graph_def()) #this line was added by YKU on 05/19/2020
    while step < training_iters:
       
       
            
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            num_iter = step+1
            avg_acc = 100*acc_total/display_step
            avg_loss = loss_total/display_step
            acc_history.append([num_iter, avg_acc])
            #word_in = [training_data[i] for i in range(offset, offset + n_input)]
            #inside.append(word_in)
            #insider = ''.join([training_data[i] for i in range(offset, offset + n_input)], " ")
            insider = ', '.join(map(str, [training_data[i] for i in range(offset, offset + n_input)]))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
            
            #Iter= 8000, Average Loss= 6.012484, Average Accuracy= 6.20%
            #['Despotism,', 'it', 'is'] - [their] vs [warned]

            #--------------------DB part
            print ("insider") #YKU 20200603
            print (num_iter) #YKU 20200603
            print (avg_loss) #YKU 20200603
            avg_lossInString = str(avg_loss)  # float -> str
            print (avg_acc) #YKU 20200603
            avg_accInString = str(avg_acc)  # float -> str
            
            now = datetime.date.today()
            date = str(now)  # str
            print (symbols_out) #YKU 20200603
            out = str(symbols_out)
            print (symbols_out_pred) #YKU 20200603
            predict = str(symbols_out_pred)
            inp = str(symbols_in)
            bad_chars = ['[', ']', "'"]
            for i in bad_chars : 
              inp = inp.replace(i, '') 
            try:
                connection = mysql.connector.connect(host='131.125.81.8',
                                         database='jendb',
                                         user='jen',
                                         password='KeanCS15')  
                cursor = connection.cursor()                  
                mySql_insert_query = """INSERT INTO the_d_of_I (input,iter,loss,accuracy,prediction, actual, submission_date) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s) """
                recordTuple = (inp, num_iter, avg_lossInString, avg_accInString, predict, out, date)
                #recordTuple = (symbols_in, num_iter, avg_loss, avg_acc, symbols_out_pred)
                cursor.execute(mySql_insert_query, recordTuple)     
                connection.commit()
                print("Record inserted successfully into the_d_of_I table")
                cursor.close()
            except mysql.connector.Error as error:
                print("Failed to insert into MySQL table {}".format(error))
            finally:
                      if (connection.is_connected()):
                        connection.close()
                        print("MySQL connection is closed")           
            #-----------------------------------------------------DB part is over
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("You can now use the model for prediction of sentences.")
    
    #with open("/home/students/ncwit/python/test.csv", 'w', newline='') as csvfile:
    #  writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #  writer.writerow(mylist)

    xvalues = []
    yvalues = []
    for row in acc_history:
      xvalues.append(int(row[0]))
      yvalues.append(int(row[1]))
       
    plt.plot(xvalues,yvalues)
    plt.xlabel('number iteration')
    plt.ylabel('average accuracy')
    plt.title('Rnn Accuracy')
    plt.legend()
    plt.show()
    
    while True:
        prompt = "%s words: " % n_input
        #sentence = input(prompt)#error was here input was changed to raw_input
        sentence = raw_input(prompt)#this line was added 
        sentence = sentence.strip() #trimming spaces
        words = sentence.split(' ')#creating an array (delimeter by space)
    
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
            
            
