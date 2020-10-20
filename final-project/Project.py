from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import when, count, col

from pyspark import SQLContext
from pyspark.sql import Row

sqlContext = SQLContext(sc)

data = sqlContext.read.options(header = 'true', inferschema = 'true').csv('/FileStore/tables/all_stocks_5yr.csv')

data.printSchema()

data.head()

closeDF= data.select("close")
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["close"],outputCol="features")
closeAss=assembler.transform(closeDF)
closeAss.show(40)

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler

MinMaxScalerizer=MinMaxScaler().setMin(0).setMax(100).setInputCol("features").setOutputCol("MinMax_Scaled_features")
input_data = MinMaxScalerizer.fit(closeAss).transform(closeAss).select("MinMax_Scaled_features").collect()
l=len(input_data)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()   

batch_Size,window_Size,hidden_layer,learning_rate,epochs,clip_margin= 50,50,256,0.001,200,4
inputs=tf.placeholder(tf.float32,[batch_Size,window_Size,1])
targets=tf.placeholder(tf.float32,[batch_Size,1])

def create_input():
  X = []
  Y = []
  i = 0
  while (i+window_Size)<=len(input_data)-1:
    X.append(input_data[i:i+window_Size])
    Y.append(input_data[i+window_Size])
    i+=1
  assert len(X)==len(Y)
  return X,Y
X,Y = create_input()
X_train,X_test = X[:int(0.8*l)],X[int(0.8*l):]
Y_train,Y_test = Y[:int(0.8*l)],Y[int(0.8*l):]

#weights for input gate
weights_input_gate = tf.Variable(tf.truncated_normal([1,hidden_layer],stddev = 0.05))
weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer,hidden_layer],stddev = 0.05))
bias_input = tf.Variable(tf.zeros([hidden_layer]))

#weights for forgot gate
weights_forgot_gate = tf.Variable(tf.truncated_normal([1,hidden_layer],stddev = 0.05))
weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer,hidden_layer],stddev = 0.05))
bias_forget = tf.Variable(tf.zeros([hidden_layer]))

#weights for output gate
weights_output_gate = tf.Variable(tf.truncated_normal([1,hidden_layer],stddev = 0.05))
weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer,hidden_layer],stddev = 0.05))
bias_output = tf.Variable(tf.zeros([hidden_layer]))

#weights for memory gate
weights_memory_gate = tf.Variable(tf.truncated_normal([1,hidden_layer],stddev = 0.05))
weights_memory_hidden = tf.Variable(tf.truncated_normal([hidden_layer,hidden_layer],stddev = 0.05))
bias_memory = tf.Variable(tf.zeros([hidden_layer]))

#output layer weights
weights_output = tf.Variable(tf.truncated_normal([hidden_layer,1],stddev = 0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))

def lstm_cell(input,output,state):
  input_gate  = tf.sigmoid(tf.matmul(input, weights_input_gate)+ tf.matmul(output,weights_input_hidden)+bias_input)
  forget_gate = tf.sigmoid(tf.matmul(input, weights_forgot_gate)+ tf.matmul(output,weights_forget_hidden)+bias_forget)
  output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate)+ tf.matmul(output,weights_output_hidden)+bias_output)
  memory_gate = tf.sigmoid(tf.matmul(input, weights_memory_gate)+ tf.matmul(output,weights_memory_hidden)+bias_memory)
  state = state * forget_gate + input_gate* memory_gate
  output = memory_gate * tf.tanh(state)
  return state,output


import numpy as np
outputs = []
for i in range(batch_Size):
  batch_state = np.zeros([1,hidden_layer],dtype = np.float32)
  batch_output = np.zeros([1,hidden_layer],dtype = np.float32)
  for j in range(window_Size):
    batch_state,batch_output = lstm_cell(tf.reshape(inputs[i][j],(-1,1)), batch_state, batch_output)
  outputs.append(tf.matmul(batch_output, weights_output)+bias_output_layer)
outputs

losses = []
for i in range(len(outputs)):
  losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i],(-1,1)),outputs[i]))
loss = tf.reduce_mean(losses)

session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(epochs):
  trained_scores = []
  j=0
  epoch_loss= []
  while (j+batch_Size)<=len(X_train):
    X_batch= X_train[j:j+batch_Size]
    y_batch= Y_train[j:j+batch_Size]
    o,c,_ = session.run([outputs, loss,1],feed_dict={inputs:X_batch, targets:y_batch})
    epoch_loss.append(c)
    trained_scores.append(c)
    j+=batch_Size
  if i%50 ==0:
    print('Epoch {}/{} '.format(i,epochs),' Current loss:()'.format( np.mean(epoch_loss)))