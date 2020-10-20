# Databricks notebook source
sc.install_pypi_package("matplotlib==3.3.0")
sc.install_pypi_package("pandas==0.24.2")
sc.install_pypi_package("tensorflow==1.14.0")
sc.install_pypi_package("sklearn")

# COMMAND ----------

# Import statements
import urllib
import operator
import numpy as np 
import pandas as pd
import tensorflow as tf
from pyspark.sql import Row
from pyspark.sql import Window
from pyspark import SQLContext
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.functions import when, count, col

# COMMAND ----------

# Used for normalizing and scaling close prices
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# Creating pySpark Context 
sqlContext = SQLContext(sc)

# Loading data from mounted s3 bucket
data = sqlContext.read.options(header = 'true', inferschema = 'true').csv("s3://bigdatasu20/all_stocks_5yr.csv")

# Selecting American Airlines Data
aal = data.filter("Name == 'AAL'")

# COMMAND ----------

# For Visualization of close prices with data
df = aal.toPandas()

df.loc[:, 'date'] = pd.to_datetime(df.loc[:,'date'], format="%Y/%m/%d")
df = df.sort_values('date')

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),df['close'])
plt.xticks(range(0,df.shape[0],50),df['date'].loc[::50],rotation=60)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closing price',fontsize=18)
plt.legend()
plt.show()
%matplot plt

# COMMAND ----------

# Selecting only close prices for predicition and converting them in to numpy array
close_list = aal.select('close').collect()
close_price = [row['close'] for row in close_list]
close_prices = np.array(close_price)

# COMMAND ----------

# Scaling close prices 
scaler = StandardScaler()
standardized_data = scaler.fit_transform(close_prices.reshape(-1,1))

# COMMAND ----------

# Calculating the split time to split train and test data
size = int(len(close_prices))
print("Size of Dataset", size)

# 80% train and 20% test data
split_time = int(size * 80 / 100)

print("Size of Training Dataset", split_time)
print("Size of Testing Dataset", size - split_time)

# COMMAND ----------

# Preprocessing close prices so that all values are between 0 and 1
scaler = MinMaxScaler()

window_length = 50
for i in range(0, size, window_length):
    scaler.fit(standardized_data[i:i + window_length,:])
    standardized_data[i:i + window_length,:] = scaler.transform(standardized_data[i:i + window_length,:])

# COMMAND ----------

# Reshaping in to original data
scaled_data = standardized_data.reshape(-1)

# COMMAND ----------

# Smoothing using exponential moving average
exp_moving_avg = 0.0
gamma = 0.1
for i in range(split_time):
  exp_moving_avg = gamma * scaled_data[i] + (1-gamma)*exp_moving_avg
  scaled_data[i] = exp_moving_avg

# COMMAND ----------

# Splitting data in to train and test sets
trainingData = scaled_data[:split_time]
testingData = scaled_data[split_time:]

# Concatinating for visualization purposes
closingData = np.concatenate([trainingData,testingData],axis=0)

plt.figure(figsize = (18,9))
plt.plot(closingData, label='Scaled Close')
plt.xticks(range(0,df.shape[0],50),df['date'].loc[::50],rotation=60)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closing price',fontsize=18)
plt.legend()
plt.show()
%matplot plt

# COMMAND ----------

# Data Generator is used for training model
class DataGen(object):

    def __init__(self,prices, sizeOfBatch, numOfTimeSteps):
        self._Prices = prices
        self._pricesLength = len(self._Prices) - numOfTimeSteps
        self._sizeOfBatch = sizeOfBatch
        self._numOfTimeSteps = numOfTimeSteps
        self._batches = self._pricesLength // self._sizeOfBatch
        self._pointer = [_val * self._batches for _val in range(self._sizeOfBatch)]

    def upcomingBatch(self):

        batchData = np.zeros((self._sizeOfBatch),dtype=np.float32)
        batchLabels = np.zeros((self._sizeOfBatch),dtype=np.float32)

        for b in range(self._sizeOfBatch):
            if self._pointer[b]+1>=self._pricesLength:
                self._pointer[b] = np.random.randint(0,(b+1)*self._batches)

            batchData[b] = self._Prices[self._pointer[b]]
            batchLabels[b]= self._Prices[self._pointer[b]+np.random.randint(0,5)]

            self._pointer[b] = (self._pointer[b]+1)%self._pricesLength

        return batchData, batchLabels

#   This method unpack_future_batches will unpack batches of close prices data
    def unpack_future_batches(self):

        unpackData, unpackLabels = [],[]
        init_data, init_label = None,None
        for ui in range(self._numOfTimeSteps):

            data, labels = self.upcomingBatch()    

            unpackData.append(data)
            unpackLabels.append(labels)

        return unpackData, unpackLabels

    def resetIndices(self):
        for b in range(self._sizeOfBatch):
            self._pointer[b] = np.random.randint(0,min((b+1)*self._batches,self._pricesLength-1))

generatedData = DataGen(trainingData,5,5)
data_, labels_ = generatedData.unpack_future_batches()

# COMMAND ----------

# Parameters for LSTM model
D = 1
futureTimeSteps = 50
sizeOfBatch = 20
hiddenNodeCount = [256, 256, 128]
# Number of hidden layers
layerCount = len(hiddenNodeCount) 
# Dropuout Probability
dropout = 0.20

tf.compat.v1.reset_default_graph()

# COMMAND ----------

# We have to enable eager execution to enable some of the features
tf.compat.v1.disable_eager_execution()

# Input data
trainingInputs = []
trainingOutputs = []

for i in range(futureTimeSteps):
    trainingInputs.append(tf.compat.v1.placeholder(tf.float32, shape=[sizeOfBatch,D],name='trainingInputs%d'%i))
    trainingOutputs.append(tf.compat.v1.placeholder(tf.float32, shape=[sizeOfBatch,1], name = 'trainingOutputs%d'%i))

# COMMAND ----------

# Defining LSTM cell
LSTMCells = [
    tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hiddenNodeCount[i],
                            state_is_tuple=True,
                            initializer= tf.contrib.layers.xavier_initializer()
                           )
 for i in range(layerCount)]


# Using dropout LSTM cells to reduce overfitting
dropLSTMCells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout) for lstm in LSTMCells]

dropMultiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(dropLSTMCells)
multiRNNCell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(LSTMCells)

# Defining weights and bias for the model
w = tf.compat.v1.get_variable('w',shape=[hiddenNodeCount[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
b = tf.compat.v1.get_variable('b',initializer=initializer(shape=[1]))

# COMMAND ----------

# Calculating and storing both cell and hidden state
cellState, hiddenState = [],[]
initialState = []
for i in range(layerCount):
  cellState.append(tf.Variable(tf.zeros([sizeOfBatch, hiddenNodeCount[i]]), trainable=False))
  hiddenState.append(tf.Variable(tf.zeros([sizeOfBatch, hiddenNodeCount[i]]), trainable=False))
  initialState.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(cellState[i], hiddenState[i]))

inputs = tf.concat([tf.expand_dims(t,0) for t in trainingInputs],axis=0)

LSTMOutputs, state = tf.compat.v1.nn.dynamic_rnn(
    dropMultiRNNCell, inputs, initial_state=tuple(initialState),
    time_major = True, dtype=tf.float32)

# Feeding the sate to get predictions
LSTMOutputs = tf.reshape(LSTMOutputs, [sizeOfBatch*futureTimeSteps,hiddenNodeCount[-1]])

allOutputs = tf.compat.v1.nn.xw_plus_b(LSTMOutputs,w,b)

split_outputs = tf.split(allOutputs,futureTimeSteps,axis=0)

# COMMAND ----------

# Calculating loss which is Mean Squared Error
loss = 0.0
with tf.compat.v1.control_dependencies([tf.compat.v1.assign(cellState[i], state[i][0]) for i in range(layerCount)]+
                             [tf.compat.v1.assign(hiddenState[i], state[i][1]) for i in range(layerCount)]):
  for k in range(futureTimeSteps):
    loss += tf.reduce_mean(0.5*(split_outputs[k]-trainingOutputs[k])**2)

# Learning rate decay
Gstep = tf.Variable(0, trainable=False)
GStepInc = tf.compat.v1.assign(Gstep,Gstep + 1)
tfLr = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)
tfMinLr = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)

lr = tf.maximum(tf.compat.v1.train.exponential_decay(tfLr, Gstep, decay_steps=1, decay_rate=0.5, staircase=True),tfMinLr)

# Defining optimizer nad used Adam 
optimizer = tf.compat.v1.train.AdamOptimizer(lr)
grads, v = zip(*optimizer.compute_gradients(loss))
grads, _ = tf.clip_by_global_norm(grads, 5.0)
optimizer = optimizer.apply_gradients(zip(grads, v))

# COMMAND ----------

sampleInputs = tf.compat.v1.placeholder(tf.float32, shape=[1,D])

# Storing LSTM state which will be used later for predicting output

cell, hidden, sampleInitialState = [],[],[]
for i in range(layerCount):
  cell.append(tf.Variable(tf.zeros([1, hiddenNodeCount[i]]), trainable=False))
  hidden.append(tf.Variable(tf.zeros([1, hiddenNodeCount[i]]), trainable=False))
  sampleInitialState.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(cell[i],hidden[i]))

#   Resetting both state variables
sampleStateReset = tf.group(*[tf.compat.v1.assign(cell[i],tf.zeros([1, hiddenNodeCount[i]])) for i in range(layerCount)],
                            *[tf.compat.v1.assign(hidden[i],tf.zeros([1, hiddenNodeCount[i]])) for i in range(layerCount)])

sampleOutput, sampleState = tf.compat.v1.nn.dynamic_rnn(multiRNNCell, 
                                                            tf.expand_dims(sampleInputs,0),
                                                            initial_state=tuple(sampleInitialState),
                                                            time_major = True,
                                                            dtype=tf.float32)

with tf.control_dependencies([tf.compat.v1.assign(cell[i],sampleState[i][0]) for i in range(layerCount)]+
                              [tf.compat.v1.assign(hidden[i],sampleState[i][1]) for i in range(layerCount)]):  
  predictionSample = tf.compat.v1.nn.xw_plus_b(tf.reshape(sampleOutput,[1,-1]), w, b)

# COMMAND ----------

# Training and predicting close values of AAL stocks using LSTM
NoOfEpochs = 30
test_interval = 1

stepsPerPredicition = 10

trainingLength = trainingData.size

# Storing MSE losses for training and testing
training_MSELoss = []
testing_MSELoss = []
predictions = []
lossHistory = {}

session = tf.compat.v1.InteractiveSession()

tf.compat.v1.global_variables_initializer().run()

# Keeping track of changes in loss
lossCount = 0
lossThreshold = 2
avgLoss = 0

dataGen = DataGen(trainingData,sizeOfBatch,futureTimeSteps)

xAxis = []

# Test predictions from here
test_points_seq = np.arange(1007, 1258-5, 10).tolist()

for epoch in range(NoOfEpochs):       

# Training
    for s in range(trainingLength//sizeOfBatch):

        data_, labels_ = dataGen.unpack_future_batches()

        dict = {}
        for i,(data,label) in enumerate(zip(data_,labels_)):            
            dict[trainingInputs[i]] = data.reshape(-1,1)
            dict[trainingOutputs[i]] = label.reshape(-1,1)

        dict.update({tfLr: 0.0001, tfMinLr:0.000001})

        _, l = session.run([optimizer, loss], feed_dict=dict)

        avgLoss += l

# Validation
    if (epoch + 1) % test_interval == 0:

      avgLoss = avgLoss/(test_interval*(trainingLength//sizeOfBatch))

      # Calculating ahe average loss
      if (epoch + 1)%test_interval==0:
        print("[========== Epoch %d/%d ==========]" % (epoch + 1, NoOfEpochs))
        print("Average Validation Loss: %f" % (avgLoss))

      training_MSELoss.append(avgLoss)

      avgLoss = 0

      predictionsSeq = []

      MSETestLoss = []

# Calculating Predictions
      for w in test_points_seq:
        MSE_Testloss = 0.0
        ourPredictions = []

        if (epoch + 1) - test_interval==0:
           x_Axis_=[]

# Sending this to recent state of stock prices
        for j in range(w-futureTimeSteps+1,w-1):
          currentPrice = closingData[j]
          dict[sampleInputs] = np.array(currentPrice).reshape(1,1)    
          _ = session.run(predictionSample, feed_dict=dict)

        dict = {}

        currentPrice = closingData[w-1]

        dict[sampleInputs] = np.array(currentPrice).reshape(1,1)

        for p in range(stepsPerPredicition):
          pred = session.run(predictionSample,feed_dict=dict)
          ourPredictions.append(pred.item())
          dict[sampleInputs] = np.asarray(pred).reshape(-1,1)

          if (epoch + 1)-test_interval==0:
            x_Axis_.append(w+p)
            
          MSE_Testloss += 0.5*(pred-closingData[w+p])**2

        session.run(sampleStateReset)
        predictionsSeq.append(np.array(ourPredictions))

        MSE_Testloss /= stepsPerPredicition
        MSETestLoss.append(MSE_Testloss)

        if (epoch + 1)-test_interval==0:
          xAxis.append(x_Axis_)

      current_test_mse = np.mean(MSETestLoss)

# Performing Decay of Learning Rate
      if len(testing_MSELoss)>0 and current_test_mse > min(testing_MSELoss):
          lossCount += 1
      else:
          lossCount = 0

      if lossCount > lossThreshold :
            session.run(GStepInc)
            lossCount = 0
            print('Because the test error rate has not increased in the last 2 epochs we are decreasing learning rate by 0.5')

      testing_MSELoss.append(current_test_mse)
      print('MSE of the Testing Data: %f'%np.mean(MSETestLoss))
#       Storing loss history to use it in next step for plotting the loss for best epoch
      lossHistory[epoch+1] =  np.mean(MSETestLoss)
      predictions.append(predictionsSeq)

# COMMAND ----------

# Selecting epoch with least loss
epochWithLeastLoss = min(lossHistory.items(), key=operator.itemgetter(1))[0]

# COMMAND ----------

# Storing predictions from the best epoch
preds = []
for xval,yval in zip(xAxis,predictions[epochWithLeastLoss-1]):
  preds.append(yval)
  
flattened = [val for sublist in preds for val in sublist]

# COMMAND ----------

# Visuzlizing true values and predicted values of test data
plt.figure(figsize = (18,9))

plt.plot(testingData, label="True")
plt.plot(flattened, label="Predicted")
plt.title('True Values vs Predicted Close Prices',fontsize=18)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Close Price', fontsize=15)

plt.legend()
plt.show()
%matplot plt

# COMMAND ----------

# Visualizing loss for each epoch
plt.figure(figsize = (18,9))
plt.plot(lossHistory.values())
plt.title('MSE Test Loss for each Epoch',fontsize=18)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('MSE', fontsize=15)
%matplot plt

# COMMAND ----------

# Calculating MAE
mae = tf.keras.losses.MeanAbsoluteError()
loss = mae(testingData[:-2], flattened)
print("Mean Absolute Error",loss.eval())

# COMMAND ----------

# Calculating MSE
mse = min(lossHistory.items(), key=operator.itemgetter(1))[1]
print("Mean Squared Error",mse)

# COMMAND ----------

# Calculating MSLE
msle = tf.keras.losses.MSLE(testingData[:-2], flattened)
print("Mean Squared Logarithmic Error", msle.eval())

# COMMAND ----------

session.close()

# COMMAND ----------

hiddenNodeCount

# COMMAND ----------


