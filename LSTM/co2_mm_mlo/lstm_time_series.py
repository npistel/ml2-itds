
# coding: utf-8

# In[58]:

import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM
from keras.optimizers import RMSprop
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# In[2]:

data = pd.read_csv('co2_mm_mlo.csv')


# In[9]:

data = data.iloc[:-1, 4:5]

# In[23]:

dataset = data.values[:-1]


# In[24]:

dataset


# In[25]:

dataset = dataset.astype('float32')


# In[26]:

from sklearn.preprocessing import MinMaxScaler


# In[27]:

scaler = MinMaxScaler(feature_range=(0, 1))


# In[28]:

dataset = scaler.fit_transform(dataset)


# In[31]:

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# In[37]:

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


# In[38]:

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[39]:

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# In[44]:

model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[50]:

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[51]:

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[52]:

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# Train Score: 0.75 RMSE
# Test Score: 4.11 RMSE

# In[54]:

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# In[56]:

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[59]:

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig('predict_plot.png')
plt.show()


# In[ ]:



