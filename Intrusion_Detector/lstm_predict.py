from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
 
def readdata():
    xtrain = read_csv("train.in", header=None).values
    ytrain = read_csv("train.out", header=None).values.reshape(-1,1)
    xtest = read_csv("test.in", header=None).values
    ytest = read_csv("test.out", header=None).values.reshape(-1,1)

    return xtrain[:, :-1], ytrain, xtest[:, :-1], ytest

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

xtrain, ytrain, xtest, ytest = readdata()

#dataset = read_csv('pollution.csv', header=0, index_col=0)
#values = dataset.values
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])
#values = values.astype('float32')
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
train_X = series_to_supervised(xtrain, 1, 1)
test_X = series_to_supervised(xtest, 1, 1)
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
#print(train_X.head())
 
train_X = train_X.values
test_X = test_X.values
#n_train_hours = 365 * 24
#train = values[:n_train_hours, :]
#test = values[n_train_hours:, :]
#train_X, train_y = train[:, :-1], train[:, -1]
#test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

for idt, label in enumerate(ytest):
    if label > 1:
        ytest[idt] = 1

for idt, label in enumerate(ytrain):
    if label > 1:
        ytrain[idt] = 1

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, ytrain[1:], epochs=50, batch_size=72, validation_data=(test_X, ytest[1:]), verbose=1, shuffle=False)
model.save('model1.mdl')
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
#model = load_model('model.mdl')

# make a prediction
yhat = model.predict(test_X)
scaler = MinMaxScaler(feature_range=(0, 1))
yhat = scaler.fit_transform(yhat)

fpr, tpr, _ = roc_curve(ytest[1:], yhat)
pyplot.plot(fpr, tpr, marker='.', label='Classifier')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

# FPR of 0.1% gives TPR of 89.36% with 1
# FÜR of 0.1% gives TPR of 83.34% with 3