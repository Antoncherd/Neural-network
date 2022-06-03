import numpy as np

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense


# training inputs
c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

c_val = np.array([-30, -2, 10, 30])
f_val = np.array([-22, 35.6, 50, 86])


model = keras.Sequential() # import method of matrix of neurons, sequential
model.add(Dense(units=20, input_shape=(1, ), activation='linear')) # create layer of neuron, full connection, 20 neuron, 1 input
model.add(Dense(units=10, activation='linear')) # create layer of neuron, full connection, 1 neuron
model.add(Dense(units=1, activation='linear')) # create layer of neuron, full connection, 1 neuron

myAdam = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=myAdam)

history = model.fit(c, f, batch_size=2, epochs=100, verbose=False, validation_data = (c_val, f_val)) # algorithm of learning inputs, outputs, numbers of iterations, print information
a = model.predict([100]) # compute network with input 100
print(int(a))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid(True)
plt.show()
