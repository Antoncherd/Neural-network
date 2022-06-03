import numpy as np

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
# analog neiro_1

# training inputs
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

#plt.figure(figsize=(10,5))
#for i in range(30):
#    plt.subplot(6, 5, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.imshow(x_train[i], cmap=plt.cm.binary)

#plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(400, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(10, activation='softmax')

])

print(model.summary())

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


myAdam = keras.optimizers.Adam(learning_rate=0.01)
myOpt_SGD = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=True)
model.compile(optimizer=myOpt_SGD, loss='categorical_crossentropy', metrics=['accuracy'])

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat,
                                                                          test_size=0.2)

#model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_data = (x_val_split, y_val_split))

model.evaluate(x_test, y_test_cat)

#for i in range(30):
#    plt.subplot(6, 5, i+1)
#   plt.xticks([])
#   plt.yticks([])
#    plt.imshow(x_test[i], cmap=plt.cm.binary)

#plt.show()

n=23
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"распознанная цифра: {np.argmax(res)}")

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()