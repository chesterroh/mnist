#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from  mnist import MnistData

import numpy as np

pickle_file = 'notMNIST_sanit.pickle'
mnist = MnistData(pickle_file,one_hot=True)

model = Sequential([
#    Dense(32,input_shape=(784,)),     # (784,) means that batch dimension not defined
    Dense(512,input_dim=784),
    Activation('relu'),
    Dropout(0.75),
    Dense(512,activation='relu'),
    Dropout(0.5),
    Dense(512,activation='relu'),
    Dropout(0.75),
    Dense(10),
    Activation('softmax'),
])

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
x_train = np.reshape(mnist.train_data.images,[-1,784])
y_labels = mnist.train_data.labels
model.fit(x_train,y_labels, epochs=5,batch_size=100)

x_test = np.reshape(mnist.test_data.images,[-1,784])
y_test = mnist.test_data.labels
score = model.evaluate(x_test,y_test, batch_size=128)
print(score)





