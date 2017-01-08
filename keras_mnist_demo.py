import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import SVG

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils.visualize_util import model_to_dot
from keras.datasets import mnist

# Number of epoch
NP_EPOCH = 20

# Load data from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Loading MNIST data from keras.")
print(X_train.shape)
print(X_test.shape)

for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
plt.tight_layout()
plt.show()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print(Y_train[0])

# Take only 10000 out of 60000

X_train = X_train[0:10000]
X_test = X_test[0:1000]
Y_train = Y_train[0:10000]
Y_test = Y_test[0:1000]

#basic model
model = Sequential()

# input layer
model.add(Dense(input_dim=28*28, output_dim=500, activation='sigmoid'))

# hidden layer
model.add(Dense(output_dim=500, activation='sigmoid'))

# output layer
model.add(Dense(output_dim=10, activation='softmax'))

model.compile(loss='mse',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])


def train_and_show_result(model):
    t_start = time.time()
    training_history = model.fit(X_train, Y_train,
                                 batch_size=100, nb_epoch=NP_EPOCH,
                                 verbose=2)

    score = model.evaluate(X_test, Y_test)
    t_end = time.time()
    print("\n--------------------")
    print("Total Testing Loss: {} ".format(score[0]))
    print("Testing Accuracy: {} ".format(score[1]))
    print("Time= %f second\n" % (t_end - t_start))
    return training_history

def plot_training_history(training_history):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_xlabel('Epoch')
    axarr[0].set_ylabel('Training Accuracy')
    axarr[0].plot(list(range(1, NP_EPOCH+1)), training_history.history['acc'])
    axarr[1].set_ylabel('Training Error')
    axarr[1].plot(list(range(1, NP_EPOCH+1)), training_history.history['loss'])
    plt.show()

history = train_and_show_result(model)

plot_training_history(history)

# Cross Entrotpy
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

history = train_and_show_result(model)

plot_training_history(history)


# Deeper
print("\n----------------------")
print("Deeper\n")

model = Sequential()

model.add(Dense(input_dim=28*28, output_dim=500, activation='sigmoid'))

model.add(Dense(output_dim=500, activation='sigmoid'))
model.add(Dense(output_dim=500, activation='sigmoid'))
model.add(Dense(output_dim=500, activation='sigmoid'))
model.add(Dense(output_dim=500, activation='sigmoid'))
model.add(Dense(output_dim=500, activation='sigmoid'))
model.add(Dense(output_dim=500, activation='sigmoid'))
model.add(Dense(output_dim=500, activation='sigmoid'))
model.add(Dense(output_dim=500, activation='sigmoid'))

model.add(Dense(output_dim=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

history = train_and_show_result(model)

plot_training_history(history)

# ReLu
print("Relu\n")

model = Sequential()

model.add(Dense(input_dim=28*28, output_dim=500, activation='relu'))

model.add(Dense(output_dim=500, activation='relu'))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dense(output_dim=500, activation='relu'))

model.add(Dense(output_dim=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

history = train_and_show_result(model)

plot_training_history(history)

# Adam  (Adaptive Learning Rate + Momentum)
print("Adam\n")

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = train_and_show_result(model)

plot_training_history(history)

# Dropout
print("Dropout\n")

model = Sequential()

model.add(Dense(input_dim=28*28, output_dim=500, activation='relu'))

model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(output_dim=500, activation='relu'))

model.add(Dense(output_dim=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


history = train_and_show_result(model)

plot_training_history(history)

