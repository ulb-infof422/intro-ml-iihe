#!/usr/bin/env python
"""
   Run script to perform simple training
   on MNIST digit samples.

.. codeauthor: Deep Learning with Keras, Zig Hampel
"""

__version__ = "$Id"

try:
    #from __future__ import print_function
    import numpy as np
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils
except ImportError as e:
    print e
    raise ImportError


np.random.seed(1671)

# Network & training
N_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
N_CLASSES = 10 # No. outputs = No. digits
OPTIMIZER = Adam() # SGD optimizer
#OPTIMIZER = RMSprop() # SGD optimizer
#OPTIMIZER = SGD() # SGD optimizer
N_HIDDEN = 128 # No. hidden nodes in layer
DROPOUT = 0.3 # Dropout probability
VALIDATION_SPLIT = 0.2 # fraction of training set used for validation
LOSS = 'categorical_crossentropy' #categorical_crossentropy, binary_crossentropy, mse
METRIC = 'accuracy' #accuracy, precision, recall

# Data -> shuffled and split between training and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train is 60,000 rows of 28x28 values -> to be reshaped to 60,000 x 784
RESHAPED = 784
X_train = X_train.reshape(60000,RESHAPED)
X_test = X_test.reshape(10000,RESHAPED)

# Need to make float32 for GPU use
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize grey-scale values
X_train /= 255
X_test /= 255

print(X_train.shape[0], ' training samples')
print(X_test.shape[0], ' testing samples')

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, N_CLASSES)
Y_test = np_utils.to_categorical(y_test, N_CLASSES)

# N_CLASSES outputs, final stage is normalized via softmax
model = Sequential()
#1st layer
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
#2nd layer
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
#Output layer
model.add(Dense(N_CLASSES))
model.add(Activation('softmax'))
model.summary()

# Compile the model
model.compile(loss=LOSS,optimizer=OPTIMIZER, metrics=[METRIC])


# Train the model
history = model.fit(X_train, Y_train, \
                    batch_size=BATCH_SIZE,\
                    epochs=N_EPOCH,\
                    verbose=VERBOSE,\
                    validation_split=VALIDATION_SPLIT)


# Validation of the model with test set
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

