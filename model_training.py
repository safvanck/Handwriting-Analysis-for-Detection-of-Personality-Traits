from __future__ import print_function
# coding: utf-8

######################################################################################
# # TRAIN:
import os
import cv2
# simplified interface for building models
import keras
import pickle
import numpy as np
import variables as vars
import matplotlib.pyplot as plt
# because our models are simple
from keras.models import Sequential
from keras.models import model_from_json
# for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from scipy.misc import imread, imresize, imshow
# dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
# into respective layers
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split


img_rows, img_cols = vars.img_rows, vars.img_cols
batch_size = vars.batch_size
num_classes = vars.num_classes
epochs = vars.epochs
model_json_path = vars.model_json_path
model_path = vars.model_path
prediction_file_dir_path = vars.prediction_file_dir_path

path = 'FEATURE-BASED-IMAGES/'

data = []
labels = []


for folder, subfolders, files in os.walk(path):
  for name in files:
    if name.endswith('.jpg'):
      x = cv2.imread(folder + '/' + name, cv2.IMREAD_GRAYSCALE)
      x = imresize(x, (img_rows, img_cols))
      __, x = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)

      # dilate
      morph_size = (2, 2)
      cpy = x.copy()
      struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
      cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
      x = ~cpy

      x = np.expand_dims(x, axis=4)

      data.append(x)

      # cv2.imwrite(str(name) + '00986.jpg', x)
      labels.append(os.path.basename(folder))

data1 = np.asarray(data)
labels1 = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data1, labels1,
                                                    random_state=0,
                                                    test_size=0.5
                                                    )
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

input_shape = (img_rows, img_cols, 1)
# build our model
model = Sequential()
# convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# again
model.add(Conv2D(64, (3, 3), activation='relu'))
# choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
# one more dropout for convergence' sake :)
model.add(Dropout(0.5))
# output a softmax to squash the matrix into output probabilities
model.add(Dense(num_classes, activation='softmax'))
# Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
# categorical ce since we have multiple classes (10)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# train that ish!

lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save the model
# serialize model to JSON
model_json = model.to_json()
with open(model_json_path, "w") as json_file:
  json_file.write(model_json)

# pickle label encoder obj
with open(vars.label_obj_path, 'wb') as lb_obj:
  pickle.dump(lb, lb_obj)

# serialize weights to HDF5
model.save_weights(model_path)
print("Saved model to disk")
