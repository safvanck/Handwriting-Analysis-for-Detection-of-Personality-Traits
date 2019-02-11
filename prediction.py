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

###############################################################################
# PREDICTION
#############################################################################@##


def print_results(class_lbl, out):
  print('\n', '~' * 60)
  for k, lbl in enumerate(class_lbl):
    if lbl == 'LEFT_MARG':
      print('\n > Courageous :', '\t' * 5, out[k] * 100, '%')
      print('\n > Insecure and devotes oneself completely :\t',
            100 - (out[k] * 100), '%')
    elif lbl == 'RIGHT_MARG':
      print('\n > Avoids future and a reserved person :\t', out[k] * 100, '%')
    elif lbl == 'SLANT_ASC':
      print('\n > Optimistic :', '\t' * 5, out[k] * 100, '%')
    elif lbl == 'SLANT_DESC':
      print('\n > Pessimistic :', '\t' * 4, out[k] * 100, '%')
  print('~' * 60, '\n')


def predict_personalities(filename):

  try:
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    from keras.models import model_from_json
    loaded_model = model_from_json(
        open(
            model_json_path).read())
    # load woeights into new model
    loaded_model.load_weights(model_path)
    print("*****Loaded Model from disk******")
  except Exception:
    return '\n\n> Need to train the model first!\n'
  x = cv2.imread(prediction_file_dir_path + filename, cv2.IMREAD_GRAYSCALE)
  x = imresize(x, (img_rows, img_cols))
  __, x = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)

  # dilate
  morph_size = (2, 2)
  cpy = x.copy()
  struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
  cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
  x = ~cpy
  x = np.expand_dims(x, axis=4)
  x = np.expand_dims(x, axis=0)
  out = loaded_model.predict(x, batch_size=32, verbose=0)

  with open(vars.label_obj_path, 'rb') as lb_obj:
    lb = pickle.load(lb_obj)

  result = lb.inverse_transform(np.argmax(out[0]))
  print_results(lb.classes_, out[0])

  return '\n> Prediction Completed!'


if __name__ == '__main__':
  fpath = None
  for dir_0, sub_dir_0, files in os.walk(prediction_file_dir_path):
    fpath = files
    break
  if fpath:
    res = predict_personalities(fpath[0])
    print(res)
  else:
    print('No file found for prediction!')
