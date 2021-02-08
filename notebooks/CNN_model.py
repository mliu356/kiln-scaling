#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import keras.preprocessing.image 
from keras.models import Sequential,Model,load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Activation, Dropout, Flatten, Dense,  GlobalMaxPooling2D
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.layers import Input
import csv
import pandas as pd   
import os
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# In[25]:


#Global Variables
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, NUM_OG_BANDS = (64, 64, 3, 13) 
MODEL_NAME = "CNN-({})input-({}, {}, {})".format("Resnet50", IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS)
PATH = "/atlas/u/mhelabd/data/kiln-scaling/models/{}/".format(MODEL_NAME) 
DATA_PATH = "/atlas/u/mhelabd/data/kiln-scaling/tiles/"
MODEL_WEIGHTS_PATH = PATH + "weights/"
MODEL_HISTORY_PATH = PATH + "history/"
VERBOSE = True


# In[ ]:


def load_data_from_h5(preprocess=True, verbose=VERBOSE):
    X, y = [], []
    for i, filename in enumerate(os.listdir(DATA_PATH)):
        print("extracting: ",filename) 
        with h5py.File(DATA_PATH + filename, "r") as f:
            if i == 0:
                X = np.array(f["images"]).reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_OG_BANDS))[:, :, :, 0:NUM_BANDS]
                y = np.array(f["labels"])
            else:
                x_i = np.array(f["images"]).reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_OG_BANDS))[:, :, :, 0:NUM_BANDS]
                X = np.concatenate((X, x_i))
                y_i = np.array(f["labels"])
                y = np.concatenate((y, y_i))
        print(X.shape)
        print(y.shape)
    if preprocess:
        X = preprocess_input(X)

    if verbose:
        print("x shape: ", X.shape)
        print("y shape: ", y.shape)
    return X, y
    


# In[26]:


def load_data_from_csv(preprocess=True, verbose=VERBOSE):
    for i in range(24):
        if i == 0:
            x_pos = np.loadtxt(DATA_PATH + 'pos_x_examples'+ str(i)+'.csv', delimiter=',')                .reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS))
            x_neg = np.loadtxt(DATA_PATH + 'neg_x_examples'+ str(i)+'.csv', delimiter=',')                .reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS))
        else:
            x_pos_i = np.loadtxt(DATA_PATH + 'pos_x_examples'+ str(i)+'.csv', delimiter=',')                .reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS))
            x_pos = np.concatenate((x_pos, x_pos_i))
            x_neg_i = np.loadtxt(DATA_PATH + 'neg_x_examples'+ str(i)+'.csv', delimiter=',')                .reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS))
            x_neg = np.concatenate((x_neg, x_neg_i))
    if verbose:
        print("x_pos.shape: ", x_pos.shape)
        print("x_neg.shape: ", x_neg.shape)
    if preprocess:
        x_pos = preprocess_input(x_pos)
        x_neg = preprocess_input(x_neg)

    X = np.concatenate((x_pos, x_neg))    
    # y is a vector of ones (kilns present) and zeros (kilns absent)
    y = np.concatenate((np.ones(len(x_pos)), np.zeros(len(x_neg) ) )).reshape(-1, 1)
    if verbose:
        print("x shape: ", X.shape)
        print("y shape: ", y.shape)
    return X, y
    


# In[28]:


def split_data(X, y, train_percent=0.7, val_percent=0.1, test_percent=0.2):
    assert train_percent + val_percent + test_percent == 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=42)
    updated_val_precent = val_percent/(1 - test_percent)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=updated_val_precent, random_state=42)
    return (X_train, X_val, X_test, y_train, y_val, y_test)


# In[29]:


def make_model(weights="imagenet", 
               include_top=False, 
               load_weights=None, # path of weights
               loss=keras.losses.binary_crossentropy, 
               optimizer=keras.optimizers.Adam(), 
               metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]):
    
    image_input = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS))
    base_model = ResNet50(include_top=include_top, weights=weights, input_tensor=image_input, classes=2)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(1024,activation='relu')(x) 
    x = Dense(1024,activation='relu')(x) 
    x = Dense(512,activation='relu')(x) 
    x = Dense(1, activation= 'sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = x)
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    if load_weights:
        model.load_weights(load_weights)
    return model


# In[47]:


def train_model(model, 
                X_train, 
                y_train,
                X_val,
                y_val,
                trial_name, 
                epochs=20,
                batch_size=64,
                multiprocessing=True, 
                early_stopping=False, 
                save_history=True):
    
    callbacks = []
    
    if early_stopping:
        early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
        callbacks.append(early_stop)
    checkpoint_file = MODEL_WEIGHTS_PATH + "{}.h5".format(trial_name)
    checkpoint = ModelCheckpoint(checkpoint_file, 
                                  save_weights_only=True,
                                  monitor='val_accuracy',
                                  mode='max',
                                  save_best_only=True)
    callbacks.append(checkpoint)                        
    
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=epochs, 
                        batch_size=batch_size, 
                        use_multiprocessing=multiprocessing, 
                        validation_data=(X_val, y_val), 
                        callbacks=callbacks)
    if save_history:
        with open(MODEL_HISTORY_PATH + trial_name, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    return model, history


# In[52]:


def graph_model_performance(history):
  # IF METRICS ARE UPDATED, YOU MUST UPDATE THIS
  # summarize history for accuracy
  plt.plot(list(history['binary_accuracy']))
  plt.plot(list(history['val_binary_accuracy']))
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(list(history['loss']))
  plt.plot(list(history['val_loss']))
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()


# In[74]:


def print_images(x):
  normalized_x = x/np.max(x)
  plt.imshow(normalized_x)
  plt.show()
def evaluate_dataset(model, X, y, threshold=0.5, pictures=True):
  # Evaluate the model on the test data using `evaluate`
  results = model.evaluate(X, y, batch_size=128)
  print(dict(zip(model.metrics_names, results)))
  y_pred = model.predict(X) > threshold
  print(tf.math.confusion_matrix(y.reshape(-1), y_pred.reshape(-1)))
  if pictures:
    print("Generating three false positives...")
    false_positives = np.logical_and(y != y_pred, y_pred == 1)
    X_fp = X[false_positives.reshape(-1), :, :]
    for i in range(3):
      randi = np.random.randint(0, len(X_fp))
      print_images(X_fp[randi])
    print("Generating three false negatives...")
    false_negatives = np.logical_and(y != y_pred, y_pred == 0)
    X_fn = X[false_negatives.reshape(-1), :, :]
    for i in range(3):
      randi = np.random.randint(0, len(X_fp))
      print_images(X_fn[randi])
def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, threshold=0.5, pictures=True):
  print("Evaluating Training data: ")
  evaluate_dataset(model, X_train, y_train, threshold=threshold, pictures=pictures)
  
  print("Evaluating Validation data: ")
  evaluate_dataset(model, X_val, y_val, threshold=threshold, pictures=pictures)

  print("Evaluating Training data: ")
  evaluate_dataset(model, X_test, y_test, threshold=threshold, pictures=pictures)
    


# In[48]:

print("extracting data...")
X, y = load_data_from_h5()
print("Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_percent=0.7, val_percent=0.1, test_percent=0.2)
print("making model")
model = make_model()
print("training model")
model, history = train_model(model, X_train, y_train, X_val, y_val, "trial_1_epoch_40", epochs=40, save_history=False)


# In[56]:


graph_model_performance(history.history)


# In[75]:


evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)

