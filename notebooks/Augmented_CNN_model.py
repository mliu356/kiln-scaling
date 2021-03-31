#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import sys
import keras.preprocessing.image 
from keras.models import Sequential,Model,load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Activation, Dropout, Flatten, Dense,  GlobalMaxPooling2D
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.layers import Input
import csv
import pandas as pd   
import os
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# In[2]:


sys.argv[1] = "mhelabd"


# In[39]:


#Global Variables
WANTED_BANDS = [3, 2, 1]
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, NUM_OG_BANDS = (64, 64, len(WANTED_BANDS), 13) 
MODEL_NAME = "Augmented-drop-neighbors-CNN-({})-input-({}, {}, {})-bands-({})".format("Resnet50", IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, str(WANTED_BANDS))
PATH = "/atlas/u/{}/data/kiln-scaling/models/{}/".format(sys.argv[1], MODEL_NAME) 
# DATA_PATH = "/atlas/u/mliu356/data/kiln-scaling/tiles_drop_neighbors/"
DATA_PATH = "/atlas/u/mliu356/data/kiln-scaling/final_tiles_indices/"
MODEL_WEIGHTS_PATH = PATH + "weights/"
MODEL_HISTORY_PATH = PATH + "history/"
MODEL_EVALUATION_PATH = PATH + "evaluation/"
MODEL_EVALUATION_IMAGE_PATH = MODEL_EVALUATION_PATH + "images/"
VERBOSE = True
BALANCE_DATASET = True
USE_BALANCED_DATASET = True
PREPROCESS = False
# BALANCED_DATA_PATH = "/atlas/u/{}/data/kiln-scaling/balanced_tiles/".format(sys.argv[1])
BALANCED_DATA_PATH = "/atlas/u/mhelabd/data/kiln-scaling/balanced_tiles/"
RANDOM_STATE = 42
NUM_RANDOM_IMAGES = 5


# In[4]:


def mkdirs(names):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)
mkdirs([MODEL_WEIGHTS_PATH, MODEL_HISTORY_PATH, MODEL_EVALUATION_PATH, MODEL_EVALUATION_IMAGE_PATH])


# In[37]:


def save_h5_file(X, y, bounds):
    filename = BALANCED_DATA_PATH + ("preprocessed_" if PREPROCESS else "") + "final_tiles_drop_neighbors.hdf5"
    print("Saving file", filename)
    f = h5py.File(filename, 'w')
    bounds_dset = f.create_dataset("bounds", data=bounds)
    examples_dset = f.create_dataset("images", data=X)
    labels_dset = f.create_dataset("labels", data=y)
    f.close()

def balance_and_save_h5_data(preprocess=False, verbose=VERBOSE):
    X, y = [], []
    print("processing files...")
    for i, filename in enumerate(os.listdir(DATA_PATH)):
        print(".", end="")
        with h5py.File(DATA_PATH + filename, "r") as f:
            if i == 0:
                X = np.array(f["images"][()])                    .reshape((-1, NUM_OG_BANDS, IMAGE_HEIGHT, IMAGE_WIDTH))
                y = np.array(f["labels"][()])
                bounds = np.array(f["bounds"][()])
            else:
                x_i = np.array(f["images"][()])                    .reshape((-1, NUM_OG_BANDS, IMAGE_HEIGHT, IMAGE_WIDTH))
                X = np.concatenate((X, x_i))

                y_i = np.array(f["labels"][()])
                y = np.concatenate((y, y_i))

                bounds_i = np.array(f["bounds"][()])
                bounds = np.concatenate((bounds, bounds_i))
            try:
                n= y[y==1].shape[0]
                mask = np.hstack([np.random.choice(np.where(y == l)[0], n, replace=False) for l in np.unique(y)])
                X = X[mask]
                y = y[mask]
                bounds = bounds[mask]
            except:
                print("file has more positive examples")
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    print("bounds.shape: ", bounds.shape)
    save_h5_file(X, y, bounds)
    return X, y, bounds


# In[40]:


X, y, bounds = balance_and_save_h5_data()


# In[ ]:





# In[41]:


def load_data_from_h5(preprocess=PREPROCESS, verbose=VERBOSE,                       balance_dataset=BALANCE_DATASET, use_balanced_dataset=USE_BALANCED_DATASET):
    X, y = [], []
    data_path = BALANCED_DATA_PATH if use_balanced_dataset else DATA_PATH
    if use_balanced_dataset:
        filename = "final_tiles_drop_neighbors.hdf5"
        with h5py.File(data_path + filename, "r") as f:
            print("extracting: ", filename) 
            X = np.array(f["images"][()])                .reshape((-1, NUM_OG_BANDS, IMAGE_HEIGHT, IMAGE_WIDTH))
            X = np.moveaxis(X, 1, -1)[:, :, :, WANTED_BANDS]
            y = np.array(f["labels"][()])
            bounds = np.array(f["bounds"][()])
    else:
        for i, filename in enumerate(os.listdir(data_path)):
            print("extracting: ",filename) 
            with h5py.File(data_path + filename, "r") as f:
                if i == 0:
                    X = np.array(f["images"][()])                        .reshape((-1, NUM_OG_BANDS, IMAGE_HEIGHT, IMAGE_WIDTH))
                    X = np.moveaxis(X, 1, -1)[:, :, :, WANTED_BANDS]
                    y = np.array(f["labels"][()])
                    bounds = np.array(f["bounds"][()])
                else:
                    x_i = np.array(f["images"][()])                        .reshape((-1, NUM_OG_BANDS, IMAGE_HEIGHT, IMAGE_WIDTH))
                    x_i = np.moveaxis(x_i, 1, -1)[:, :, :, WANTED_BANDS]
                    X = np.concatenate((X, x_i))
                    y_i = np.array(f["labels"][()])
                    y = np.concatenate((y, y_i))

                    bounds_i = np.array(f["bounds"][()])
                    bounds = np.concatenate((bounds, bounds_i))
            if balance_dataset:
                #since y = 1 is always less
                n= y[y==1].shape[0]
                mask = np.hstack([np.random.choice(np.where(y == l)[0], n, replace=False)
                                  for l in np.unique(y)])
                X = X[mask]
                y = y[mask]
                bounds = bounds[mask]
    if preprocess:
        X = preprocess_input(X)  
    if verbose:
        print("x shape: ", X.shape)
        print("y shape: ", y.shape)
    return X, y, bounds


# In[42]:


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
    


# In[43]:


def split_data(X, y, train_percent=0.7, val_percent=0.1, test_percent=0.2):
    assert train_percent + val_percent + test_percent == 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=RANDOM_STATE)
    updated_val_precent = val_percent/(1 - test_percent)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=updated_val_precent, random_state=RANDOM_STATE)
    return (X_train, X_val, X_test, y_train, y_val, y_test)


# In[44]:


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


# In[45]:


def train_model(model, 
                X_train, 
                y_train,
                X_val,
                y_val,
                trial_name, 
                epochs=20,
                batch_size=64,
                multiprocessing=False, 
                early_stopping=True, 
                save_history=True, 
                datagen=None):
    
    callbacks = []
    
    if early_stopping:
        early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, mode='min', verbose=1)
        callbacks.append(early_stop)
    checkpoint_file = MODEL_WEIGHTS_PATH + "{}.h5".format(trial_name)
    checkpoint = ModelCheckpoint(checkpoint_file, 
                                  save_weights_only=True,
                                  monitor='val_accuracy',
                                  mode='max',
                                  save_best_only=False)
    callbacks.append(checkpoint)                        
    if datagen != None:
        history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_val, y_val), 
                        steps_per_epoch=len(X_train) // batch_size,
                        use_multiprocessing=multiprocessing, 
                        epochs=epochs, 
                        callbacks=callbacks)
    else: 
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


# In[46]:


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
    plt.savefig(MODEL_EVALUATION_PATH + "accuracy.png")
    plt.clf()
    # summarize history for loss
    plt.plot(list(history['loss']))
    plt.plot(list(history['val_loss']))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(MODEL_EVALUATION_PATH + "loss.png")
    plt.clf()


# In[47]:


def print_images(x, name):
    normalized_x = x/np.max(x)
    plt.imshow(normalized_x)
    plt.show()
    plt.savefig(MODEL_EVALUATION_IMAGE_PATH + name)
    plt.clf()
def evaluate_dataset(model, X, y, threshold=0.5, pictures=True, dataset_type="test"):
    # Evaluate the model on the test data using `evaluate`
    results = model.evaluate(X, y, batch_size=128)
    print(dict(zip(model.metrics_names, results)))

    y_pred = model.predict(X) > threshold
    print(tf.math.confusion_matrix(y.reshape(-1), y_pred.reshape(-1)))
    
    if pictures:
        print("Generating {} false positives and {} false negatives...".format(NUM_RANDOM_IMAGES, NUM_RANDOM_IMAGES))

        false_positives = np.logical_and(y != y_pred, y_pred == 1)
        X_fp = X[false_positives.reshape(-1), :, :]
        #gets you the location of every false positive in X
        false_positives_index = np.argwhere(false_positives)

        false_negatives = np.logical_and(y != y_pred, y_pred == 0)
        X_fn = X[false_negatives.reshape(-1), :, :]
        false_negatives_index = np.argwhere(false_negatives)

        
    for i in range(NUM_RANDOM_IMAGES):
        if len(X_fp) > 0:
            randi = np.random.randint(0, len(X_fp))
            X_fp_i = false_positives_index[randi][0]
            confidence = model.predict(np.expand_dims(X[X_fp_i], axis=0))[0][0]
            print_images(X_fp[randi], "{}_fp_example{}_confidence_{}_coordinates({}).png"                         .format(dataset_type, X_fp_i, confidence, str(bounds[X_fp_i])))
        if len(X_fn) > 0:
            randi = np.random.randint(0, len(X_fn))
            X_fn_i = false_negatives_index[randi][0]
            confidence = model.predict(np.expand_dims(X[X_fn_i], axis=0))[0][0]
            print_images(X_fn[randi],  "{}_fn_example{}_confidence_{}_coordinates({}).png"                         .format(dataset_type, X_fn_i, confidence, str(bounds[X_fn_i])))

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, threshold=0.5, pictures=True):
    print("Evaluating Training data: ")
    evaluate_dataset(model, X_train, y_train, threshold=threshold, pictures=pictures, dataset_type="train")

    print("Evaluating Validation data: ")
    evaluate_dataset(model, X_val, y_val, threshold=threshold, pictures=pictures, dataset_type="val")

    print("Evaluating Test data: ")
    evaluate_dataset(model, X_test, y_test, threshold=threshold, pictures=pictures, dataset_type="test")


# In[48]:


def Augment_data():
    datagen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True
    )
    return datagen


# In[49]:


X, y, bounds = load_data_from_h5()


# In[ ]:





# In[52]:


X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_percent=0.7, val_percent=0.1, test_percent=0.2)


# In[30]:


datagen = Augment_data()


# In[53]:


model = make_model()


# In[54]:


# model, history = train_model(model, X_train, y_train, X_val, y_val, "trial_4_epoch_35", epochs=35, save_history=True)
model, history = train_model(model, 
                             X_train, 
                             y_train, 
                             X_val, 
                             y_val, 
                             "trial_6_epoch_55", 
                             epochs=55, 
                             batch_size=64,
                             save_history=True,
                             early_stopping=False, 
                             multiprocessing=False,
                             datagen=Augment_data())


# In[23]:


graph_model_performance(history.history)


# In[24]:


evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)


# In[24]:


MODEL_EVALUATION_IMAGE_PATH


# In[ ]:




