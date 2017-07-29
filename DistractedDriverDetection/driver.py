# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 15:37:25 2017

@author: maohui
"""
from keras.models import Sequential,Model
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.layers import MaxPooling2D,ZeroPadding2D,Conv2D
from keras.layers import Dense,Dropout,Flatten
from keras.applications import VGG16
from keras.optimizers import SGD
import keras.backend as K
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
import os
import pickle
import time
import matplotlib.pyplot as plt
import h5py


color_type_global = 3
use_cache = 0


def get_image(path, rows=224, cols=224, color_type=3):
    if color_type == 3:
        im = cv2.imread(path)
    else:
        im = cv2.imread(path, 0)
    im = cv2.resize(im, (cols, rows))
#    im=np.transpose(im,[2,0,1])
    return im


def get_driver_data():
    file = 'driver_imgs_list.csv'
    f = open(file)
    f.readline()
    dr = dict()
    while True:
        line = f.readline()
        if line == '':
            break
        line = line.strip().split(',')
        dr[line[2]] = line[0]
    return dr


def load_train_data(rows, cols, color_type):
    files = os.listdir('train')
    X_train = []
    y_train = []
    driver_id = []
    driver_data = get_driver_data()
    start_time = time.time()
    for classification, file in enumerate(files):
        path = os.path.join('train', file)
        images = os.listdir(path)
        count = 0
        for image in images:
            im = get_image(os.path.join(path, image), rows, cols, color_type)
            X_train.append(im)
            y_train.append(classification)
            driver_id.append(driver_data[image])
            count += 1
            if count > 500:
                break
    end_time = time.time()
    print('time to load train datas:{}'.format(round(end_time - start_time, 2)))
    return X_train, y_train, driver_id



def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        f = open(path, 'wb')
        pickle.dump(data, f)
        f.close()
    else:
        print('Directory doesnot exist!')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
    return data


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    f = open(os.path.join('cache', 'model.json'), 'w')
    f.write(json_string)
    f.close()
    model.save_weights(os.path.join('cache', 'model_weight.h5'), overwrite=True)


def read_model():
    f = open(os.path.join('cache', 'model.json'), 'r')
    json_string = f.read()
    f.close()
    model = model_from_json(json_string)
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model


def split_validation_set(X, y, test_size):
#    perm = np.random.permutation(len(y))
#    X = X[perm]
#    y = y[perm]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def read_and_normalize_train_data(rows, cols, color_type=3):
    cache_path = os.path.join('cache', 'train_r_' + str(rows) + '_c_' + str(cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        X_train, y_train, driver_id = load_train_data(rows, cols, color_type)
#        cache_data((X_train, y_train, driver_id), cache_path)
    else:
        X_train, y_train, driver_id = restore_data(cache_path)

    X_train = np.array(X_train, dtype=np.uint8)
    X_train = X_train.astype(np.float32)
    
    mean_pixel = [103.939, 116.779, 123.68]
    X_train[:,:,:,0] -= mean_pixel[0]
    X_train[:,:,:,1] -= mean_pixel[1]
    X_train[:,:,:,2] -= mean_pixel[2]
    
    y_train = np.array(y_train)
    Y_train = to_categorical(y_train, num_classes=10)
    return X_train, Y_train, driver_id



    

def train():
    img_rows = 224
    img_cols = 224
    
    X,Y,driver_id = read_and_normalize_train_data(img_rows, img_cols)
    X_train,X_val,Y_train,Y_val = split_validation_set(X, Y, test_size = 0.2)
    
#    print(Y_train.shape, Y_train.argmax(axis=1))
#    print(Y_val.shape, Y_val.argmax(axis=1))
    
    base_model = VGG16(include_top = False, weights = 'imagenet', input_shape = (224,224,3))
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation = 'relu', name = 'fc1')(x)
    x = Dropout(0.5)(x)
    predic = Dense(10, activation = 'softmax', name = 'fc2')(x)
    model=Model(inputs = base_model.input, outputs = predic)
    
    sgd = SGD(lr=0.00001, momentum=0.9, decay=1e-6)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(model.metrics_names)
    for layer in base_model.layers:
        layer.trainable = False
   
    hist = model.fit(X_train,Y_train,batch_size=32,epochs=10,validation_data=(X_val,Y_val))
    
    for layer in model.layers:
        layer.trainable = True
        
    sgd = SGD(lr=0.00001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train,Y_train,batch_size=32,epochs=20,validation_data=(X_val,Y_val))    
    
    print(hist.history.keys())
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc = 'upper right')
    plt.show()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc = 'upper right')
    plt.show()
    

    

train()

    
    
    
    
    
    
    
    
    
    


