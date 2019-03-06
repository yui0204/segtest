#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:29:14 2018

@author: yui-sudo
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import pydot, graphviz
import pickle

import cityspace_label as l

from keras.optimizers import SGD, Adam, Adagrad
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model

import tensorflow as tf

#from unet import UNet
import FCN32, FCN8, Segnet, Deeplab, PSPNet, Unet

labels = l.input_label()

def normalize_x(image):
    image = image / 127.5 - 1
    return image


def normalize_y(image):
    image = image / 255
    return image


def denormalize_y(image):
    image = image * 255
    return image


def extract_class(Y, n=26):        
    Y = (Y == n) * 1
    Y = Y.astype(np.float32)
    
    return Y


def to_train_id(Y, old_classes=35):
    for i in range(old_classes):
        Y = np.where(Y == i, labels[i].trainId, Y)
    Y = Y + 1.0
    Y = np.where(Y == 256, 0.0, Y)
        
    return Y
        

def to_final_layer(Y, classes=20):
    Y2 = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2], classes), np.float32) 
    Y2 = (Y == 0) * 1
    for k in range(1, classes): 
        Y2 = np.append(Y2, (Y == k) * 1, axis=3)                    
    Y2 = Y2.astype(np.float32)
    
    return Y2


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def custom_loss(y_true, y_pred):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)) 
    return cross_entropy


def load(folder_path, mode="image", find="labelIds.png", load_number=99999999):
    cities = os.listdir(folder_path)
    if mode == "image":
        RGB_dim = 3
        IMREAD = 1 # COLOR
    elif mode == "gt":
        IMREAD = 0 # GRAYSCALE
        RGB_dim = 1

    print(mode, "load start")
   
    images2 = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE * 2, RGB_dim), np.float32)
    image_files2 = []

    for j, city in enumerate(cities):
        image_path = folder_path + city
        image_files = os.listdir(image_path)
        if mode == "gt":
            image_files = [i for i in image_files if i.find(find) > 0]
        image_files.sort()
        
        images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE * 2, 
                           RGB_dim), np.float32)
        
        for i, image_file in enumerate(image_files):
            image = cv2.imread(image_path + os.sep + image_file, IMREAD)
            image = cv2.resize(image, (IMAGE_SIZE * 2, IMAGE_SIZE))
            
            if mode == "image":
                images[i] = normalize_x(image)
            elif mode == "gt":
                image = image[:, :, np.newaxis]
                images[i] = image     

        images2 = np.append(images2, images, 0)
        image_files2.extend(image_files)
        if len(image_files2) > load_number:
            break
        
    images2 = images2[1:]
    print(mode, "load finished\n")
    
    return images2, image_files2


def read_model(Model):
    if Model == "unet":
        model = UNet(3, classes, 64).get_model()
    elif Model == "FCN32":
        model = FCN32.FCN32(n_classes=classes, input_height=256, 
                            input_width=512, nChannels=3)
    elif Model == "FCN8":
        model = FCN8.FCN8(n_classes=classes, input_height=256, input_width=512, 
                          nChannels=3) 
    elif Model == "Segnet":
        model = Segnet.Segnet(n_classes=classes, input_height=256, input_width=512) 

    elif Model == "Unet":
        model = Unet.Unet(n_classes=classes, input_height=256, input_width=512) 

    elif Model == "Deeplab":
        model = Deeplab.Deeplabv3(weights=None, input_tensor=None, 
                                  input_shape=(256,512,3), classes=classes, OS=16)
    elif Model == "PSPNet":
        model = PSPNet.build_pspnet(nb_classes=classes, resnet_layers=101, 
                                    input_shape=(256,512), activation='softmax')
        
    return model


def train_unet(root_dir, X_train, Y_train, Model):
    model = read_model(Model)
    if classes == 2:
        loss = "binary_crossentropy"
        #loss = dice_coef_loss
    elif classes > 2:
        loss = "categorical_crossentropy"
    print("Loss function is " + loss + "\n")
    
    for i in range(100, 0, -10):
        if os.path.exists(results_dir + "/checkpoint/" + model_name
                          + "_" + str(i) + ".hdf5"):        
            model.load_weights(results_dir + "/checkpoint/" + model_name
                          + "_" + str(i) + ".hdf5")
            print(results_dir + "checkpoint/" + model_name
                          + "_" + str(i) + ".hdf5 was loaded")
            break
        
    #model.compile(loss=custom_loss, optimizer=Adam(lr=0.0001), metrics=["accuracy"])
    model.compile(loss=loss, optimizer=Adam(lr=0.0001),metrics=["accuracy"])
    
    #plot_model(model, to_file = results_dir + model_name + '.png')
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    checkpoint = ModelCheckpoint(filepath=results_dir + "/checkpoint/" + model_name + "_{epoch}.hdf5",
                                 save_best_only=False, period=10)
    
    tensorboard = TensorBoard(log_dir=results_dir, histogram_freq=0, 
                              write_graph=True)
    
    #model.summary()
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                        epochs=NUM_EPOCH, verbose=1, 
                        validation_split=0.2,
                        callbacks=[checkpoint, early_stopping, tensorboard])
    
    with open(results_dir + "history.pickle", mode="wb") as f:
        pickle.dump(history.history, f)

    model_json = model.to_json()
    with open(results_dir + "model.json", mode="w") as f:
        f.write(model_json)
    
    model.save_weights(results_dir + model_name + '_weights.hdf5')
    model.save_weights(model_name + '_weights.hdf5')
    
    return history


def plot_history(history, model_name):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(results_dir + model_name + "_accuracy.png")
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig(results_dir + model_name + "_loss.png")
    plt.show()


def predict(X_test, Model):
    model = read_model(Model)
    model.load_weights(model_name + '_weights.hdf5')
    
    print("predicting...")
    Y_pred = model.predict(X_test, BATCH_SIZE)
    
    if classes == 2:
            Y_pred[Y_pred < 0.5] = 0
            Y_pred[Y_pred >= 0.5] = 1
    
    elif classes > 2:
        Y_pred = np.argmax(Y_pred, axis=3)
        Y_pred = Y_pred[:, :, :, np.newaxis]
        
    Y_pred = Y_pred.astype(np.float32)
    print("prediction finished\n")
    
    return Y_pred


def imshow(Y_gt, Y_color):
    for i, y in enumerate(Y_color):
        plt.imshow(Y_gt[i] / 255)#.reshape(256, 512))
        plt.savefig(results_dir + "prediction/"
                    + filenames[i][:len(filenames[i])-19] + "gt.png")
        plt.show()
        plt.imshow(y / 255)#.reshape(256, 512))
        #plt.savefig(results_dir + "prediction/"
        #            + filenames[i][:len(filenames[i])-19] + "pred.png")
        plt.show()
        
        
def save_image(Y_color):
    for i, y in enumerate(Y_color):
        y = cv2.resize(y, (2048, 1024))
        y = y[:, :, [2, 1, 0]]
        cv2.imwrite(results_dir + "prediction/"
                    + filenames[i][:len(filenames[i])-19] + "pred.png", y)
        
        
def Y_to_color(Y_pred):
    Y2 =  np.zeros((len(Y_pred), IMAGE_SIZE, IMAGE_SIZE * 2, 3))
    for i in range(34):
        Y2 = Y2 + (Y_pred == labels[i].trainId + 1) * labels[i].color

    return Y2


def calculate_iou(classes, label, pred):
    print("calculating IOU...")
    flat_pred  = np.ravel(pred)
    flat_label = np.ravel(label)
    
    cmat = confusion_matrix(flat_label, flat_pred)
    acc = (cmat[0, 0] + cmat[1,1]) / cmat.sum()
    
    I = np.diag(cmat) # TP 
    U = np.sum(cmat, axis=0) + np.sum(cmat, axis=1) - I
    IoU = I / U
    
    #for i in range(classes):
    #    iIoU[i] = cmat[i, i] * np.sum(cmat, axis=1)[i] / cmat.sum()
    
    # need to write iTP, iFN and iTP / (iTP + FP + iFN)
    
    return cmat, acc, IoU



if __name__ == '__main__':
    if os.getcwd() == '/home/yui-sudo/document/segmentation/segtest':
        root_dir = "/home/yui-sudo/document/dataset/cityspaces_dataset/"
        BATCH_SIZE = 1
        NUM_EPOCH = 10
        n_class = [20]
        load_number = 10
        plot = True
    else:
        root_dir = "/export2/sudou/cityspaces_dataset/"             # labserver
        BATCH_SIZE = 8
        NUM_EPOCH = 100
        n_class = [20]
        load_number = 3000
        plot = False
    IMAGE_SIZE = 256                                     # original:1024 * 2048

    Model = "Unet" #FCN32, FCN8, etc.
    
    for i in n_class:
        classes = i #1 or 20
        model_name = Model + "_" + str(classes) + "_class"

        print("Dataset directory is", root_dir)
        print("Model name is", model_name, "\n")
        
        today = datetime.datetime.today().strftime("%Y_%m%d")
        results_dir = "./model_results/" + today + "/" + model_name + "/"
        
        if not os.path.exists(os.path.join("./model_results", today, 
                                           model_name, "prediction")):
            os.makedirs(results_dir + "prediction/")
            os.makedirs(results_dir + "checkpoint/")
            print(results_dir, "was newly made")
          
        """
        # load train data
        X_train, filenames = load(root_dir + "leftImg8bit/train/", mode="image",
                                  load_number=load_number)
        Y_train, filenames = load(root_dir + "gtFine/train/", mode="gt", 
                                  load_number=load_number)
        Y_train = to_train_id(Y_train)
        if classes == 1:
            Y_train = extract_class(Y_train, n=14)
        elif classes == 20:
            Y_train = to_final_layer(Y_train, classes=20)
        else:
            print("something wrong!")
        
        print("X", X_train.shape)
        print("Y", Y_train.shape)
        
        # train   
        history = train_unet(root_dir, X_train, Y_train, Model)
        """
        
        # prediction
        X_test, filenames = load(root_dir + "leftImg8bit/val/", mode="image")
        Y_pred = predict(X_test, Model)
        
        Y_test, filenames = load(root_dir + "gtFine/val/", mode="gt")
        Y_test = to_train_id(Y_test)        
  
        if plot == True:
            if classes > 1:
                Y_color = Y_to_color(Y_pred)
                Y_gt = Y_to_color(Y_test)
           
            else:
                Y_color = Y_pred * (0, 0, 142)
                Y_gt = (Y_test == 14) * (0, 0, 142)
            plot_history(history, model_name)
            imshow(Y_gt[:10], Y_color[:10])
            save_image(Y_color[:10])
            
        if classes == 1:    
            Y_test = extract_class(Y_test, n=14.0)
                
        # evaluation
        cmat, acc, IoU = calculate_iou(classes=i, label=Y_test, pred=Y_pred)
        print("IoU =", IoU)

        #del cmat, acc, IoU
        #gc.collect()        
        #csv_data = np.loadtxt("D:\\170825QCT\\wav\\170825QCT.csv", delimiter=",")
        #X = pd.DataFrame()
        
#        with open(results_dir + "log.txt", mode="w") as f:
#            f.write(model_name)
        