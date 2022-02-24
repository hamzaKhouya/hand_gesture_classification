#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:26:39 2022

@author: Abdessalam Kabouri
"""

""" Importer les bibliotheques necessaires """
import numpy as np
import os
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
from tensorflow import keras

""" Le dossiers contenant le dataset """
train_dir = '../Data/train_data'
test_dir = '../Data/test_train'

""" Les differentes classes des donnees """
labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,
               'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,
               'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,'space':26,
               'del':27,'nothing':28}


def charger_donnees():
    """
    Charge les données et le prétraitement. Renvoie les données d'entraînement 
    et de test avec les étiquettes.
    """
    images = []
    labels = []
    size = (64,64)
    print("Charger les images de la classe : ",end = "")
    for folder in os.listdir(train_dir):
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(labels_dict[folder])
    
    images = np.array(images)
    images = images.astype('float32')/255.0
    
    labels = keras.utils.to_categorical(labels,29)
    
    """ Splitter les donnees en donnees de tests et donnees d'apprentissage """
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.05)
    
    print('Charger', len(X_train),'images pour training,',
          'la taille de données de Train  =',X_train.shape)
    print('Charger', len(X_test),'images pour testing',
          'la taille de données de Test =', X_test.shape)
    
    return X_train, X_test, Y_train, Y_test

""""  Appel de la fonction du chargement des donnees """
X_train, X_test, Y_train, Y_test = charger_donnees()

    
""" Creation du modele CNN """
model = Sequential()
    
model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', 
                 input_shape = (64,64,3)))
model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = [3,3]))
    
model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = [3,3]))
    
model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))
model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = [3,3]))
    
model.add(BatchNormalization())
    
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
model.add(Dense(29, activation = 'softmax'))
    
model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy,
              metrics = ["accuracy"])
    
print("Modele Créer")
""" Affichier la description et les parametres du modele """
model.summary()
    
""" Entrainer le modele avec les donnees genererees """
model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_split = 0.1)

""" Sauvegarde du modele pour l'utiliser dans le script de test """
model.save("model_ASL.h5")