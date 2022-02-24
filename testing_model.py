#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:42:11 2022

@author: Abdessalam Kabouri
"""
import keras.models
from keras.models import load_model
import cv2
import numpy as np


def load_model_from_disk(MODEL_PATH):
     
    print('Charger Votre Model')
    model = load_model(MODEL_PATH)           
    
    return model
  
weights_file = "model_ASL.h5"

model =  load_model_from_disk(weights_file)
model.summary()
map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
                  8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
                  15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                  22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing',
                  28: 'space', 29: 'other'}

""" Lancer le camera Opencv """
cap = cv2.VideoCapture(0)

last_pred = 'A'
smoothed_pred = 'A'
pred_cnt = 0

while True:
    ret, frame = cap.read()
    frame = frame[0:480, 100:580] 
    frame_resized = cv2.resize(frame,(64,64))
    frame_resized = frame_resized.astype('float32')/255.0

    prediction = model.predict(frame_resized.reshape(1,64,64,3))
    y_pred = map_characters[int(np.argmax(prediction, axis=1))]


    if y_pred == last_pred:
        pred_cnt +=1
    else:
        last_pred = y_pred
        pred_cnt = 0    
    if  pred_cnt >= 5:
        smoothed_pred = y_pred
        
    cv2.putText(frame, smoothed_pred, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                (255, 0, 0), 5)
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()