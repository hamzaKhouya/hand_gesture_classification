#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:42:11 2022

@author: pc
"""

import cv2
import numpy as np
from PIL import Image
from keras import models
import tensorflow as tf



# model = models.load_model('model_ASL.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        im = im.resize((64,64))
        img_array = np.array(im)

        
        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()