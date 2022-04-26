# -*- coding: utf-8 -*-
"""
Processing data and model

@author: Mega Pertiwi
"""
from keras.models import model_from_json
import cv2
import numpy as np

def load_model():
    json_file = open("final_model_resnet-gssdg2.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights("weight_model_resnet-gssdg2.h5")
    return loaded_model

def preprocessing(img):
    blur = cv2.blur(img,(5,5))
    blur0 = cv2.medianBlur(blur,5)
    blur1 = cv2.GaussianBlur(blur0,(5,5),0)
    blur2 = cv2.bilateralFilter(blur1,9,75,75)
    hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
                
    low = np.array([0, 58, 15])
    high = np.array([50, 210, 255])
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img,img, mask= mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res

def recognition(res, loaded_model, getChar):
    res = cv2.resize(res, (224, 224))
    pixels = np.asarray(res)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    prediksi = loaded_model.predict(pixels.reshape(-1, 224,224, 1))
    predhur = getChar(prediksi.argmax())
    return predhur
