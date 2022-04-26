# -*- coding: utf-8 -*-
"""
Testing Data

The dataset was built by capturing the static gestures 
of the American Sign Language (ASL) alphabet, from 8 people, 
except for the letters J and Z, since they are dynamic gestures. 
To capture the images, they used a Logitech Brio webcam, 
with a resolution of 1920 × 1080 pixels, in a university laboratory 
with artificial lighting. By extracting only the hand region, 
they defined an area of 400 × 400 pixels for the final image of their dataset. 

"Static Hand Gesture Recognition Based on Convolutional Neural Networks"
(https://doi.org/10.1155/2019/4167890 )



@author: Mega Pertiwi
""" 

import matplotlib.pyplot as plt
from processing import load_model
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator

loaded_model = load_model()
print("Loaded model Resnet-50")

img_gen = ImageDataGenerator(rescale=1./255)

test_gen = img_gen.flow_from_directory('./Datates_Grayscale',
                                       target_size=(224,224),
                                       color_mode='grayscale',
                                        batch_size=32, shuffle=False)

#prediksi = loaded_model.predict_generator(test_gen)
prediksi = loaded_model.predict(test_gen)

y_pred = np.argmax(prediksi, axis=1)
getChar = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q',
           'R','S','T','U','V','W','X','Y']
fig, ax = plt.subplots(figsize=(12, 12))
cm = confusion_matrix(test_gen.classes,y_pred, normalize=None)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp = disp.plot(ax=ax,cmap=plt.cm.Blues)
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(getChar)
ax.yaxis.set_ticklabels(getChar)
plt.show()

print(accuracy_score(test_gen.classes, y_pred))
print(confusion_matrix(test_gen.classes,y_pred))
print(classification_report(test_gen.classes,y_pred))
