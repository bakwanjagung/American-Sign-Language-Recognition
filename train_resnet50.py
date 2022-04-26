# -*- coding: utf-8 -*-
"""
Train dataset using ResNet50

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

from keras.preprocessing.image import ImageDataGenerator
from resnet50 import ResNet50

data_dir = './Dataset_Grayscale/'
target_size = (224,224)
classes = 24
train_datagen = ImageDataGenerator(rescale=1./255,
                                    zoom_range=0.1,
                                    horizontal_flip=False,
                                    validation_split=0.3)
train_gen = train_datagen.flow_from_directory(data_dir,target_size=target_size,
                                              shuffle = True, batch_size =32,
                                              color_mode='grayscale',
                                              class_mode='categorical',
                                              subset='training')
val_gen = train_datagen.flow_from_directory(data_dir,target_size=target_size,
                                            batch_size=32,
                                            color_mode='grayscale',
                                            class_mode='categorical',
                                            subset='validation')

model = ResNet50(224,224,classes,1)

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])

#model.summary()

train = model.fit_generator(train_gen, epochs=17, validation_data=val_gen)

import matplotlib.pyplot as plt

accuracy = train.history['accuracy']
loss = train.history['loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Akurasi dan loss Training ResNet-50')    
plt.legend()
plt.figure()
plt.show()

accuracy = train.history['accuracy']
val_accuracy = train.history['val_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Akurasi Training dan validation ResNet-50')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Loss Training dan validation ResNet-50')
plt.legend()
plt.show()

# Saving the model
model_json = model.to_json()
with open("final_model_resnet-gssdg2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('weight_model_resnet-gssdg2.h5')