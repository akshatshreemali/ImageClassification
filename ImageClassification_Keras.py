# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:40:24 2020

@author: as19665
"""



# Importing libraries
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, load_model, save_model, Model
from tensorflow.python.keras.layers import Dense, Dropout, InputLayer, Input,Masking,SpatialDropout2D
from tensorflow.python.keras.layers import Reshape, Flatten, MaxPooling2D, Conv2D,BatchNormalization
#from tensorflow.keras.layers import Embedding,LSTM
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np


# Creating train and test directories
train_data_dir = 'C:/Users/as19665/Desktop/ICLR Challenge/train_new/train'

test_data_dir = 'C:/Users/as19665/Desktop/ICLR Challenge/test/'

# Part 2 - Fitting the CNN to the images

img_height=256
img_width=256
batch_size=32

# Creating train and validation data fro
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

num_classes = len(train_generator.class_indices)
print(num_classes)


# define class weights
leaf_rust_count = 859
stem_rust_count = 900
healthy_wheat_count = 339
total = leaf_rust_count + stem_rust_count + healthy_wheat_count

leaf_rust_weight = (1/leaf_rust_count) * (total) / 3.0
stem_rust_weight = (1/stem_rust_count) * (total) / 3.0
healthy_wheat_weight = (1/healthy_wheat_count) * (total) / 3.0

class_weight = {0:healthy_wheat_weight, 1:leaf_rust_weight, 2:stem_rust_weight}
print(class_weight)

IMG_SHAPE = (256, 256, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.Resnet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 130

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

# add new classifier layers
flat1 = Flatten()(base_model.layers[-1].output)
fcon1 = Dense(256, activation='relu')(flat1)
fdrop1 = Dropout(0.25)(fcon1)
fbn1 = BatchNormalization()(fdrop1)

fcon2 = Dense(256, activation='relu', kernel_initializer='he_uniform')(fbn1)
fdrop2 = Dropout(0.25)(fcon2)
fbn2 = BatchNormalization()(fdrop2)

#fcon3 = Dense(256, activation='relu', kernel_initializer='he_uniform')(fbn2)
#fdrop3 = Dropout(0.25)(fcon3)
#fbn3 = BatchNormalization()(fdrop3)

output = Dense(num_classes, activation='softmax')(fbn2)

# define new model
model = Model(inputs=base_model.inputs, outputs=output)

# compile model
opt = Adam(lr=0.0001)
#SGD(lr=0.01, momentum=0.9,decay=0.005)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


model.fit_generator(train_generator,
                    steps_per_epoch = len(train_generator), # batch_size,
                    validation_data = validation_generator, 
                    validation_steps = len(validation_generator), # batch_size,
                    epochs = 14,
                    class_weight=class_weight)


model.save('C:/Users/as19665/Desktop/ICLR Challenge/finetuning_model_120.h5')

old_model = tf.keras.models.load_model('C:/Users/as19665/Desktop/ICLR Challenge/finetuning_model.h5')

# Predicting on the test dataset
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    classes=['test'],
    shuffle=False    
)

def str_clean(s):
    st= s.split('\\')[1].split('.')[0]
    return st

preds = old_model.predict_generator(test_generator)
pred_list = dict((v,k) for k,v in train_generator.class_indices.items())
#prednames = [pred_list[k] for k in preds_cls_idx]
filenames = test_generator.filenames
prediction_data = pd.DataFrame(preds)
columns = ['healthy_wheat','leaf_rust','stem_rust']
prediction_data.columns = columns
filenames = pd.DataFrame(filenames)
filenames.head()
filenames[0] = filenames[0].apply(lambda x:(str_clean(x)))
result = filenames.join(prediction_data,how='inner')


result.to_csv('C:/Users/as19665/Desktop/ICLR Challenge/finetuning_resnet_256_32_15_old.csv',index=False)


