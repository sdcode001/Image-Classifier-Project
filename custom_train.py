#TensorFlow Keras is an implementation of the Keras API that uses TensorFlow as a backend.
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder





dataset_path = os.listdir('dataset')
print(dataset_path)


#-----------------------------------------------------------------------------------------------------------------------

class_labels = []
# creating a list of tuples of respective image class and image path
for item in dataset_path:
    images_name = os.listdir('dataset'+'/'+item)
    for image_name in images_name:
        class_labels.append((item, str('dataset'+'/'+item+'/'+image_name)))


# build a data-frame
df = pd.DataFrame(data=class_labels, columns=['Label', 'Image'])
print(df.head())
print(df.tail())

#-----------------------------------------------------------------------------------------------------------------------

'''
EfficientNet base models vs resolution
Model             Resolution
EfficientNetB0    224
EfficientNetB1    240
EfficientNetB2    260
EfficientNetB3    300
EfficientNetB4    380
EfficientNetB5    456
EfficientNetB6    528
EfficientNetB7    600
'''

# Images Pre-processing-------------------------------------------------------------------------------------------------
# Take the images and resize them(according to the EfficientNet base model resolution) and store in an list

# For EfficientNetB0
img_size = 224

images = []
labels = []

for item in dataset_path:
    images_name = os.listdir('dataset'+'/'+item)
    for image_name in images_name:
        img = cv2.imread('dataset'+'/'+item+'/'+image_name)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(item)

# Converting the images into array every EfficientNet model accepts input data range [0,255]
images = np.array(images)
images = images.astype('float32') / 255.0


# Lables Pre-processing-------------------------------------------------------------------------------------------------
# computer/deep learning models don't understand labels as string, it only accepts OneHotEncoded form of labels

y = df['Label'].values
print(y)

y_labelEncoder = LabelEncoder()
y = y_labelEncoder.fit_transform(y)
print(y)

# OneHotEncoding of labels........
y = y.reshape(-1, 1)
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y)
print(Y)

#-----------------------------------------------------------------------------------------------------------------------

# Now images and labels pre-processing are done so we will devide out dataset(images and labels) into train and test.

images, Y = shuffle(images, Y, random_state=1)
# as our dataset is very small so we keep the test size = 5%
train_images, test_images, train_Y, test_Y = train_test_split(images, Y, test_size=0.05, random_state=415)

#-----------------------------------------------------------------------------------------------------------------------


# EfficientNet implementation and training model on our dataset.

NUM_CLASSES = 3  #set it according to the dataset
IMG_SIZE = 224   #set it according to the EfficientNet Base model
size = (IMG_SIZE, IMG_SIZE)

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Here using B0 model for training, here we will train the model and not using pre-train model so weights=None
outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

model = tf.keras.Model(inputs, outputs)
# configuring the model with optimizer and others.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# Finally training starts with train_data. Here the model will be trained for several epochs and
# after every epochs weights will updated and different optimizer are used to get different weights values.
with tf.device('/GPU:0'):
    hist = model.fit(train_images, train_Y, epochs=30, verbose=2)
    print(hist)
# training process is stored in hist.
#-----------------------------------------------------------------------------------------------------------------------

# Working with the test_data and see the accuracy of our trained model.
preds = model.evaluate(test_images, test_Y)
print('Test accuracy: '+str(preds[1]))
print('Loss: '+str(preds[0]))

#-----------------------------------------------------------------------------------------------------------------------

# Saving the trained model on local disk.
model.save('trained_model.h5')
































