import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import time
from time import sleep
import cv2
from sklearn.utils import shuffle
import os
############################################### Intialize Parameters ##########################################################################
Filters=32
EPOCHS=50
Activation_fn='relu'
Image_width=100
Image_height=100
Optimizer='adam'
Learning_rate=0.001
Snapshot=100
Num_classes=6
stride=2
########################################### Image shapping declare globally #####################################################################
ReshapeImages = []
outputvalues = []
testReshape = []
Labels_test = []

########################################### Train Images Load ###################################################################################
#Load Images from Super

for i in range(0, 1000):
    imagepath='Dataset/Super/super_' + str(i) + '.png'
    image = cv2.imread('Dataset/Super/super_' + str(i) +'.png')
    #cv2.imshow('1',image)
    #time.sleep(2)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ReshapeImages.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Palm
for i in range(0, 1000):
    image = cv2.imread('Dataset/Palm/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ReshapeImages.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Fist
for i in range(0, 1000):
    image = cv2.imread('Dataset/Fist/Fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ReshapeImages.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Thumbs_up
for i in range(0, 1000):
    image = cv2.imread('Dataset/Thumbs_up/Thumb_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ReshapeImages.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Victory
for i in range(0, 1000):
    image = cv2.imread('Dataset/Victory/Victory_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ReshapeImages.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Why
for i in range(0, 1000):
    image = cv2.imread('Dataset/Why/why_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ReshapeImages.append(gray_image.reshape(Image_width, Image_height, 1))


# Create OutputVector


for i in range(0, 1000):
   outputvalues.append([1, 0, 0])

for i in range(0, 1000):
    outputvalues.append([0, 1, 0])

for i in range(0, 1000):
    outputvalues.append([0, 0, 1])

for i in range(0, 1000):
    outputvalues.append([0, 0, 0,1,0,0])

for i in range(0, 1000):
    outputvalues.append([0, 0, 0,0,1,0])

for i in range(0, 1000):
    outputvalues.append([0, 0, 0,0,0,1])

########################################################### Test/Validation Set Image ###################################################################################

#Test

#Load Images from Super

for i in range(1000, 1100):
    image = cv2.imread('Dataset/Super_test/super_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testReshape.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Palm
for i in range(1000, 1101):
    image = cv2.imread('Dataset/Palm_test/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testReshape.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Fist
for i in range(1000, 1101):
    image = cv2.imread('Dataset/Fist_test/Fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testReshape.append(gray_image.reshape(Image_width, Image_height, 1))


#Load Images From Thumbs_up
for i in range(1000, 1101):
    image = cv2.imread('Dataset/Thumbs_up_test/Thumb_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testReshape.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Victory
for i in range(1000, 1100):
    image = cv2.imread('Dataset/Victory_test/Victory_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testReshape.append(gray_image.reshape(Image_width, Image_height, 1))

#Load Images From Why
for i in range(1000, 1100):
    image = cv2.imread('Dataset/Why_test/why_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testReshape.append(gray_image.reshape(Image_width, Image_height, 1))




for i in range(1000, 1101):
    Labels_test.append([1, 0, 0])

for i in range(1000, 1101):
   Labels_test.append([0, 1, 0])

for i in range(1000, 1101):
    Labels_test.append([0, 0, 1])

for i in range(1000, 1100):
    Labels_test.append([0, 0, 0,1,0,0])

for i in range(1000, 1100):
    Labels_test.append([0, 0, 0,0,1,0])

for i in range(1000, 1100):
    Labels_test.append([0, 0, 0,0,0,1])



############################################ CNN Model ##############################################################################################################
# Define the CNN Model
tf.reset_default_graph()
# INPUT Layer.........................
convnet=input_data(shape=[None,Image_width, Image_height,1],name='input')
#Layer 1
convnet=conv_2d(convnet,Filters,stride,activation=Activation_fn)
convnet=max_pool_2d(convnet,2)
#Layer 2
convnet=conv_2d(convnet,Filters*2,stride,activation=Activation_fn)
convnet=max_pool_2d(convnet,2)
#Layer 3
convnet=conv_2d(convnet,Filters*4,stride,activation=Activation_fn)
convnet=max_pool_2d(convnet,2)
#Layer 4
convnet=conv_2d(convnet,Filters*8,stride,activation=Activation_fn)
convnet=max_pool_2d(convnet,2)
#Layer 5
convnet=conv_2d(convnet,Filters*8,stride,activation=Activation_fn)
convnet=max_pool_2d(convnet,2)
#Layer 6
convnet=conv_2d(convnet,Filters*4,stride,activation=Activation_fn)
convnet=max_pool_2d(convnet,2)
#Layer 7
convnet=conv_2d(convnet,Filters*2,stride,activation=Activation_fn)
convnet=max_pool_2d(convnet,2)
#Layer 8
convnet=fully_connected(convnet,1000,activation=Activation_fn)
convnet=dropout(convnet,0.75)
#Softmax
convnet=fully_connected(convnet,Num_classes,activation='softmax')

convnet=regression(convnet,optimizer=Optimizer,learning_rate=Learning_rate,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Shuffle Training Data
ReshapeImages, outputvalues = shuffle(ReshapeImages, outputvalues, random_state=0)

# Train model
model.fit(ReshapeImages, outputvalues, n_epoch=EPOCHS,
           validation_set = (testReshape, Labels_test),
           snapshot_step=Snapshot, show_metric=True, run_id='convnet_coursera')

model.save("TrainedModel/Handposesbacksub.tfl")

print ("Done.............")
