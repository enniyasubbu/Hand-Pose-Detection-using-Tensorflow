
  
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
############################################### Intialize Parameters ##########################################################################
Filters=32
EPOCHS=50
Activation_fn='relu'
Image_width=100
Image_height=100
Optimizer='adam'
Learning_rate=0.001
Snapshot=100
Num_classes=3
stride=2
# global variables
bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    #wpercent = (basewidth/float(img.size[0]))
    hsize = 100
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=20):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(1)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
            print num_frames
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            start_recording = True

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(100, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    #if predictedClass == 0:
    #    className = "Super"
    if predictedClass == 0:
        className = "Palm"
    elif predictedClass == 1:
        className = "Fist"
    elif predictedClass == 2:
        className = "Thumbsup"
   # elif predictedClass == :
    #    className = "Victory"
    #elif predictedClass == 5:
    #    className = "Why"

    cv2.putText(textImage,"Predicted Class : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)




# Model defined
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

# Load Saved Model
model.save("TrainedModel/Handposesbacksub.tfl")

main()
