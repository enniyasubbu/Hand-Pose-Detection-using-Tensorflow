# HandPose_Detection
Background Subtraction for pose detection using Convolutional Neural Network
The work is inspired from https://gogul09.github.io/software/hand-gesture-recognition-p1

## Dataset Preparation
    Run python2 dataset.py to prepare the dataset
    Classes=Super,Palm,Fist,Thumbs_up,Victory,Why
    run the code after 30 frame pose your hand in the bounding box displayed on the video
## Training    
    Then Resize the images using resize.py
    The file read.py helps to find the image resolution and no of channels which can be helpful reshaping
    set some default parameters
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
    Then Run python2 train.py to train the dataset
    
## Results
    The results were uploaded in the above
    Run predict.py to detect 
    
    
