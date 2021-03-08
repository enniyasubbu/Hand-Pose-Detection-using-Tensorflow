import cv2

image = cv2.imread('fist_0.png')
print("The type of this input is {}".format(type(image)))
print("Shape: {}".format(image.shape))
cv2.imshow('image',image)
