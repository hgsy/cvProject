import numpy as np
import cv2

img = cv2.imread('road1.jpg',1)
cv2.imshow('Original',img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray',gray)
edges = cv2.Canny(gray,250,500,apertureSize = 3)
# cv2.imshow('Edges',edges)
# lines = cv2.HoughLines(edges,1,np.pi/180,170)


# cv2.imshow('Lines',img)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
