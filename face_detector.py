
import numpy as np
import cv2
from matplotlib import pyplot as plt

xml = 'xml/haarcascade_frontalcatface_default.xml'
# image = cv2.imread('image/testImage.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detect = cv2.CascadeClassifier(xml)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# faces = face_cascade.detectMultiScale(gray, 1.05, 5)
#
# if len(faces)!=0:
#     for face in faces:
#         #face >> [x,y,w,h]
#         cv2.rectangle(image,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0),2)
#
# plt.imshow(image, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.show()

cam = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):
    ret, video = cam.read()
    fgmask = fgbg.apply(video)
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 130, 130, apertureSize=3)
    faces = detect.detectMultiScale(gray, 1.05, 5)
    if len(faces) != 0:
        for face in faces:
                #face >> [x,y,w,h]
                cv2.rectangle(fgmask,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0),2)
    #
    # cv2.imshow('result', edges)
    cv2.imshow('frame', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release
cv2.destroyAllWindows()