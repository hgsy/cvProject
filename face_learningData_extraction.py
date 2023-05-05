import cv2
import numpy as np
xml = 'xml\haarcascade_frontalcatface_default.xml'
detector = cv2.CascadeClassifier(xml)

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3,5)
    if len(faces) == 0:
        return None
    for(x,y,w,h) in faces:
        # face >> [x,y,w,h]
        cropped_face = img[y:y+h,x:x+w]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

name = 'geun'

while(True):
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face_extractor(frame),(200,200))

        path = 'cropface\\'+name +'_'+ str(count)+'.jpg'
        cv2.imwrite(path, face)
        print(count)
        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
        cv2.imshow('result',face)
    else:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 13 or count == 100:
        break
cap.release()
cv2.destroyAllWindows()
print('Done')

