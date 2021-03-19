import numpy
import cv2
import os

people = []

for i in os.listdir(r'C:\Users\flore\Desktop\PeyeView\Photos\train'):
    people.append(i)


haar_cascades = cv2.CascadeClassifier("haar_face.xml")

features = numpy.load("features.npy", allow_pickle = True)
labels = numpy.load("labels.npy", allow_pickle = True)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv2.imread('Photos/elton.jpg')

cap = cv2.VideoCapture(0) # Open up camera


while(True):
    ret, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_rect = haar_cascades.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10 )


    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)

        cv2.putText(gray, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255,0,0), thickness=1)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (0,0,255), thickness= 1)
        
        cv2.imshow('Live Footage of Me', gray)
        
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()