#Modules required
import numpy as np
import face_recognition as fr
from datetime import datetime as dt
import cv2 as cv

#Marking attendance in already made csv file
def attendance(name):
    with open('attendance.csv', 'r+') as f:
        nameList = []
        for line in f.readlines():
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            recTime = dt.now()
            tstr = dt.strftime('%H:%M:%S')
            dstr = dt.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tstr},{dstr}')

#Dectecting faces after training
def detect_face(encodings, images, people):
    while True:
        print('For in-built camera enter 0\tFor external camera enter 1')
        a = input()
        try:
            if int(a) == 0 or int(a) == 1:
                break 
            else:
                print('Invalid Input')
        except ValueError:
            print('Please enter either 1 or 0')
    cap = cv.VideoCapture(int(a))
    while True:
        ret, frame = cap.read()
        faces = cv.resize(frame, (0,0), None, 0.25, 0.25)
        faces = cv.cvtColor(faces, cv.COLOR_BGR2RGB)

        faceCurrFrame = fr.face_locations(faces)
        faceEncode = fr.face_encodings(faces,faceCurrFrame)
        for face_enc, faceLoc in zip(faceEncode, faceCurrFrame):
            matches = fr.compare_faces(encodings, face_enc)
            faceDist = fr.face_distance(encodings, face_enc)
            mindex = np.argmin(faceDist)

            if matches[mindex]:
                name = people[mindex]
                y1, x2, y2, x1 = faceLoc
                x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4
                cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv.rectangle(frame, (x1,y2-35), (x2,y2), (0,255,0), cv.FILLED)
                cv.putText(frame, name, (x1+6,y2-6), cv.FONT_ITALIC, 1, (0,255,0), 2)
                attendance(name)

        cv.imshow('Face Dectected',frame)
        if cv.waitKey(10) == 13:
            break
    cap.release()
    cv.destroyAllWindows()