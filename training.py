#Modules required
from msilib.schema import File
import face_recognition as fr
import os
import cv2 as cv

#Training the system for given faces
def train_system():
    while True:
        try:
            print('Enter folder path')
            path = input(r'')
            mylist = os.listdir(path)
            break
        except FileNotFoundError:
            print('Folder not found\nPlease try again!')
    images = []
    people = []
    for cur_img in mylist:
        img  = cv.imread(f'{path}\{cur_img}')
        images.append(img)
        people.append(os.path.splitext(cur_img)[0])
    


    def faceEncoding(images):
        encodeList = []
        for img in images:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            encode = fr.face_encodings(img)[0]
            encodeList.append(encode)

    encodings = faceEncoding(images)
    print('System is Trained for Given Faces')
    return encodings, people, images