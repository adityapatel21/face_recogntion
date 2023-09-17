import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='imagesattendence'
images=[]
Avengers=[]
myimages=os.listdir(path)
print(myimages)

#reading the images
for SH in myimages:
    currSH=cv2.imread(f'{path}/{SH}')
    images.append(currSH)
    Avengers.append(os.path.splitext(SH)[0])

print (Avengers)

def markAttendence(name):
    with open('attendence.csv','r+') as f:
        myAttendencelist = f.readlines()
        namelist=[]
        for line in myAttendencelist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            datestring =now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring}')

#encoding all the images
def encoding(images):
    enclist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        enclist.append(encode)
    return enclist
encodelist=encoding(images)
print ("------encoding Completed------")

#capturing the video
cap = cv2.VideoCapture(0)

#encoding the image from camera
while True:
    success, pic = cap.read()
    Cimg=cv2.resize(pic,(0,0),None,0.50,0.50)
    Cimg=cv2.cvtColor(Cimg,cv2.COLOR_BGR2RGB)

    faceinframe=face_recognition.face_locations(Cimg)
    encodecurframe=face_recognition.face_encodings(Cimg,faceinframe)
    
    for encodeface,faceloc in zip(encodecurframe,faceinframe):
        matches=face_recognition.compare_faces(encodelist,encodeface)
        facedis=face_recognition.face_distance(encodelist,encodeface)
        matchindex=np.argmin(facedis)

        if matches[matchindex]:
            name=Avengers[matchindex].upper()
            # print(name)
            # print(faceloc)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*2,x2*2,y2*2,x1*2
            cv2.rectangle(pic,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(pic,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(pic,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            markAttendence(name)
    
    cv2.imshow('webcam',pic)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows
        

