import numpy as np
import cv2 as cv
import glob # 파일 여러장 가져오기

face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

n = 0
for file in glob.glob("./test/*.jpg"):
    print(file)
    img=cv.imread(file)

    cv.imshow("image", img)
    cv.waitKey(0)
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray=cv.resize(gray,(200,200))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        print(n)
        cv.imwrite('./test1/beauty_'+str(n)+'.jpg', img)
        n+=1
    else:
        print("no face")