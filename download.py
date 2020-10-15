import urllib.request
import pathlib
import cv2
import numpy as np
import json

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def DownloadFile(fileURL, tag, cnt):
    print('Downloading image...' + str(cnt))

    pathlib.Path('./' + tag).mkdir(exist_ok=True)
    pathlib.Path('./etccc').mkdir(exist_ok=True)


    #fileName = './' + tag + '/insta' + str("%06d"%cnt) + '.jpg'
    #urllib.request.urlretrieve(fileURL, fileName)
    #print('Done. ' + fileName)

    #res = urllib.request.urlopen(fileURL)
    #data = res.read()
    #img = cv2.imread(fileURL)

    #cv2.imshow('image', img)

    resp = urllib.request.urlopen(fileURL)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        cv2.imwrite('./' + tag + '/' + tag + '_' + str("%06d"%cnt) + '.jpg', img)
    else:
        cv2.imwrite('./etccc/' + tag + '_' + str("%06d"%cnt) + '.jpg', img)


if __name__ == '__main__':
    with open('C:/Users/Joseph/PycharmProjects/datavoucher/selfie.json', 'rt', encoding='UTF-8') as data_file:
        data = json.load(data_file)

    tag = 'selfie'
    for i in range(0, len(data)):
        instagramURL = data[i]['img_url']
        DownloadFile(instagramURL, tag, i)