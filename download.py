import urllib.request
import pathlib
import cv2
import numpy as np
import json
import dlib

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
gender_list = ['Male', 'Female']
detector = dlib.get_frontal_face_detector()
gender_net = cv2.dnn.readNetFromCaffe(
    './models/deploy_gender.prototxt',
    './models/gender_net.caffemodel'
)

def ClassifyGender(fileURL, tag, cnt):
    pathlib.Path('./' + tag).mkdir(exist_ok=True)
    pathlib.Path('./male').mkdir(exist_ok=True)
    pathlib.Path('./more').mkdir(exist_ok=True)
    pathlib.Path('./etccc').mkdir(exist_ok=True)

    resp = urllib.request.urlopen(fileURL)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    faces = detector(img)

    print('Downloading image...' + str(cnt))

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy()

        try:
            blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
                                     mean=(78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False, crop=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            if gender == 'Female':
                cv2.imwrite('./' + tag + '/' + tag + '_' + str("%06d" % cnt) + '.jpg', img)
            else:
                cv2.imwrite('./male/' + tag + '_' + str("%06d" % cnt) + '.jpg', img)

        except Exception as e:
            print(str(e))
            cv2.imwrite('./etccc/' + tag + '_' + str("%06d" % cnt) + '.jpg', img)

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
    tag = 'selfie'

    with open('./' + tag + '.json', 'rt', encoding='UTF-8') as data_file:
        data = json.load(data_file)

    for i in range(0, len(data)):
        instagramURL = data[i]['img_url']
        ClassifyGender(instagramURL, tag, i)
