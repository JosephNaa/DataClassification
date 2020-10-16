import urllib.request
import pathlib
import cv2
import numpy as np
import json
import dlib
import argparse

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
gender_list = ['Male', 'Female']
detector = dlib.get_frontal_face_detector()
gender_net = cv2.dnn.readNetFromCaffe(
    './models/deploy_gender.prototxt',
    './models/gender_net.caffemodel'
)


def ClassifyGender(fileURL, tag, cnt):
    pathlib.Path('./' + tag).mkdir(exist_ok=True)
    pathlib.Path('./' + tag + '_male').mkdir(exist_ok=True)
    pathlib.Path('./' + tag + '_more').mkdir(exist_ok=True)
    pathlib.Path('./' + tag + '_except').mkdir(exist_ok=True)
    pathlib.Path('./' + tag + '_not_detect').mkdir(exist_ok=True)

    resp = urllib.request.urlopen(fileURL)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    faces = detector(img)
    # print(faces)

    print('Downloading image...' + str(cnt))

    if len(faces) > 1:
        cv2.imwrite('./' + tag + '_more/' + tag + '_' + str("%06d" % cnt) + '.jpg', img)

    elif not faces:
        cv2.imwrite('./' + tag + '_not_detect/' + tag + '_' + str("%06d" % cnt) + '.jpg', img)

    else:
        x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
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
                cv2.imwrite('./' + tag + '_male/' + tag + '_' + str("%06d" % cnt) + '.jpg', img)

        except Exception as e:
            print(str(e))
            cv2.imwrite('./' + tag + '_except/' + tag + '_' + str("%06d" % cnt) + '.jpg', img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--tag")

    args = parser.parse_args()

    with open('./' + args.tag + '.json', 'rt', encoding='UTF-8') as data_file:
        data = json.load(data_file)

    for i in range(0, len(data)):
        instagramURL = data[i]['img_url']
        ClassifyGender(instagramURL, args.tag, i)


#####################################
# python download.py -t selfie
#####################################
