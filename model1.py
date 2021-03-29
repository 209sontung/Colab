import cv2
import keras
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import os
import time

model = load_model('model_5class_densenet_86%.h5')
# img = cv2.imread('trash8.jpg')


image_path = list(paths.list_images('Dataset'))
labels = [p.split(os.path.sep)[-2] for p in image_path]
le = LabelEncoder()
labels = le.fit_transform(labels)


def pre(img_path):
    img = cv2.resize(img_path, (224, 224))
    img = img_to_array(img)
    img = imagenet_utils.preprocess_input(img)
    img = np.expand_dims(img / 255, 0)
    return img

cap = cv2.VideoCapture(0)
address = 'https://192.168.1.31:8080/video'
cap.open(address)
success, img = cap.read()

font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)
org1 = (50, 150)

# fontScale
fontScale = 3

# Green color in RGB
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

while True:
    start_time = time.time()
    success, img = cap.read()
    img1 = pre(img)
    p = model.predict(img1)
    confidences = max(np.squeeze(model.predict(img1)))
    conf = round(confidences, 3)

    predicted_class = le.classes_[np.argmax(p[0], axis=-1)]
    print('FPS:', 1.0 / (time.time() - start_time))
    if float(to_str(conf)) > 0.77:
        cv2.putText(img, predicted_class, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(img, to_str(conf), org1, font,
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Model1', img)
    k = cv2.waitKey(1)

