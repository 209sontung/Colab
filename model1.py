import cv2
import keras
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import onnx

# model = onnx.load('mymodel.onnx')
model = load_model('model_85%.h5')
# img = cv2.imread('trash8.jpg')


labels = ['G&M', 'Organic', 'Other', 'Paper', 'Plastic']
le = LabelEncoder()
labels = le.fit_transform(labels)


def pre(img_path):
    img = cv2.resize(img_path, (224, 224))
    img = img_to_array(img)
    print(np.max(img))
    img = imagenet_utils.preprocess_input(img)
    print(np.max(img))
    print(np.min(img))
    img = np.expand_dims(img / 255, 0)
    print(np.max(img))
    print(np.min(img))
    return img

cap = cv2.VideoCapture(0)
address = 'https://172.16.8.146:8080/video'
cap.open(address)
success, img = cap.read()

font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)
org1 = (50, 150)

# fontScale
fontScale = 3

# Green color in RGB
color = (0, 255, 0)

# Line thickness of 2 px
thickness = 2

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


def Mouse_event(event, x, y, f, img):

    if event == cv2.EVENT_LBUTTONDOWN:
        Mouse_event.x0 = x
        Mouse_event.y0 = y
        Mouse_event.draw = True
    if event == cv2.EVENT_LBUTTONUP:
        Mouse_event.x1 = x
        Mouse_event.y1 = y
        Mouse_event.draw = False
        miny = min(Mouse_event.y0,Mouse_event.y1)
        maxy = max(Mouse_event.y0, Mouse_event.y1)

        minx = min(Mouse_event.x0, Mouse_event.x1)
        maxx = max(Mouse_event.x0, Mouse_event.x1)
        Mouse_event.img = img[miny:maxy,minx:maxx]
    if event == cv2.EVENT_MOUSEMOVE:
        Mouse_event.x = x
        Mouse_event.y = y

Mouse_event.img = None
Mouse_event.x0 =0
Mouse_event.y0 =0
Mouse_event.x1 =0
Mouse_event.y1 =0
Mouse_event.x =0
Mouse_event.y =0
Mouse_event.draw = False

while True:
    start_time = time.time()
    success, img = cap.read()
    img1 = pre(img)
    p = model.predict(img1)
    print(p.shape)
    confidences = max(np.squeeze(model.predict(img1)))
    conf = round(confidences, 3)

    predicted_class = le.classes_[np.argmax(p[0], axis=-1)]
    print('FPS:', 1.0 / (time.time() - start_time))
    if float(to_str(conf)) > 0.60:
        cv2.putText(img, predicted_class, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(img, to_str(conf), org1, font,
                    fontScale, color, thickness, cv2.LINE_AA)
    # cv2.imshow('Model1', img_lone)
    # cv2.setMouseCallback("Model1", Mouse_event, img)
    cv2.imshow('Model1', img)
    k = cv2.waitKey(1)

