import cv2
from tensorflow.keras.applications import ResNet50
from keras import layers
import numpy as np

cap = cv2.VideoCapture(0)

resnet = ResNet50(
    include_top=False, 
    weights=None, 
    input_tensor= layers.Input(shape=(512, 512, 3)), 
    input_shape = (512, 512, 3),
    pooling='avg',
    )

if not cap.isOpened():
    print("Camera can not be opened")
while True:
    ret, frame = cap.read()
    if not ret:
        print("video capture failed")
        break
    image = cv2.resize(frame, (512, 512))
    image = np.expand_dims(image, axis=0)
    result = resnet.predict(image)
    cv2.imshow("CV2 video feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()