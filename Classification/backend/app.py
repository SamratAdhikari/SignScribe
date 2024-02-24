import os
import cv2
import math
import base64
import pickle
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from HandTrackingModule import HandDetector
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)
CORS(app, supports_credentials=True, origins='http://localhost:3000')
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for WebSocket


with open('assets/categories.pkl') as file:
    categories = eval(file.read())
offset = 30
img_size = 128
detector = HandDetector(maxHands=1)
model = tf.keras.models.load_model('../assets/vgg16_model.h5')



def backgroundSubtraction(img):
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsvim, lower, upper)
    skinMask = cv2.blur(skinMask, (2, 2))
    _, thresh = cv2.threshold(skinMask, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = max(contours, key=lambda x: cv2.contourArea(x))
    black_bg = np.zeros_like(img)
    cv2.drawContours(black_bg, [hand_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    hand_pixels = cv2.bitwise_and(img, black_bg)

    return hand_pixels






def process_frame(frame):
    print(frame)
    try:
        # print('wish i was here')
        hands, frame = detector.findHands(frame)
        frame = cv2.flip(frame, 1)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            frameCrop = frame[y - offset : y + h + offset, -x - w + offset - 75 : -x + offset]

            aspect_ratio = h / w

            try:
                if aspect_ratio > 1:
                    k = img_size / h
                    wCal = math.ceil(k * w)
                    frameResize = cv2.resize(frameCrop, (wCal, img_size))
                    wGap = math.ceil((img_size - wCal) / 2)
                    frameWhite = np.zeros((img_size, img_size, 3), np.uint8) * 255
                    frameWhite[:, wGap : wCal + wGap] = frameResize
                else:
                    k = img_size / w
                    hCal = math.ceil(k * h)
                    frameResize = cv2.resize(frameCrop, (img_size, hCal))
                    hGap = math.ceil((img_size - hCal) / 2)
                    frameWhite = np.zeros((img_size, img_size, 3), np.uint8) * 255
                    frameWhite[hGap : hCal + hGap, :] = frameResize

            except Exception as e:
                pass

            frameWhite = cv2.flip(frameWhite, 1)
            frameWhite = backgroundSubtraction(frameWhite)
            frameArr = image.img_to_array(frameWhite)
            framePixel = np.expand_dims(frameArr, axis=0)
            framePixel /= 255

            prediction = model.predict(framePixel, verbose=False)
            prediction_class = np.argmax(prediction, axis=1)
            c = categories[tuple(prediction_class)[0]]
            p = str(round(np.max(prediction), 2))
            return c, p
    
    except Exception as e:
        print('nooooo')
        return 'A', 1.0
    

@socketio.on('image_frame')
def handle_image_frame(data):
    frame = data  # Assuming image_data contains the frame directly
    c, p = process_frame(frame)  # Process the frame to get 'c' and 'p'
    emit('result', {'c': c, 'p': p})

    # Emit a result back to the client if needed
    # For example, if you have a machine learning model to process the image
    # you can emit the result back to the client
    # emit('result', {'char': 'A', 'prob': 0.95})

if __name__ == '__main__':
    socketio.run(app, debug=True)
