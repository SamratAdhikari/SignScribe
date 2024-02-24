import os
import cv2
import math
import pickle
import numpy as np
import tensorflow as tf
from HandTrackingModule import HandDetector
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, Response, jsonify



app = Flask(__name__)
cam = cv2.VideoCapture(0)


def backgroundSubtraction(img):
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsvim, lower, upper)
    # Blur the mask to help remove noise
    skinMask = cv2.blur(skinMask, (2, 2))
    # Get threshold image
    _, thresh = cv2.threshold(skinMask, 100, 255, cv2.THRESH_BINARY)

    # Find the hand contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = max(contours, key=lambda x: cv2.contourArea(x))
    # Create a black background
    black_bg = np.zeros_like(img)

    # Draw the hand contour on the black image
    cv2.drawContours(black_bg, [hand_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Apply the mask to the original frame
    hand_pixels = cv2.bitwise_and(img, black_bg)

    return hand_pixels



offset = 30
img_size = 128

model = tf.keras.models.load_model('assets/vgg16_model.h5')
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

with open('assets/categories.pkl') as file:
    categories = eval(file.read())

def gen_frames():
    while True:
        success, frame = cam.read()
      
        if not success:
            break
        
        else:            
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

                
                print('char:', c)
                print('prob:', p)
                # # Create a larger canvas
                # newWindow = np.zeros((frameWhite.shape[0] + 50, frameWhite.shape[1], 3), dtype=np.uint8)

                # # Paste the original image onto the new canvas
                # newWindow[:frameWhite.shape[0], :] = frameWhite

                # cv2.putText(newWindow, c, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.putText(newWindow, p, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            
            # yield(b'--frame\r\n'
            #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'rn')
            return jsonify({'char': c, 'prob': p})

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/video', methods = ['POST', 'GET'])
def video():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)