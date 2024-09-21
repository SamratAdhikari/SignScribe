from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import tensorflow as tf
from HandTrackingModule import HandDetector
from tensorflow.keras.preprocessing import image
from PIL import Image
import math
import io
import time
import warnings
from flask_cors import CORS


warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={r"/socket.io/*": {"origins": "https://signscribe-q1ki.onrender.com"}})
socketio = SocketIO(app, async_mode='eventlet')

# Load the ASL model
string = ""
offset = 20
img_size = 128
prev_char = None
start_time = None
model = tf.keras.models.load_model('assets/model.h5', compile=False)
detector = HandDetector(maxHands=1)  # Move outside to avoid re-initialization

with open('assets/categories.pkl') as file:
    categories = eval(file.read())

def process_image(img_data):
    img_data = img_data.split(',')[1]  # Remove the data:image/jpeg;base64, part
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def handle_frame(img):
    global string, prev_char, start_time
    img = process_image(img)

    hands, img = detector.findHands(img)
    img = cv2.flip(img, 1)  # flip the frame horizontally

    # Initialize char and prob with default values
    char = ''
    prob = ''

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y-offset : y+h+offset, -x-w+offset-35 : -x+offset]
        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255

        aspectRation = h / w

        if aspectRation > 1:
            k = img_size / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, img_size))
            wGap = math.ceil((img_size - wCal) / 2)
            imgWhite[:, wGap : wCal + wGap] = imgResize

        else:
            k = img_size / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (img_size, hCal))
            hGap = math.ceil((img_size - hCal) / 2)
            imgWhite[hGap : hCal + hGap, :] = imgResize

        # imgWhite = cv2.flip(imgWhite, 1)
        my_image_arr = image.img_to_array(imgWhite)
        my_image_pixel = np.expand_dims(my_image_arr, axis=0)
        my_image_pixel = my_image_pixel / 255
        prediction = model.predict(my_image_pixel, verbose=False)
        prediction_class = np.argmax(prediction, axis=1)

        char = categories[tuple(prediction_class)[0]]
        prob = str(round(np.max(prediction), 2))

        if char == prev_char:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= 1:  # Check if  seconds have passed

                print("here")
                if char == 'SPACE':
                    char = "  "
                
                string += char
                start_time = None
        else:
            start_time = None
            prev_char = char

    return string, char, prob

@socketio.on('frame')
def receive_frame(data):
    try:
        text, char, prob = handle_frame(data)
        emit('frame_processed', {
            'status': 'success',
            'text': text,
            'char': char,
            'prob': prob
        }, broadcast=True)
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('frame_processed', {
            'status': 'error',
            'message': str(e)
        }, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    # Use eventlet as the web server for production
    socketio.run(app, host='0.0.0.0', port=8501, debug=True, allow_unsafe_werkzeug=True, server_options={"use_reloader": False}, **{'async_mode': 'eventlet'})


    # app.run(debug=True)
