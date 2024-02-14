import os
import cv2
import math
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from HandTrackingModule import HandDetector
from tensorflow.keras.preprocessing import image

# Load the ASL model
offset = 20
img_size = 128
model = tf.keras.models.load_model('vgg16_model5.h5')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
st.subheader('ASL recognition')

with open('categories.pkl') as file:
    categories = eval(file.read())

gotOutput = False
string = ''
string_placeholder = st.empty()
frame_placeholder = st.empty()
stop_button_pressed = st.button('Stop')

while cap.isOpened() and not stop_button_pressed:
    if cv2.waitKey(10) & 0xFF == 27:
        break

    success, img = cap.read()
    hands, img = detector.findHands(img)
    img = cv2.flip(img, 1) # flip the frame horizontally


    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgCrop = img[y-offset : y+h+offset, -x-w+offset-35 : -x+offset]
        imgWhite = np.ones((img_size, img_size, 3), np.uint8)*255

        aspectRation = h/w

        if aspectRation > 1:
            try:
                k = img_size / h
                wCal = math.ceil(k*w)

                imgResize = cv2.resize(imgCrop, (wCal, img_size))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((img_size - wCal) / 2)
                imgWhite[:, wGap : wCal+wGap] = imgResize

            except Exception as e:
                pass

        else:
            try:
                k = img_size / w
                hCal = math.ceil(k*h)

                imgResize = cv2.resize(imgCrop, (img_size, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((img_size - hCal) / 2)
                imgWhite[hGap : hCal+hGap, :] = imgResize


            except Exception as e:
                pass


        try:
            pass
            # cv2.imshow('', imgWhite)

        except Exception as e:
            pass

        imgWhite = cv2.flip(imgWhite, 1)
        my_image_arr = image.img_to_array(imgWhite)
        my_image_pixel = np.expand_dims(my_image_arr, axis=0)
        my_image_pixel = my_image_pixel / 255
        # my_image_pixel = preprocess_input(my_image_pixel)
        prediction = model.predict(my_image_pixel, verbose=False)
        prediction_class = np.argmax(prediction, axis=1)

        char = categories[tuple(prediction_class)[0]]
        prob = str(round(np.max(prediction), 2))
        cv2.putText(img, char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (92, 13, 195), 3, cv2.LINE_AA)
        cv2.putText(img, prob, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (92, 13, 195), 3, cv2.LINE_AA)

        gotOutput = True
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img, channels='RGB')

# Print the last predicted label before the hand is removed
print("Last predicted label before hand removal:", char)
if gotOutput:
    string += char
    gotOutput = False

string_placeholder.write(f'Text: {string}')

cap.release()
cv2.destroyAllWindows()