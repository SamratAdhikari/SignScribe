from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from HandTrackingModule import HandDetector
import math
import gc

# Initialization
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

offset = 20
img_size = 128
string = ""
prev_char = None
start_time = None
client_connected = False

# Load the ASL model and categories
model = load_model('assets/model.h5')

with open('assets/categories.txt', 'r') as file:
    categories = eval(file.read())

# Initialize HandDetector
detector = HandDetector(maxHands=1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global client_connected
    client_connected = True
    print("Client connected, starting predictions.")

    try:
        while client_connected:
            frame = await websocket.receive_bytes()
            result = process_frame(frame)
            await websocket.send_json(result)
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        client_connected = False
        clear_memory()
        print("Client disconnected, stopping predictions.")

def process_frame(frame):
    global string, prev_char, start_time
    try:
        np_frame = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y-offset : y+h+offset, -x-w+offset-35 : -x+offset]
            imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = img_size / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, img_size))
                wGap = math.ceil((img_size - wCal) / 2)
                imgWhite[:, wGap : wCal+wGap] = imgResize
            else:
                k = img_size / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (img_size, hCal))
                hGap = math.ceil((img_size - hCal) / 2)
                imgWhite[hGap : hCal+hGap, :] = imgResize

            img_array = np.expand_dims(imgWhite / 255.0, axis=0)
            prediction = model.predict(img_array, verbose=False)

            predicted_class = np.argmax(prediction, axis=1)[0]
            char = categories[predicted_class]
            prob = round(float(np.max(prediction)), 2)

            if char == prev_char:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 2:
                    if char == 'SPACE':
                        char = "  "
                    string += char
                    start_time = None
            else:
                start_time = None

            prev_char = char
            return {'char': char, 'prob': prob, 'text': string}
        else:
            return {'char': '', 'prob': '', 'text': string}

    except Exception as e:
        print(f"Error processing frame: {e}")
        return {'char': '', 'prob': '', 'text': ''}

def clear_memory():
    global string, prev_char, start_time
    string = ""
    prev_char = None
    start_time = None
    gc.collect()
    print("Memory cleared.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)
