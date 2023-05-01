from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import streamlit as st
import torch
from PIL import Image
from bokeh.models.widgets import Button
from bokeh.models import CustomJS

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
frame_placeholder = st.empty()

# Add a "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

show_isl = st.button("Show ISL Sign")

device = torch.device('cpu')

model = YOLO("F:/Suhas Sem 6 Github/Sign-Trans/yolov8/asl to text/aslbest.pt")
# !yolo task=detect mode=predict model=/content/American-Sign-Language-Letters-1/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
              "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
              ]

# prev_frame_time = 0
# new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            frame_placeholder.image(frame, channels="RGB")

            if show_isl:
                if classNames[cls] == "W":
                    result = Image.open(r"F:\Suhas Sem 6 Github\Sign-Trans\New folder\j.jpg")
                    st.image(result, caption='W',  use_column_width=True )
                    time.sleep(3)

                    text = "W"

                    tts_button = Button(label="Speak", width=100)

                    tts_button.js_on_event("button_click", CustomJS(code=f"""
                        var u = new SpeechSynthesisUtterance();
                        u.text = "{text}";
                        u.lang = 'en-US';

                        speechSynthesis.speak(u);
                        """))

                    st.bokeh_chart(tts_button)

                else:
                    st.write("helo")
                    time.sleep(3)
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break
    # fps = 1 / (new_frame_time - prev_frame_time)
    # prev_frame_time = new_frame_time
    # print(fps)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)
