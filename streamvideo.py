import streamlit as st
import cv2
import numpy as np
import requests

# Title of the app
st.title( 'Mobile Camera Preview ' )

# An empty image container that will hold the frames we'll fetch from the server
frame_window = st.image( [] )

take_picture_button = st.button( 'Take Picture' )
while True:
    # Request the image from the server
    response = requests.get(url="http://192.168.1.7:8080//photo.jpg")         //Add the url in your IP Webcam app on phone
    imgNp = np.array(bytearray(response.content), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, cv2.IMREAD_UNCHANGED ) 
    # As OpenCV decodes images in BGR format, we'd convert it to the RGB format
    frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )

    frame_window.image(frame)

    if take_picture_button:
        # Pass the frame to a model
        # And show the output here...
        break
