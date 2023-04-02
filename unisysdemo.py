import tensorflow as tf

# Load the model from the h5 file
model = tf.keras.models.load_model('isl-to-text.h5')

import streamlit as st
from PIL import Image
import numpy as np
import cv2
# Define the Streamlit app
arr1 = ['1', '2' , '3', '4' , '5' , '6' , '7' ,'8' ,'9' ,'A' ,'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I' , 'J', 'K' ,'L' ,'M' ,'N' ,'O' ,'P' ,'Q' ,'R' ,'S' ,'T' ,'U' ,'V' ,'W' ,'X' ,'Y' ,'Z']
    # Add a title to the app
st.title('My Streamlit App')
    
    # Allow the user to upload an image
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    
    # If an image is uploaded
if uploaded_file is not None:
        # Read the image file
    img = Image.open(uploaded_file)
        
        # Display the image to the user
    st.image(img, caption='Uploaded Image', use_column_width=True)
        
    # img = cv2.resize(img, (256, 256))
        # Preprocess the image
    # img = img.resize((256, 256))
    # img_array = np.asarray(img) 
    # img_array = np.expand_dims(img_array, axis=0)
        
        # Use the loaded model to generate output
    
    img_gray = img.convert('L')

    # Resize the image to (256, 256)
    img_resized = img_gray.resize((256, 256))

    # Convert the image to a NumPy array
    img_array = np.array(img_resized)

    # Add an extra dimension to the array to indicate that it has a single channel
    img_array = np.expand_dims(img_array, axis=-1)

    # Reshape the array to match the expected input shape of the model
    img_array = img_array.reshape((1, 256, 256, 1))
        
    output = model.predict(img_array)

    index = np.where(output==1)[1][0]   
   

    # output_dict = {}
    # for row in output:
    #     key = row[0]
    #     value = row[1]
    #     output_dict[key] = value

    # # Function to extract the key of a given value in a dictionary
    # def get_key(output_dict, value):
    #     for k, v in output_dict.items():
    #         if v == value:
    #             return k
    #     return None

    # # Example usage
    # key = get_key(output_dict, 2)
    
    # Display the output to the user
    st.write('Output:', arr1[index] )
    # st.write('Output:', output )
