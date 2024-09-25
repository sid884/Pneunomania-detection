import warnings
from PIL import Image
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from gtts import gTTS
import os
import pygame

# Initialize pygame mixer to play the audio
pygame.mixer.init()

def speak(text):
    # Convert the text to speech using gTTS
    tts = gTTS(text=text, lang='en', slow=False)
    
    # Save the speech as an MP3 file
    speech_file = "output_speech.mp3"
    tts.save(speech_file)
    
    # Load and play the speech
    pygame.mixer.music.load(speech_file)
    pygame.mixer.music.play()
    
    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Stop and unload the music, then remove the MP3 file
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()  # This releases the file so it can be deleted
    
    # Remove the MP3 file after playback
    os.remove(speech_file)

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_pneumonia(model, img_path):
    img_file = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img_file)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    prediction = model.predict(img_data)
    return prediction

# Streamlit UI

# Set the page title and header
st.set_page_config(page_title="PNEUMONIA Detection App", layout="centered")
st.title("Chest X-ray PNEUMONIA Detection")

# Upload Image Section
uploaded_image = st.file_uploader("Upload Chest X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image is not None:
    # Display uploaded image
    img = load_image(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Load the pre-trained model
    model = load_model('chest_xray.h5')

    # Predict result on button click
    if st.button("Predict"):
        result = predict_pneumonia(model, "temp_image.png")
        
        # Check the prediction and display result
        if result[0][0] > 0.5:
            st.success("Result: Normal")
            speak("Result is Normal")
        else:
            st.error("Result: Affected by PNEUMONIA")
            speak("Affected by PNEUMONIA")
