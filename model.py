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

def speak(text):
    # Convert the text to speech using gTTS
    tts = gTTS(text=text, lang='en', slow=False)
    
    # Save the speech as an MP3 file
    speech_file = "output_speech.mp3"
    tts.save(speech_file)
    
    # Play the audio in the Streamlit app
    audio_file = open(speech_file, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

    # Clean up the saved file
    os.remove(speech_file)

# Streamlit UI setup
st.title("Chest X-ray Pneumonia Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Load the model
    model = load_model('chest_xray.h5') 
    img_file = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img_file)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    
    # Display the prediction result
    if classes[0][0] > 0.5:
        result_message = "Result: Normal"
        st.success(result_message)
    else:
        result_message = "Result: Affected by PNEUMONIA"
        st.error(result_message)

    # Speak the result
    speak(result_message)
