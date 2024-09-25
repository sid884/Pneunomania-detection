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

    # Attempt to play audio; handle cases where there's no audio device
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(speech_file)
        pygame.mixer.music.play()

        # Wait until the audio finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Stop and unload the music
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()  # This releases the file so it can be deleted
    except Exception as e:
        print(f"Error playing sound: {e}. Audio playback skipped.")

    # Remove the MP3 file after playback
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
