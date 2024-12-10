import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import load_model

# Define the CNN model architecture
def get_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model

# Load the model architecture
model = get_model()

# Load the pre-trained weights
model.load_weights('Model/best_model.weights.h5')

# Define the classes
classes = {
    4: ('nv', 'Melanocytic nevi'),
    6: ('mel', 'Melanoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    1: ('bcc', 'Basal cell carcinoma'),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'Dermatofibroma')
}

# Streamlit interface
st.title("Skin Disease Detection - HAM10000")
st.write("Upload an image to detect the skin disease category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", channels="BGR", use_column_width=True)

    # Resize the image to the model's input size
    img_resized = cv2.resize(img, (28, 28))

    # Normalize the image
    img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)

    # Predict the class
    img_reshaped = img_normalized.reshape(1, 28, 28, 3)
    result = model.predict(img_reshaped)
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind]

    # Display prediction results
    st.write(f"### Prediction: {class_name[1]}")
    st.write(f"**Confidence:** {max_prob * 100:.2f}%")
