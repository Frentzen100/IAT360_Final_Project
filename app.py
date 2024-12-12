import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model, mean, and std
model = tf.keras.models.load_model('Model/final_model.h5')
mean = np.load('Model/mean.npy')
std = np.load('Model/std.npy')

# Label dictionary for the classes with full details
label_dict = {
    0: ('Actinic keratoses and intraepithelial carcinomae', 
        'Actinic keratoses (AK) are precancerous growths on sun-damaged skin. They are rough, dry, scaly patches that are red, pink, or brown. If left untreated, they can progress to squamous cell carcinoma.',
        'Typically found on sun-exposed areas, these lesions appear as dry, scaly patches.',
        'Caused by prolonged sun exposure or UV damage over time.',
        'Advise the patient to do preventive measures, such as sun protection and regular skin checks.'),
    1: ('Basal cell carcinoma', 
        'Basal cell carcinoma (BCC) is the most common type of skin cancer. It often appears as a shiny bump or a scaly patch on sun-exposed areas of the skin.',
        'Characteristics include a shiny or waxy bump or a scaly patch with visible blood vessels.',
        'Caused by prolonged UV exposure from the sun or tanning beds.',
        'Advise the patient to do biopsy or removal.'),
    2: ('Benign keratosis-like lesions', 
        'Benign keratosis-like lesions are often harmless growths that resemble skin cancer. These growths can be waxy or scaly and appear on sun-damaged skin.',
        'They typically appear as small, scaly, or raised lesions on sun-exposed areas.',
        'These are usually caused by aging or sun exposure.',
        'Advise the patient to do excision or cryotherapy.'),
    3: ('Dermatofibroma', 
        'Dermatofibromas are benign skin growths that often appear as small, firm nodules that are brown or red in color. They are typically non-cancerous.',
        'They often appear as firm, brown or red nodules on the skin.',
        'The exact cause is unknown, but they may develop after an injury or trauma to the skin.',
        'Advise the patient to monitor the lesion for significant changes in size, color, or texture and option to do an excision'),
    4: ('Melanocytic nevi', 
        'Melanocytic nevi, also known as moles, are generally harmless. They can vary in size and color but are typically round or oval with a smooth border.',
        'Moles are typically round or oval with a smooth border, and they can vary in color.',
        'Genetic factors, sun exposure, and hormonal changes can contribute to mole development.',
        'Encourage regular skin self-examinations, periodic professional skin checks and biopsy.'),
    5: ('Pyogenic granulomas and hemorrhage', 
        'Pyogenic granulomas are rapidly growing, red, and sometimes bleeding lesions that can appear after an injury. They are typically non-cancerous.',
        'They are often red, raised, and may bleed when irritated.',
        'They are caused by trauma or injury, often to the skin or mucous membranes.',
        'Advise the patient to consider treatment options such as curettage and cauterization, laser therapy, and surgical excision'),
    6: ('Melanoma', 
        'Melanoma is a serious form of skin cancer that often looks like a mole or skin lesion. It can have irregular borders, different colors, and asymmetry.',
        'Melanoma often has irregular borders, multiple colors, and can be asymmetrical.',
        'It is mainly caused by UV exposure and genetic factors.',
        'Urgent, this lesion need to be treated immediately! Consider to perform diagnostic procedures, such as a biopsy or referral to an oncologist')
}

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image has 3 channels
    image = image.resize((28, 28))  # Resize image to 28x28
    image = np.asarray(image)
    image = (image - mean) / std  # Standardize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make prediction
def predict_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image, verbose=0)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_name, description, characteristics, causes, recommendation = label_dict[predicted_class_idx]
    confidence = np.max(predictions)
    return predicted_class_name, description, characteristics, causes, recommendation, confidence

# Add custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .sub-title {
            font-size: 24px;
            font-weight: bold;
            color: #FF6347;
        }
        .description {
            font-size: 18px;
            color: #555555;
        }
        .recommendation {
            font-size: 16px;
            color: #FF6347;
            font-weight: bold;
        }
        .disclaimer {
            background-color: #fff3cd;
            border-left: 6px solid #ffeeba;
            padding: 15px;
            margin-top: 15px;
            margin-bottom: 15px;
            font-size: 16px;
            color: #856404;
            font-weight: bold;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app interface
st.markdown('<h1 class="main-title">Skin Lesion Classification</h1>', unsafe_allow_html=True)
st.write("Upload a dermoscopic skin lesion image to predict its type and receive personalized recommendations")

# Add disclaimer
st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> The results provided by this application are based on machine learning predictions and may not be accurate. 
        For an accurate diagnosis and appropriate treatment, it is essential to consult a licensed dermatologist.
    </div>
""", unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_image is not None:
    # Display image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Make prediction
    with st.spinner('Classifying...'):
        predicted_class_name, description, characteristics, causes, recommendation, confidence = predict_image(uploaded_image)
    
    # Display result with sections for each part
    st.markdown(f"### **Predicted Lesion Type: {predicted_class_name}**", unsafe_allow_html=True)
    confidence_percentage = confidence * 100
    st.markdown(f"**Confidence**: {confidence_percentage:.2f}%", unsafe_allow_html=True)
    
    # Display detailed information
    st.markdown(f"### **Definition:**", unsafe_allow_html=True)
    st.markdown(f"{description}", unsafe_allow_html=True)
    
    st.markdown(f"### **Characteristics:**", unsafe_allow_html=True)
    st.markdown(f"{characteristics}", unsafe_allow_html=True)
    
    st.markdown(f"### **Causes:**", unsafe_allow_html=True)
    st.markdown(f"{causes}", unsafe_allow_html=True)
    
    st.markdown(f"### **Recommendation:**", unsafe_allow_html=True)
    st.markdown(f"{recommendation}", unsafe_allow_html=True)
