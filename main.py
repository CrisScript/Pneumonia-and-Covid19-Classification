import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import json

#Load classifier model 
model = load_model('model_v4.h5')

class_names = json.load(open("class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_names):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Check if class_indices is a list or a dictionary
    if isinstance(class_names, list):
        # If it's a list, access it directly with an integer index
        predicted_class_name = class_names[predicted_class_index]
    else:
        # If it's a dictionary, use string keys
        predicted_class_name = class_names[str(predicted_class_index)]
    
    return predicted_class_name


# Streamlit App
st.title('⚕️Clasificación de Neumonia y Covid19')

uploaded_image = st.file_uploader("Por favor, sube una radiografia de tórax", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Clasificar'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_names)
            st.success(f'Predicción: {str(prediction)}')
