import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained DenseNet201 model
model = DenseNet201(weights='imagenet')

# Streamlit app title
st.title("Pre-trained DenseNet201 Model")

# Sidebar for uploading images
st.sidebar.title("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Make a prediction using the model
    prediction = model.predict(img_array)
    decoded_prediction = decode_predictions(prediction, top=3)[0]

    # Display the top-3 predictions
    st.write("Top 3 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_prediction):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    # Display the prediction probabilities as a bar graph
    st.write("Prediction Probabilities:")
    fig, ax = plt.subplots()
    ax.bar([x[1] for x in decoded_prediction], [x[2] for x in decoded_prediction])
    st.pyplot(fig)
