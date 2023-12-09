import keras
from tensorflow.keras.models import load_model
import h5py

import streamlit as st
import numpy as np
import pandas as pd

import altair as alt
import base64
from PIL import Image
from keras.preprocessing.image import img_to_array


# Load the saved model
with h5py.File('pages/model_2nd.hdf5', 'r', libver='latest', driver='core') as f:
    model = load_model(f, compile=False)

# Streamlit app
st.title('StyleWard')
st.subheader("Classifying Clothing Items in Your Wardrobe")
image = Image.open('pages/clothes-image.png')
st.image(image, caption='', width=700)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create a dictionary to hold counts of clothes by category
clothes_count = {
    'T-shirt/top': 0,
    'Trouser': 0,
    'Pullover': 0,
    'Dress': 0,
    'Coat': 0,
    'Sandal': 0,
    'Shirt': 0,
    'Sneaker': 0,
    'Bag': 0,
    'Ankle boot': 0
}

# Mapping of category names to emojis
emoji_mapping = {
    'T-shirt/top': '👕',
    'Trouser': '👖',
    'Pullover': '🧥',
    'Dress': '👗',
    'Coat': '🧥',
    'Sandal': '👡',
    'Shirt': '👔',
    'Sneaker': '👟',
    'Bag': '👜',
    'Ankle boot': '👢'
}

# Function to update the count for a predicted category
def update_count(category):
    clothes_count[category] += 1

# Initialize the count dictionary
if 'clothes_count' not in st.session_state:
    clothes_count = {category: 0 for category in class_names}
    st.session_state.clothes_count = clothes_count
else:
    clothes_count = st.session_state.clothes_count

# Initialize the wardrobe dataframe
df = pd.DataFrame(list(clothes_count.items()), columns=['Item', 'Count'])
df['Emoji'] = df['Item'].map(emoji_mapping)

# Display the total count of items before uploading
#st.write(f"Total Items: {df['Count'].sum()}")

uploaded_file = st.file_uploader("Upload your image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Resize and convert image to grayscale
    img_resized = image.resize((28, 28)).convert('L')
    
    # Convert image to array and normalize
    img_array = np.array(img_resized)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0  # Normalize pixel values

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    # Map predicted class index to class name
    class_name = class_names[predicted_class]

    # Display the image and table side by side
    col1, col2 = st.columns([1, 2])

    with col1: 
        st.image(image, width=200) 
    
    with col2:
        st.write(f"Item: {emoji_mapping.get(class_name, '❓')} {class_name}")
        st.write("Your Wardrobe:")
        
        if st.button("Add"):
            # Update the count for the predicted category
            update_count(class_name)
            
            # Update the wardrobe dataframe and total count
            df.loc[df['Item'] == class_name, 'Count'] += 1
            df.loc[df['Item'] == 'Total', 'Count'] = df['Count'].sum()
            
            # Update the session state with the new count
            st.session_state.clothes_count = clothes_count
        
        # Display the updated wardrobe dataframe
        styled_df = df.set_index('Item').style.format({'Emoji': '{:}'}).hide_index()
        st.write(styled_df)
        
        # Display the total count of items
        st.write(f"Total Items: {df['Count'].sum()}")