
##Prediction model used - Logistic Regression (file "text_model_lr.pkl")

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import altair as alt
import base64

import joblib
from PIL import Image
from io import BytesIO

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

import text_hammer as th
import re

st.set_page_config(page_title="Text Emotion Prediction",
                   page_icon="😊")

# Import models
model = joblib.load(open("text_model_lr.pkl", "rb"))
vectorizer = joblib.load('vectorizer.pkl')


# Defining labels 
emotions = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}

emotions_emoji_dict = {"anger": "😠", "fear": "😨😱", "joy": "😝", "love": "🥰", "sad": "😔", "surprise": "😮"}


# Defining necessary functions

def text_preprocessing(dataset, colname):
    try:
        dataset[colname] = dataset[colname].progress_apply(lambda x: str(x).lower())
        dataset[colname] = dataset[colname].progress_apply(lambda x: th.cont_exp(x))
        dataset[colname] = dataset[colname].progress_apply(lambda x: th.remove_emails(x))
        dataset[colname] = dataset[colname].progress_apply(lambda x: re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', " ", x))
        dataset[colname] = dataset[colname].progress_apply(lambda x: th.remove_html_tags(x))
        dataset[colname] = dataset[colname].progress_apply(lambda x: th.remove_special_chars(x))
        dataset[colname] = dataset[colname].progress_apply(lambda x: th.remove_accented_chars(x))        
        return dataset
    except ValueError as ve:
        print(f"Exception during text preprocessing {ve}")


def get_prediction_proba(docx):
    results = model.predict_proba([docx])
    results = np.array(results)
    results = [emotions[label] for label in results]
    st.write(results)
    return results


def load_and_predict(df, model):
    df = text_preprocessing(df, "text")
    text_vectorized = vectorizer.transform(df["text"])
    # Get predicted labels from the model
    predicted_labels = model.predict(text_vectorized)  
    # Map numerical labels to emotion names
    predicted_emotions = [emotions[label] for label in predicted_labels]   
    df['Predicted Emotion'] = predicted_emotions
    return df


def get_csv_download_link(df):
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_emotions.csv">Download Updated CSV File</a>'
    return href



def main():
    st.title("EmoDetect")
    st.subheader("Predicting Emotions in Text")

    image = Image.open('Lego-image.png')
    st.image(image, caption='', width=700)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


    if uploaded_file is not None:
     
        df = pd.read_csv(uploaded_file)

        # Predict emotions and update CSV
        df_with_predictions = load_and_predict(df, model)

        # Show emotion distribution
        emotion_distribution = df_with_predictions['Predicted Emotion'].value_counts(normalize=True) * 100
        emotion_distribution_sorted = emotion_distribution.reindex(emotions.values())

        chart = alt.Chart(emotion_distribution_sorted.reset_index()).mark_bar().encode(
            x=alt.X('index:N', axis=alt.Axis(title="Emotion")),
            y='Predicted Emotion',
            color=alt.Color('index', scale=alt.Scale(range=['red', 'blue', 'yellow', 'gray', "green", "pink"])),

            # Specify tooltips with emotions as labels and percentage as title
            tooltip=[
                alt.Tooltip('index:N', title='Emotion'),
                alt.Tooltip('Predicted Emotion:Q', title='Percentage', format='.2f')
            ]).properties(
                width=alt.Step(80)
            ).interactive()

        # Process the text column to create a single string for word cloud generation
        text = ' '.join(df['text'].dropna())

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        col1, col2 = st.columns(2)

        with col1:
            st.altair_chart(chart, use_container_width=True)
            csv_download_link = get_csv_download_link(df_with_predictions)
            st.markdown(csv_download_link, unsafe_allow_html=True)

        with col2:
            #st.markdown("### Word Cloud from Comments")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)  # Pass the figure to st.pyplot()


if __name__ == '__main__':
    main()