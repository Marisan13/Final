
##Prediction model used - https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion 

import streamlit as st
import pandas as pd
import altair as alt
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline


st.set_page_config(page_title="Text Emotion Prediction",
                   page_icon="😊")

classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

emotions = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}

emotions_emoji_dict = {"anger": "😠", "fear": "😨😱", "joy": "😝", "love": "🥰", "sad": "😔", "surprise": "😮"}


def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_emotions.csv">Download Updated CSV File</a>'
    return href


def main():
    st.title("EmoDetect")
    st.subheader("Predicting Emotions in Text")

    image = Image.open('Lego-image.png')
    st.image(image, caption='', width=700)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
     
        # Display uploaded file
        df = pd.read_csv(uploaded_file)

        predict_sentences = df["text"].tolist()
        # Empty list to store predicted labels
        predicted_labels = []
        
        # Loop through each sentence for prediction
        for i in predict_sentences:
            prediction = classifier(i)
            # Extracting the list of label-score dictionaries
            scores = prediction[0]
            # Get the label with the highest score
            label_with_highest_score = max(scores, key=lambda x: x['score'])['label']
            # Append the label to the list
            predicted_labels.append(label_with_highest_score)
            #st.write(predicted_labels)

        df['Predicted Emotion'] = predicted_labels
        st.write(df)

        # Show emotion distribution
        emotion_distribution = df['Predicted Emotion'].value_counts(normalize=True) * 100
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

        # Display the bar chart and word cloud side by side
        col1, col2 = st.columns(2)

        with col1:
            st.altair_chart(chart, use_container_width=True)
            csv_download_link = get_csv_download_link(df)
            st.markdown(csv_download_link, unsafe_allow_html=True)

        with col2:
            #st.markdown("### Word Cloud from Comments")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)  # Pass the figure to st.pyplot()


if __name__ == '__main__':
    main()