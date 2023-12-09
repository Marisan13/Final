These are two tools developed as part of the final project.
---
# 1. EmoDetect - Text Emotion Prediction

# Description 
This is the initial version of an easy-to-use tool that displays emotion prediction results and keywords extracted from text data in CSV files. The application is designed for use within Streamlit.

# Features
It predicts emotion in text data using a machine learning model (logistic regression) or [pre-trained deep learning model](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion).

# Installation 
1. Clone the repository:
```git clone https://github.com/yourusername/Final.git```
2. Navigate to the project directory:
```cd Final/EmoDetect```
3. Install dependencies:
```pip install -r requirements.txt```

# Usage
1. Run the app file in Streamlit (choose from available app_.py files)::
```streamlit run filename.py```
2. Upload a CSV file (sample files included in the directory):
Ensure the CSV file contains a column with text data (named "text") for emotion analysis.
3. Interpreting the results:
Emotion prediction results will be displayed, and a word cloud showcasing keywords will be generated.
---
# 2. StyleWard - Clothing Image Classification

# Description 
This is the initial version of a Streamlit application that classifies clothing images.

# Features
- It performs image classification using a machine learning model. 
- The application is based on this [model](https://github.com/snova301/fashion_mnist) instead of the one in the EDA file, as the latter's prediction accuracy was quite low.
- Both models have been trained with the [fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset.

# Installation 
1. Clone the repository:
```git clone https://github.com/yourusername/Final.git```
2. Navigate to the project directory:
```cd Final/StyleWard```
3. Install dependencies:
```pip install -r requirements.txt```

# Usage
1. Run app file in Streamlit:
```streamlit run app.py```
2. Upload a JPEG or PNG file (sample files included in the directory):
The file should contain a clear image of clothing with a white or light-colored background and without a model.
3. Interpreting the results:
Classification results will be displayed for the uploaded image.

# Others
Fashion MNIST dataset is not included in the directory. Refer to [here](https://www.tensorflow.org/datasets/catalog/fashion_mnist) for downloading.
