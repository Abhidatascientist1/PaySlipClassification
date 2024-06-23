import re
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import joblib
from collections import Counter
import pandas as pd
from io import BytesIO
import fitz
from PIL import Image
import pytesseract

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = joblib.load(file)

# Load the classifiers
classifiers = {}
for name in ['Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine', 'Gradient Boosting', 'Naive Bayes',
             'Logistic Regression']:
    with open(f'{name.lower().replace(" ", "_")}_classifier.pkl', 'rb') as file:
        classifiers[name] = joblib.load(file)


# Prediction function
def predict_remarks(components):
    input_tfidf = tfidf.transform([components])

    # Collect predictions from all models
    all_predictions = []
    for model_name, model in classifiers.items():
        prediction = model.predict(input_tfidf)
        all_predictions.append(prediction[0])

    # Determine the most common prediction
    most_common_prediction = Counter(all_predictions).most_common(1)[0][0]
    return most_common_prediction


# Streamlit app
st.title('Payslip Component Classification')

st.write('Enter the payslip components to predict the remark:')

components = st.text_area('Payslip Components', '', key='payslip_components')

if st.button('Predict Remark'):
    if components:
        predicted_remark = predict_remarks(components)
        st.write(f'Predicted Remark: {predicted_remark}')
    else:
        st.write('Please enter some components to predict.')


# Optionally, you can add file upload functionality for document prediction
def extract_text_from_pdf(pdf_bytes):
    text = ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def extract_text_from_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return text


# Function to predict from uploaded document
def predict_from_document(uploaded_file):
    extracted_text = ""

    # Read the uploaded file as bytes
    file_bytes = uploaded_file.read()

    # Determine file type and extract text accordingly
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    elif uploaded_file.type in ["image/jpeg", "image/png"]:
        extracted_text = extract_text_from_image(file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or an image file.")

    # Split the extracted text into individual components and filter out numeric values
    components_list = re.findall(r'[A-Za-z, ]+', extracted_text)

    # Assuming employer_name and abn are not extracted and are empty
    employer_name = ""
    abn = ""

    predictions = []
    for component in components_list:
        predicted_remark = predict_remarks(component)
        predictions.append({'Component': component.strip(), 'Predicted Component Type': predicted_remark})

    return pd.DataFrame(predictions)

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "jpg", "jpeg", "png"], key='file_uploader')

if uploaded_file is not None:
    document_predictions = predict_from_document(uploaded_file)
    st.write(document_predictions)