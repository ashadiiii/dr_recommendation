import streamlit as st
import pickle
from drug_recom import review_to_words,top_drugs_extractor
import pandas as pd

MODEL_PATH = 'model/trained_model.sav'
TOKENISER_PATH = 'vectorizer/trained_vect.sav'
DATA_PATH = 'data/drugsComTrain_raw.tsv'

model = pickle.load(open(MODEL_PATH,'rb'))

with open(TOKENISER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

st.title('Drug recommendation system')
description= st.text_input("Description about your symptomps",key="descr")

if description:
    # Preprocess the input description
    processed_input = review_to_words(description)
    
    # Convert the single preprocessed string to a list with one element
    input_list = [processed_input]
    
    # Transform the input using the vectorizer
    tfidf_input = vectorizer.transform(input_list)
    
    # Predict the medical condition
    condition = model.predict(tfidf_input)
    
    # Display the predicted condition
    st.write("Predicted Condition:", condition[0])

    df = pd.read_csv(DATA_PATH,sep='\t')
    medication = top_drugs_extractor(condition[0],df)

    st.write("Recommended medication: ", medication)

else:
    st.write("Please enter your symptoms to get a recommendation.")


