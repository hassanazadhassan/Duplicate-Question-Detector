# STREAMLIT UI
import streamlit as st
import pickle
import time
from helper import query_point_creator

# Load the model once
model = pickle.load(open('model.pkl', 'rb'))

# UI Setup
st.set_page_config(page_title="Duplicate Question Detector", layout="centered")

st.markdown("""
    <style>
    .stApp { border: 2px solid #e9ecef; border-radius: 15px; padding: 20px; }
    div.stButton > button { background-color: #4F46E5; color: white; width: 100%; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Duplicate Question Detector")

tab1, tab2 = st.tabs(["Analyze", "How it works"])

with tab1:
    col1, col2 = st.columns(2)
    q1 = col1.text_area("Question 1", height=120)
    q2 = col2.text_area("Question 2", height=120)

    if st.button("Check Similarity"):
        if not q1.strip() or not q2.strip():
            st.warning("Please enter both questions.")
        else:
            with st.spinner('Analyzing...'):
                query = query_point_creator(q1, q2)
                result = model.predict(query)[0]
                if result:
                    st.success("These are duplicate questions!")
                    st.balloons()
                else:
                    st.error("These are distinct questions.")

with tab2:
    st.info("""
    How this works
    This tool doesn't just look for matching words. It uses a **Random Forest Classifier** trained on several sophisticated feature sets:

    * **Token Features:** Analyzes the ratio of shared words and common starting/ending tokens.
    * **Fuzzy Matching:** Uses Levenshtein distance and ratio scoring to identify questions that are phrased differently but mean the same thing.
    * **Length Features:** Compares the structure and length of the questions.
    * **Bag of Words (CountVectorizer):** Captures the core vocabulary overlap between the two inputs.

    By combining these, the model can detect semantic similarity even when the questions share few exact words.
    """)

#     By combining these, the model can detect semantic similarity even when the questions share few exact words.
#     """)
