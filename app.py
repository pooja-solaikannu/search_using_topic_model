import streamlit as st
import pandas as pd

from src.inference import *
from src.data import *
from config.config import *


def op_transformation(sorted_op):
    related_flowers = {}
    for i in range(len(op["combined_text"])):
        related_flowers[i] = {"similar_flower": sorted_op["combined_text"].iloc[i].strip(), "similarity": round(sorted_op["k_dist"].iloc[i], 3)}

    return related_flowers

# Title
st.title("Topic Model based Searching System")
st.info("🔍 Explore the available searching system, Thank you.")

st.header("Data")
df = data_cleaning(filepath=TRAIN_DATA)
st.text(f"Rows : {df.shape[0]}, Columns : {df.shape[1]}")
st.write(df.head(5))


vectorizer, model, doc_topic_dist = load_artifacts(VECTORIZER_MODEL, LDA_MODEL, DOC_TOPIC_DIST)
st.header("Searching")
query = st.text_input(label="", placeholder="Edible flowers for bees")
if query:
    # get_model_results -> create src folder with model training and prediction script
    # transformed to the output sepcific format
    doc_dist = pd.Series(model.transform(vectorizer.transform([query]))[0])
    k_nearest, k_distances = prediction(doc_dist, doc_topic_dist, k)
    op = df[df["index"].isin(k_nearest)]
    op["k_dist"] = 1 - k_distances
    sorted_op = op.sort_values(by="k_dist", axis=0, ascending=False)
    related_flowers = op_transformation(sorted_op)
    st.write(related_flowers)
    # st.write({0: "val1", 1: "val2", 2: "val3", 3: "val4"})
