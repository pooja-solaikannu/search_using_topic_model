import streamlit as st
import pandas as pd

from src.inference import *
from src.data import *

removals = ["url", "Soil pH Preferences", "Inflorescence Height", "Plant Spread", "Awards and Recognitions", "Parentage",
            "Child plants", "Plant Height", "Minimum cold hardiness", "Toxicity", "Bloom Size"]

filepath = "data/flowers_desc_dataset.xlsx"
topic_dist_path = "data/doc_topic_dist.csv"
lda_model_path = "data/models/lda.pkl"
cv_model_path = "data/models/count_vec.pkl"

k=5

def op_transformation(sorted_op):
    related_flowers = {}
    for i in range(len(op["combined_text"])):
        related_flowers[i] = {"similar_flower": sorted_op["combined_text"].iloc[i].strip(), "similarity": round(sorted_op["k_dist"].iloc[i], 3)}

    return related_flowers

# Title
st.title("Topic Model based Searching System")
st.info("🔍 Explore the available searching system, Thank you.")

st.header("Data")
df = data_cleaning(filepath=filepath)
st.text(f"Rows : {df.shape[0]}, Columns : {df.shape[1]}")
st.write(df.head(5))


count_vec, model, topic_dist = load_artifacts(cv_model_path, lda_model_path, topic_dist_path)
st.header("Searching")
query = st.text_input(label="", placeholder="Edible flowers for bees")
if query:
    # get_model_results -> create src folder with model training and prediction script
    # transformed to the output sepcific format
    doc_dist = pd.Series(model.transform(count_vec.transform([query]))[0])
    k_nearest, k_distances = prediction(doc_dist, topic_dist, k)
    op = df[df["index"].isin(k_nearest)]
    op["k_dist"] = 1 - k_distances
    sorted_op = op.sort_values(by="k_dist", axis=0, ascending=False)
    related_flowers = op_transformation(sorted_op)
    st.write(related_flowers)
    # st.write({0: "val1", 1: "val2", 2: "val3", 3: "val4"})