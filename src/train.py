import pandas as pd
import pickle as pkl

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import data

filepath = "data/flowers_desc_dataset.xlsx"
op_filename = "data/doc_topic_dist.csv"
op_model = "data/models/lda.pkl"
cv_model = "data/models/count_vec.pkl"


def train(trimmed_data):

    stops = set(stopwords.words('english'))

    count_vec = CountVectorizer(stop_words=stops)
    X_input = count_vec.fit_transform(trimmed_data["cleaned_text"])

    lda = LatentDirichletAllocation(n_components=15, random_state=0)
    lda.fit(X_input)

    doc_topic_dist = pd.DataFrame(lda.transform(X_input))

    return count_vec, lda, doc_topic_dist


def save_artifacts(count_vec, model, topic_dist):

    topic_dist.to_csv(op_filename, index=False)
    pkl.dump(model, open(op_model, "wb"))
    pkl.dump(count_vec, open(cv_model, "wb"))


if __name__ == "__main__":
    trimmed_data = data.data_cleaning(filepath=filepath)
    count_vec, model, dataset_topic_dist = train(trimmed_data)
    save_artifacts(count_vec, model, dataset_topic_dist)
