import pandas as pd
import pickle as pkl

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import data
from config.config import *


def train(trimmed_data):

    stops = set(stopwords.words('english'))

    vectorizer = CountVectorizer(stop_words=stops)
    X_input = vectorizer.fit_transform(trimmed_data["cleaned_text"])

    model = LatentDirichletAllocation(n_components=7, random_state=0)
    model.fit(X_input)

    doc_topic_dist = pd.DataFrame(model.transform(X_input))

    return vectorizer, lda, doc_topic_dist


def save_artifacts(vectorizer, model, doc_topic_dist):

    doc_topic_dist.to_csv(DOC_TOPIC_DIST, index=False)
    pkl.dump(model, open(LDA_MODEL, "wb"))
    pkl.dump(vectorizer, open(VECTORIZER_MODEL, "wb"))


if __name__ == "__main__":
    trimmed_data = data.data_cleaning(filepath=TRAIN_DATA)
    vectorizer, model, doc_topic_dist = train(trimmed_data)
    save_artifacts(vectorizer, model, doc_topic_dist)
