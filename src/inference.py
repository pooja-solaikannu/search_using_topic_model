import pandas as pd
import pickle as pkl

from scipy.spatial.distance import jensenshannon

def load_artifacts(cv_path, model_path, topic_dist_path):
    count_vec = pkl.load(open(cv_path, "rb"))
    model = pkl.load(open(model_path, "rb"))
    topic_dist = pd.read_csv(topic_dist_path)

    return count_vec, model, topic_dist

def prediction(doc_dist, topic_dist, k):
    distances = topic_dist.apply(lambda x: jensenshannon(x, doc_dist), axis=1)
    k_nearest = distances[distances != 0].nsmallest(n=k).index
    k_distances = distances[distances != 0].nsmallest(n=k)

    return k_nearest, k_distances
