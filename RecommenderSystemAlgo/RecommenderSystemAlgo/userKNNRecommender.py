import pandas as pd
import numpy as np
import scipy.sparse as sps

class UserKNN_CF_Recommender():

    """ UserKNN recommender with cosine similarity and no shrinkage"""
    def __init__(self, URM, similarity, playlists, k=100):
        self.dataset = URM
        self.k = k
        self.target_playlists = playlists
        self.similarity = similarity
        self.scores = None

    def fit(self, X, object_itemKNN):
        M, N = X.shape
        Xt = X.T.tocsr()
        # fit a ItemKNNRecommender on the transposed X matrix
        Xt = object_itemKNN.fit(Xt)

        # precompute the predicted scores for speeds
        self.scores = Xt.dot(X)
       

    def recommend(self, user_id, n=None, exclude_seen=True):
        ranking = self.scores[user_id].argsort()[::-1]
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        ranking = ranking[unseen_mask]

        return ranking[:n]
