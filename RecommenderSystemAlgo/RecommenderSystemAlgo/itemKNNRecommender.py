import pandas as pd
import numpy as np
import scipy.sparse as sps

class ItemKNN_CF_Recommender(object):
	""" ItemKNN recommender with cosine similarity and no shrinkage"""
	def __init__(self, URM, similarity, playlists, k=100):
		self.dataset = URM
		self.k = k
		self.target_playlists = playlists
		self.similarity = similarity

	def fit(self, URM):
		#get only the n nearestneightbours
		print("Item KNN")
		#csc more faster
		similarity = self.similarity.tocsc() #csc
		for col_index in range(URM.shape[1]):      
			this_item_weights = similarity.data[similarity.indices[col_index]:similarity.indices[col_index+1]]
			nearestNeightbours = np.zeros(len(this_item_weights))
			nearestNeightbours.astype(int)
			top_k_idx = np.argsort(this_item_weights)[-self.k:]
			for ind in top_k_idx:
				nearestNeightbours[ind] = this_item_weights[ind]
			similarity.data[similarity.indices[col_index]:similarity.indices[col_index+1]] = nearestNeightbours

		self.similarity = similarity.tocsr() #csr

	def recommend(self, user_id, at=None):

		# compute the scores using the dot product
		user_profile = self.dataset[user_id]
		scores = user_profile.dot(self.similarity).toarray().ravel()

		# rank items
		ranking = scores.argsort()[::-1]
		seen = user_profile.indices
		unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
		ranking = ranking[unseen_mask]

		return ranking[:at]

	def predic(self,at=5):
		#predic
		print("Predic ")
		#sort top 5 value
		track_final = []  
		playlistCount = len(self.target_playlists)
		for i in range(playlistCount):        
			track_final.append(self.recommend(self.target_playlists[i],at))

		return track_final
