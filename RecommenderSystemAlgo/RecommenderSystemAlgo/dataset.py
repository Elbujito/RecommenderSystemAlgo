import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy

def create_urm(userItemDF, userID, itemID, weightID, split=0.5):
    # get playlists, tracks and interactions
    playlists = np.array(userItemDF[userID])
    tracks = np.array(userItemDF[itemID])
    interactions = np.array(userItemDF[weightID])    
    
    # compress data using cscr matrix and pivot
    URM_all = sps.coo_matrix((interactions, (playlists, tracks)), dtype=np.float16)
    URM_all.tocsr()

    #Create Split Data Set
    numInteractions = URM_all.nnz
    train_mask = np.random.choice([True,False], numInteractions, p=[split, 1-split])

    URM_train = sps.coo_matrix((interactions[train_mask], (playlists[train_mask], tracks[train_mask])), dtype=np.float16)
    URM_train = URM_train.tocsr() 

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((interactions[test_mask], (playlists[test_mask], tracks[test_mask])), dtype=np.float16)
    URM_test = URM_test.tocsr()

    return URM_all, URM_train, URM_test

def compute_cosine(URM_all):
    #compute the cosine
    print("Compute cosine")

    similarity = URM_all.T * URM_all

    return URM_all.tocsr(), similarity

def compute_cosine_shrinkage(ICM, shrinkage=10):

    similarity = ICM.T * ICM

    # and apply the shrinkage
    if shrinkage > 0:
        similarity = apply_shrinkage(ICM, similarity)
          
        
    return dist

def apply_shrinkage(ICM, similarity, shrinkage=10):
    print("Apply shrinkage")  
    # create an "indicator" version of X (i.e. replace values in X with ones)
    ICM_ind = ICM.copy()
    ICM_ind.data = np.ones_like(ICM_ind.data)
    # compute the co-rated counts
    co_counts = ICM_ind * ICM_ind.T
    # remove the diagonal
    co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
    # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
    # then multiply dist with it
    co_counts_shrink = co_counts.copy()
    co_counts_shrink.data += shrinkage
    co_counts.data /= co_counts_shrink.data
    similarity.data *= co_counts.data

    return similarity.tocsr()