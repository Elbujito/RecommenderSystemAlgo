import pandas as pd
import numpy as np
import scipy.sparse as sps

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

