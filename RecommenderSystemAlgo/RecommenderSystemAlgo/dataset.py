import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy

def create_icm(itemTagsDF,tags_df_idx, tracks_df_idx):
    tracks = np.array(itemTagsDF[tracks_df_idx])
    tags = np.array(itemTagsDF[tags_df_idx])
    ones = np.ones(len(tags))

    ICM_all = sps.coo_matrix((ones, (tags,tracks)))
    #ICM_all = ICM_all.tocsr()

    return ICM_all

def create_urm(userItemDF, userID, itemID, weightID, split=0.5):
    # get playlists, tracks and interactions
    playlists = np.array(userItemDF[userID])
    tracks = np.array(userItemDF[itemID])
    interactions = np.array(userItemDF[weightID])    
    
    # compress data using cscr matrix and pivot
    URM_all = sps.coo_matrix((interactions, (playlists, tracks)), dtype=np.float32)
    URM_all.tocsr()

    #Create Split Data Set
    numInteractions = URM_all.nnz
    train_mask = np.random.choice([True,False], numInteractions, p=[split, 1-split])
    
    URM_train = sps.coo_matrix((interactions[train_mask], (playlists[train_mask], tracks[train_mask])), dtype=np.float32)
    URM_train = URM_train.tocsr() 

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((interactions[test_mask], (playlists[test_mask], tracks[test_mask])), dtype=np.float32)
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

    similarity.data= np.dot(co_counts.data,similarity.data)

    return similarity.tocsr()

def compute(X, shrinkage=10):

    Xsq = X.copy()
    Xsq.data **= 2
    norm = np.sqrt(Xsq.sum(axis=0))
    norm = np.asarray(norm).ravel()
    norm += 1e-6
    # compute the number of non-zeros in each column
    # NOTE: this works only if X is instance of sparse.csc_matrix
    col_nnz = np.diff(X.indptr)
    # then normalize the values in each column
    X.data /= np.repeat(norm, col_nnz)
    print("Normalized")

    # 2) compute the cosine similarity using the dot-product
    dist = X.T * X
    print("Computed")
        
    # zero out diagonal values
    dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
    print("Removed diagonal")
    
    return dist.tocsr()

def apply_shrinkage(X, dist, shrinkage=10):
    # create an "indicator" version of X (i.e. replace values in X with ones)
    X_ind = X.copy()
    X_ind.data = np.ones_like(X_ind.data)
    # compute the co-rated counts
    co_counts = X_ind.T.dot(X_ind)
    # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
    # then multiply dist with it
    dist *= co_counts / (co_counts + shrinkage)

    return dist