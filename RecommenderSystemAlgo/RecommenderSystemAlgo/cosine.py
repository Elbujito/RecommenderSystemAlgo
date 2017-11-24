import scipy.sparse as sps

def compute_cosine(URM_all):
    #compute the cosine
    print("Compute cosine")

    similarity = URM_all.T * URM_all

    return URM_all.tocsr(), similarity