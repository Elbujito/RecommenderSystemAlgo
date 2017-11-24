import pandas as pd
import numpy as np
import re
import scipy.sparse as sps
import math

def read_file(path):

    str(path)

    #extract csv file to df
    object_df = pd.read_csv(path, sep='\t')

    return object_df

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

def create_mapTable_ID ( dataframe, columnID ):
    items = dataframe[columnID].drop_duplicates().reset_index(drop=True)
    corresp_df = pd.DataFrame.from_dict(items)
    corresp_df['new_id'] = corresp_df.index.values
    
    return corresp_df

def extract_tags_list( dataframe, tagsColumn, idColumn):
    items=[]
    tags=[]

    for i in range(len(dataframe.index)):
        numbers=re.findall(r'\d+', dataframe.iloc[i][tagsColumn]) # TODO add when title empty or = none
        for x in numbers:
            items.append(dataframe.iloc[i][idColumn])
            tags.append(x)
   
    result= pd.DataFrame( {idColumn: items, tagsColumn: tags } )
    return result

def define_tags_occurence ( userItemDF, userID, itemID, tagID ):
    # tag column must be in userItemDF
    TF= userItemDF.groupby([itemID,tagID], as_index = False, sort = False).count().rename(columns = {userID: 'tag_count_TF'})[[itemID,tagID,'tag_count_TF']]
    Tag_distinct = userItemDF[[tagID,itemID]].drop_duplicates()
    DF =Tag_distinct.groupby([tagID], as_index = False, sort = False).count().rename(columns = {itemID: 'tag_count_DF'})[[tagID,'tag_count_DF']]
    a=math.log10(len(np.unique(userItemDF[itemID])))
    DF['IDF']=a-np.log10(DF['tag_count_DF'])
    TF = pd.merge(TF,DF,on = tagID, how = 'left', sort = False)
    TF['TF-IDF']=TF['tag_count_TF']*DF['IDF']
    
    TF['TF-IDF']=TF['TF-IDF'].fillna(0)
    
    Vect_len = TF[[itemID,'TF-IDF']].copy()
    Vect_len['TF-IDF-Sq'] = Vect_len['TF-IDF']**2
    Vect_len = Vect_len.groupby([itemID], as_index = False, sort = False).sum().rename(columns = {'TF-IDF-Sq': 'TF-IDF-Sq-sum'})[['track_id','TF-IDF-Sq-sum']]
    Vect_len['vect_len'] = np.sqrt(Vect_len[['TF-IDF-Sq-sum']].sum(axis=1))
    TF = pd.merge(TF,Vect_len,on = itemID, how = 'left', sort = False)
    TF['TAG_WT'] = TF['TF-IDF']/TF['vect_len']
    
    TF['TAG_WT'] = TF['TAG_WT'].fillna(0)
    
    return TF


def compute_cosine(URM_all):
    #compute the cosine
    print("Compute cosine")
    similarity = URM_all.T * URM_all

    return URM_all.tocsr(), similarity

def itemKNN(similarity, URM_train, nbKNN=100):   
    #get only the n nearestneightbours
    print("Item KNN")

    #csc more faster
    similarity = similarity.tocsc()
    
    for col_index in range(URM_train.shape[1]):      
        this_item_weights = similarity.data[similarity.indices[col_index]:similarity.indices[col_index+1]]
        nearestNeightbours = np.zeros(len(this_item_weights))
        nearestNeightbours.astype(int)
        top_k_idx = np.argsort(this_item_weights)[-nbKNN:]
        for ind in top_k_idx:
            nearestNeightbours[ind] = this_item_weights[ind]
        similarity.data[similarity.indices[col_index]:similarity.indices[col_index+1]] = nearestNeightbours

    return similarity.tocsr()

def predic(URM_all, similarity, playlists):
    #predic
    print("Predic ")

    #sort top 5 value
    track_final = []  
    playlistCount = len(playlists)
    for i in range(playlistCount):        
        user_id = playlists[i]
        user_profile = URM_all[user_id]
        scores = user_profile.dot(similarity).toarray().ravel()
        # rank items
        ranking = scores.argsort()[::-1]
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        ranking = ranking[unseen_mask]
        track_final.append(ranking[:5])
 
    return track_final

def recommend(URM, similarity, user_id, at=None, exclude_seen=True):
    # compute the scores using the dot product
    user_profile = URM[user_id]
    scores = user_profile.dot(similarity).toarray().ravel()

    # rank items
    ranking = scores.argsort()[::-1]
    seen = user_profile.indices
    unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
    ranking = ranking[unseen_mask]
            
    return ranking[:at]



    
       


    








    
 
