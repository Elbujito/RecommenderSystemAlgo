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

def create_mapTable_ID ( dataframe, columnID ):
    items = dataframe[columnID].drop_duplicates().reset_index(drop=True)
    corresp_df = pd.DataFrame.from_dict(items)
    corresp_df['new_id'] = corresp_df.index.values
    
    return corresp_df

def extract_tags_list( dataframe, itemID, tagID):
    items=[]
    tags=[]

    for i in range(len(dataframe.index)):
        numbers=re.findall(r'\d+', dataframe.iloc[i][tagID])
        for x in numbers:
            items.append(dataframe.iloc[i][itemID])
            tags.append(x)
   
    result= pd.DataFrame( {itemID: items, tagID: tags } )
    return result

def define_tags_occurence ( userItemDF, itemID, tagID ):
    # tag column must be in userItemDF
    userItemDF['tag_count_TF']=1
    TF= userItemDF.groupby([itemID,tagID], as_index = False, sort = False).count()[[itemID,tagID,'tag_count_TF']]
    Tag_distinct = userItemDF[[tagID,itemID]].drop_duplicates()
    DF =Tag_distinct.groupby([tagID], as_index = False, sort = False).count().rename(columns = {itemID: 'tag_count_DF'})[[tagID,'tag_count_DF']]
    a=math.log10(len(np.unique(userItemDF[itemID])))
    DF['IDF']=a-np.log10(DF['tag_count_DF'])
    TF = pd.merge(TF,DF,on = tagID, how = 'left', sort = False)
    TF['TF-IDF']=TF['tag_count_TF']*DF['IDF']
    
    TF['TF-IDF']=TF['TF-IDF'].fillna(0)
    
    Vect_len = TF[[itemID,'TF-IDF']].copy()
    Vect_len['TF-IDF-Sq'] = Vect_len['TF-IDF']**2
    Vect_len = Vect_len.groupby([itemID], as_index = False, sort = False).sum().rename(columns = {'TF-IDF-Sq': 'TF-IDF-Sq-sum'})[[itemID,'TF-IDF-Sq-sum']]
    Vect_len['vect_len'] = np.sqrt(Vect_len[['TF-IDF-Sq-sum']].sum(axis=1))
    TF = pd.merge(TF,Vect_len,on = itemID, how = 'left', sort = False)
    TF['TAG_WT'] = TF['TF-IDF']/TF['vect_len']
    
    TF['TAG_WT'] = TF['TAG_WT'].fillna(0)
    
    return TF

def compute_cosine(userItemDF, userID, itemID, interactionID):
    # interaction values must be in userItemDF
    
    # sort by playlist
    userItemDF = userItemDF.sort_values(userID)
    # get playlists, tracks and interactions
    playlists = np.array(userItemDF[userID])
    tracks = np.array(userItemDF[itemID])
    interactions = np.array(userItemDF[interactionID])      
    
    # compress data using cscr matrix and pivot
    URM_all = sps.coo_matrix((interactions, (playlists, tracks)), dtype=np.float16)
    URM_all.tocsr()
    
    #Create Data Set
    train_test_split = 0.8
    numInteractions = URM_all.nnz
    train_mask = np.random.choice([True,False], numInteractions, p=[train_test_split, 1-train_test_split])

    URM_train = sps.coo_matrix((interactions[train_mask], (playlists[train_mask], tracks[train_mask])), dtype=np.float16)
    URM_train = URM_train.tocsr() 

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((interactions[test_mask], (playlists[test_mask], tracks[test_mask])), dtype=np.float16)
    URM_test = URM_test.tocsr()

    #compute the cosine
    similarity = (URM_all.T * URM_all)
    pred = URM_all * similarity
    pred = pred.tocsr()
    
    return pred
