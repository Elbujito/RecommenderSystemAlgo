import pandas as pd
import numpy as np
import scipy.sparse as sps

############################################################################
#	                       EVALUATION METHODS                              #
############################################################################

# Evaluation metrics only used when your dataset(URM_all) is splited into 2 Set URM_test and URM_train

#Create the URM test Matrix
def create_URM_test(songsbyplaylists_df, train_test_split=0.8):

    numInteractions = len(songsbyplaylists_df['playlist_id'])
    interractions = np.ones(numInteractions)
    userList = np.array(list(songsbyplaylists_df['playlist_id']))
    itemList = np.array(list(songsbyplaylists_df['track_id']))
    ratioSplit = int(len(userList) * train_test_split)

    train_mask = np.random.choice([True,False], numInteractions, p=[train_test_split, 1-train_test_split])

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((interractions[test_mask], (userList[test_mask], itemList[test_mask])))
    URM_test = URM_test.tocsr()

    songsbyplaylists_df = songsbyplaylists_df[0:ratioSplit]
    return URM_test, songsbyplaylists_df

#Compute Mean Average Precision
def MAP(recommended_items, relevant_items):
   
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

#Compute Recall
#@description: recall is the proportion of good recommendations that appear in top recommendations	
def recall(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    
    return recall_score

#Compute Precision
#@description : The precision is the proportion of recommendations that are good recommendations
def precision(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    
    return precision_score

#Compute evaluation metrics
# @param : Matrix test, Dataframe predic, playlist to recommend
def evaluate_algorithm(URM_test, predic_df, target_playlist):
    
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    
    num_eval = 0

	# evaluate only recommended playlist
    recommend_itemList = np.array(predic_df['track_ids'])
    for id in np.array(predic_df['playlist_id']):

        relevant_items = URM_test[id].indices

        if len(relevant_items)>0:
            
            recommended_itemsStr = recommend_itemList[id]
            recommended_items = recommended_itemsStr.split(",")
            num_eval+=1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)


    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    
    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP)) 