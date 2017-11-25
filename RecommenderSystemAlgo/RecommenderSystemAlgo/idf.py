import pandas as pd
import numpy as np
import re
import math

def extract_tags_list( dataframe, itemID, tagID):
    #extract tags
    print("Exctract tags")
    items=[]
    tags=[]

    for i in range(len(dataframe.index)):
        if dataframe.iloc[i][tagID] == "[]" or dataframe.iloc[i][tagID] == "[None]":
            items.append(dataframe.iloc[i][itemID])
            tags.append(0)
        else:
            numbers=re.findall(r'\d+', dataframe.iloc[i][tagID])
            for x in numbers:
                items.append(dataframe.iloc[i][itemID])
                tags.append(int(x))
   
    result= pd.DataFrame( {itemID: items, tagID: tags } )
    return result

def define_tags_occurence ( userItemDF, userID, itemID, tagID ):
    # tag column must be in userItemDF
    print("Compute IDF")
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
    Vect_len = Vect_len.groupby([itemID], as_index = False, sort = False).sum().rename(columns = {'TF-IDF-Sq': 'TF-IDF-Sq-sum'})[[itemID,'TF-IDF-Sq-sum']]
    Vect_len['vect_len'] = np.sqrt(Vect_len[['TF-IDF-Sq-sum']].sum(axis=1))
    TF = pd.merge(TF,Vect_len,on = itemID, how = 'left', sort = False)
    TF['TAG_WT'] = TF['TF-IDF']/TF['vect_len']
    
    TF['TAG_WT'] = TF['TAG_WT'].fillna(0)
    
    return TF

def sum_weight(df) :
    #sum weight tags
    print("Sum Weight")

    df = df.groupby(['playlist_id', 'track_id'])['TAG_WT'].mean().reset_index()

    return df