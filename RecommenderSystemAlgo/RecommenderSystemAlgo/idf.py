import pandas as pd
import numpy as np
import re
import math

def compute_ICM_idf(ICM_all):

    num_tot_items = ICM_all.shape[0]

    #count how many items have a certain feature
    items_per_feature = (ICM_all > 0).sum(axis=0)

    IDF = np.array(np.log(num_tot_items / items_per_feature))[0]
        
    ICM_idf = ICM_all.copy()
    # compute the number of non-zeros in each col
    # NOTE: this works only if X is instance of csc_matrix
    col_nnz = np.diff(ICM_idf.tocsc().indptr)
    # then normalize the values in each col
    ICM_idf.data *= np.repeat(IDF, col_nnz)
    return ICM_idf

def extract_none_list( dataframe, tagID):
    #extract tags
    print("Extract none")
    for i in range(len(dataframe.index)):       
        if  math.isnan(dataframe.iloc[i][tagID]):
            dataframe.iloc[i][tagID]= 0
           
    return dataframe[tagID]

def extract_tags_list( dataframe, itemID, tagID):
    #extract tags
    print("Extract tags")
    items=[]
    tags=[]

    for i in range(len(dataframe.index)):
        value = dataframe.iloc[i][tagID]
        id = dataframe.iloc[i][itemID]

        tab = [0,0,0,0,0]        
        items.append(id)
        if value != "[]":                      
            numbers=re.findall(r'\d+', value)
            for x in range(len(numbers)):
                tab[x] = numbers[x]           
        tags.append(tab)
        
        
    result = pd.DataFrame(tags)
    result.columns = ['tags1', 'tags2','tags3','tags4','tags5']
    result.insert(0,itemID,items)
    return result

def extract_album_list(dataframe, itemID, tagID):
    #extract tags
    print("Extract album")
    items=[]
    tags=[]

    for i in range(len(dataframe.index)):
        value = dataframe.iloc[i][tagID]
        id = dataframe.iloc[i][itemID]
        if  value == "[]" or value == "[None]":
            items.append(id)
            tags.append(0)
        else:
            numbers=re.findall(r'\d+', value)
            items.append(id)
            tags.append(int(numbers[0]))

    result= pd.DataFrame( {itemID: items, tagID: tags } )
    return result

def extract_artistOrDuration_list(dataframe, itemID, tagID):
    #extract tags
    print("Extract artist or duration or playcount")
    items=[]
    tags=[]

    for i in range(len(dataframe.index)):
        value = dataframe.iloc[i][tagID]
        id = dataframe.iloc[i][itemID]
        items.append(id)
        tags.append(int(value))

    result= pd.DataFrame( {itemID: items, tagID: tags } )
    return result

def extract_playcount_list(dataframe, itemID, tagID):
    #extract tags
    print("Extract artist or duration or playcount")
    items=[]
    tags=[]

    for i in range(len(dataframe.index)):
        value = dataframe.iloc[i][tagID]
        id = dataframe.iloc[i][itemID]
        if  value==-1 or math.isnan(value):
            items.append(id)
            tags.append(0)
        else:
            items.append(id)
            tags.append(int(value))

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