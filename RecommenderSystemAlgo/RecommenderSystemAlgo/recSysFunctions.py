import pandas as pd
import numpy as np
import scipy.sparse as sps
import math

def read_file(path):

    str(path)

    #extract csv file to df
    object_df = pd.read_csv(path, sep='\t')

    return object_df

def export_csv(recommend_df,path):
    # creation of dataframe for csvfile
    df=pd.DataFrame(recommend_df['playlist_id'])
    strTracks = []
    for i in range(10000):
        strFinal = str(recommend_df.iloc[i]['track_id1']) + ' ' + str(recommend_df.iloc[i]['track_id2']) + ' ' + str(recommend_df.iloc[i]['track_id3']) + ' ' + str(recommend_df.iloc[i]['track_id4']) + ' ' + str(recommend_df.iloc[i]['track_id5'])
        strTracks.append(strFinal)

    df.insert(1, 'track_ids', strTracks)
    print(df)

    # write csv recommendation file
    df.to_csv(path, sep=',' ,index=False)

def create_mapTable_ID ( dataframe, columnID ):
    items = dataframe[columnID].drop_duplicates().reset_index(drop=True)
    corresp_df = pd.DataFrame.from_dict(items)
    corresp_df['new_id'] = corresp_df.index.values
    
    return corresp_df

def init():
    playlists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/playlists_final.csv')
    tracks_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/tracks_final.csv')
    targetplaylists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/target_playlists.csv')
    targettracks_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/target_tracks.csv')
    songsbyplaylists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/train_final.csv') 

    #keep only target tracks
    songsbyplaylists_df = songsbyplaylists_df[(songsbyplaylists_df.track_id.isin(targettracks_df.track_id))]
    #tracks_df = tracks_df[(tracks_df.track_id.isin(targettracks_df.track_id))]

    # creating correspondance tables
    map_playlistID = create_mapTable_ID(playlists_df, 'playlist_id')
    map_trackID = create_mapTable_ID(tracks_df, 'track_id')
    
    # mapping ids with new ids
    playlists_df['playlist_id'] = playlists_df.playlist_id.map(map_playlistID.set_index('playlist_id')['new_id'])
    tracks_df['track_id'] = tracks_df.track_id.map(map_trackID.set_index('track_id')['new_id'])
    songsbyplaylists_df['playlist_id'] = songsbyplaylists_df.playlist_id.map(map_playlistID.set_index('playlist_id')['new_id'])
    songsbyplaylists_df['track_id'] = songsbyplaylists_df.track_id.map(map_trackID.set_index('track_id')['new_id'])
    targetplaylists_df['playlist_id'] = targetplaylists_df.playlist_id.map(map_playlistID.set_index('playlist_id')['new_id'])
    targettracks_df['track_id'] = targettracks_df.track_id.map(map_trackID.set_index('track_id')['new_id'])

    return playlists_df, tracks_df, targetplaylists_df, targettracks_df, songsbyplaylists_df, map_playlistID, map_trackID

def unMap(track_final, map_playlistID, map_trackID, target_playlist):
    print("UnMap")
    recommend_df = pd.DataFrame(track_final)
    recommend_df.columns = ['track_id1', 'track_id2', 'track_id3', 'track_id4', 'track_id5']
    
    recommend_df['track_id1'] = recommend_df.track_id1.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id2'] = recommend_df.track_id2.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id3'] = recommend_df.track_id3.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id4'] = recommend_df.track_id4.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id5'] = recommend_df.track_id5.map(map_trackID.set_index('new_id')['track_id'])
    
    recommend_df.insert(0, 'playlist_id', target_playlist)
    recommend_df['playlist_id'] = recommend_df.playlist_id.map(map_playlistID.set_index('new_id')['playlist_id'])

    return recommend_df



    
       


    








    
 
