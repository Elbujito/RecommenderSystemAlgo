from recSysFunctions import *

if __name__ == "__main__":  # confirms that the code is under main function

    # importing datasets (csv files) with separator \t
    playlists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/playlists_final.csv')
    tracks_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/tracks_final.csv')
    targetplaylists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/target_playlists.csv')
    targettracks_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/target_tracks.csv')
    songsbyplaylists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/train_final.csv') 
    
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
    
    ###########################################################
    # CREATE TRACKS PROFILE USING TF-IDF FOR OCCURENCE OF TAGS
    ###########################################################
    # content-based filtering using TF-IDF
    # extracting tags from list
    # extract album of track
    songsbyplaylists_df['tag_album'] = songsbyplaylists_df.track_id.map(tracks_df.set_index('track_id')['album'])
    TF_tracksAlbum = define_tags_occurence(songsbyplaylists_df, 'track_id', 'tag_album')
    songsbyplaylists_df['TAG_WT'] = songsbyplaylists_df.track_id.map(TF_tracksAlbum.set_index('track_id')['TAG_WT'])
    
    # extract tags of tracks
    tracksByTags = extract_tags_list(tracks_df, 'track_id', 'tags')
    TF_tracksTag = define_tags_occurence(tracksByTags, 'track_id', 'tags')
    
    # extract artist_id of tracks
    TF_tracksArtist = define_tags_occurence(tracks_df, 'track_id', 'artist_id')
    
    # extract playcount of tracks  
    TF_tracksPlaycount = define_tags_occurence(tracks_df, 'track_id', 'playcount')
    
    # extract duration of tracks
    TF_tracksDuration = define_tags_occurence(tracks_df, 'track_id', 'duration')
    
    # extract title of playlist
    playlistsByTitle = extract_tags_list(playlists_df, 'playlist_id', 'title')
    TF_playlistsTitle = define_tags_occurence(playlistsByTitle, 'playlist_id', 'title')
    
    # extract owner of playlists
    TF_playlistsOwner = define_tags_occurence(playlists_df, 'playlist_id', 'duration')
    
    # extract created_at of playlists
    TF_playlistsCreationDate = define_tags_occurence(playlists_df, 'playlist_id', 'created_at')
    
    # extract duration of playlists
    TF_playlistsDuration = define_tags_occurence(playlists_df, 'playlist_id', 'duration')
    
    ############################################################
    # COMPUTE COSINE SIMILARITY
    ############################################################
    print('Cosine similarity')
    pred = compute_cosine(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')
    
    # get playlists to recommend
    targetplaylists_df = targetplaylists_df.sort_values('playlist_id')
    target_playlist = targetplaylists_df['playlist_id']
    target_playlist = np.array(target_playlist)
    target_playlist.astype(int)
    
    print('Sort topN ranking')
    #sort top n ranking
    track_final = []
    for i in target_playlist:
        
        this_item_weights = pred.data[pred.indptr[i]:pred.indptr[i+1]]
        top_k_idx = np.argsort(this_item_weights)[-5:]
        track_indices = pred.indices[pred.indptr[i]:pred.indptr[i+1]]
        track_indices = track_indices[top_k_idx]
        track_final.append(track_indices)

    #convert to data frame 
    recommend_df = pd.DataFrame(track_final)
    recommend_df.columns = ['track_id1', 'track_id2', 'track_id3', 'track_id4', 'track_id5']
    
    recommend_df['track_id1'] = recommend_df.track_id1.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id2'] = recommend_df.track_id2.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id3'] = recommend_df.track_id3.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id4'] = recommend_df.track_id4.map(map_trackID.set_index('new_id')['track_id'])
    recommend_df['track_id5'] = recommend_df.track_id5.map(map_trackID.set_index('new_id')['track_id'])
    
    recommend_df.insert(0, 'playlist_id', target_playlist)
    recommend_df['playlist_id'] = recommend_df.playlist_id.map(map_playlistID.set_index('new_id')['playlist_id'])
    
    # creation of dataframe for csvfile
    newDf=pd.DataFrame(recommend_df['playlist_id'])
    strTracks = []
    for i in range(10000):
        strFinal = str(recommend_df.iloc[i]['track_id1']) + ' ' + str(recommend_df.iloc[i]['track_id2']) + ' ' + str(recommend_df.iloc[i]['track_id3']) + ' ' + str(recommend_df.iloc[i]['track_id4']) + ' ' + str(recommend_df.iloc[i]['track_id5'])
        strTracks.append(strFinal)
    
    newDf.insert(1, 'track_ids', strTracks)
    #print(newDf)

    # write csv recommendation file
    #newDf.to_csv('~/Documents/Polimi/RecommenderSystem/outputFiles/result.csv', sep=',' ,index=False)
