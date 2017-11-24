from recSysFunctions import *
from evaluation import *

if __name__ == "__main__":  # confirms that the code is under main function

    ###########################################################
    # CREATE DATASET
    ###########################################################
    # importing datasets (csv files) with separator \t
    playlists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/playlists_final.csv')
    tracks_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/tracks_final.csv')
    targetplaylists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/target_playlists.csv')
    targettracks_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/target_tracks.csv')
    songsbyplaylists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/train_final.csv') 

    #keep only target tracks
    songsbyplaylists_df = songsbyplaylists_df[(songsbyplaylists_df.track_id.isin(targettracks_df.track_id))]

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
    songsbyplaylists_df['tag'] = songsbyplaylists_df.track_id.map(tracks_df.set_index('track_id')['album'])
    TF = define_tags_occurence(songsbyplaylists_df, 'playlist_id', 'track_id', 'tag')
    
    ############################################################
    # CF
    ############################################################
    songsbyplaylists_df['TAG_WT'] = songsbyplaylists_df.track_id.map(TF.set_index('track_id')['TAG_WT'])
    print(TF)

    #create urm all, train and test
    URM_all, URM_train, URM_test = create_urm(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')
    exportToCSV = False
    isPredict = False

    #cosine similirity
    URM_all, similarity = compute_cosine(URM_all)
    
    #itemKNN
    similarity = itemKNN(similarity, URM_train)

    # get playlists to recommend
    target_playlist = np.array(targetplaylists_df['playlist_id'])

    #predic
    if(isPredict):
        track_final = predic(URM_all, similarity, target_playlist)

    #evaluation
    evaluate_algorithm(URM_test, URM_train, similarity, target_playlist[0:1000])

    if(exportToCSV):
        #################################################################################
        #convert to data frame 
        #################################################################################
        print("get past id")
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
        print('Export data to csv')
        newDf=pd.DataFrame(recommend_df['playlist_id'])
        strTracks = []
        for i in range(10000):
            strFinal = str(recommend_df.iloc[i]['track_id1']) + ' ' + str(recommend_df.iloc[i]['track_id2']) + ' ' + str(recommend_df.iloc[i]['track_id3']) + ' ' + str(recommend_df.iloc[i]['track_id4']) + ' ' + str(recommend_df.iloc[i]['track_id5'])
            strTracks.append(strFinal)

        newDf.insert(1, 'track_ids', strTracks)
        print(newDf)

        # write csv recommendation file
        newDf.to_csv('~/Documents/Polimi/RecommenderSystem/outputFiles/result.csv', sep=',' ,index=False)


    



    
       


    








    
 
