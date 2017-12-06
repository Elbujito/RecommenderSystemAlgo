from recSysFunctions import *
from evaluation import evaluate_algorithm
from dataset import *
from itemKNNRecommender import ItemKNN_CF_Recommender
from contentBasedFiltering import ContentBasedFiltering_Recommender
from userKNNRecommender import UserKNN_CF_Recommender
from slimRecommender import SLIM_Recommender, Recommender, MultiThreadSLIM
from slim import SLIM_BPR_Recommender
from idf import *

if __name__ == "__main__":  # confirms that the code is under main function

    ###########################################################
    # CREATE DATASET
    ###########################################################
    # importing datasets (csv files) with separator \t
    playlists_df, tracks_df, targetplaylists_df, targettracks_df, songsbyplaylists_df, map_playlistID, map_trackID = init()

   ####################################################################
    # CREATE TRACKS PROFILE USING TF-IDF FOR OCCURENCE OF TAGS
    isTagsExported = False
    isAttributsExported = False
    ####################################################################

    songsbyplaylists_idx=songsbyplaylists_df.set_index(['playlist_id', 'track_id'])

    if(isTagsExported):
        # extracting tags from list
        print("Extract Tags")      

        tracksByTags = extract_tags_list(tracks_df, 'track_id', 'tags')
        tracksByTags_idx=tracksByTags.set_index('track_id')

        tracksByTags_df = pd.merge(songsbyplaylists_idx.reset_index(), tracksByTags_idx.reset_index(),on=['track_id'], how='inner')
     
        # write csv recommendation file
        tracksByTags_df.to_csv('~/Documents/Polimi/RecommenderSystem/inputFiles/train_tags.csv', sep='\t' ,index=False)

    tracksByTags_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/train_tags.csv')
    print(tracksByTags_df)

    if(isAttributsExported):
    # content-based filtering using TF-IDF

        # extract playcount of tracks  
        print("Extract Playcount")
        #tracksByArtist = extract_artistOrDuration_list(tracks_df, 'track_id','artist_id')      
        #tracksByArtist_idx=tracksByArtist.set_index('track_id')
        #tracksByArtists_df = pd.merge(songsbyplaylists_idx.reset_index(), tracksByArtist_idx.reset_index(),on=['track_id'], how='inner')
        #tracksByPlaycount = extract_playcount_list(tracks_df, 'track_id', 'playcount')
        #songsbyplaylists_df['playcount'] = tracksByPlaycount.track_id.map(map_trackID.set_index('track_id')['playcount'])

        # extract artist_id of tracks
        print("Extract Artist")
        tracksByArtist = extract_artistOrDuration_list(tracks_df, 'track_id','artist_id')      
        tracksByArtist_idx=tracksByArtist.set_index('track_id')
        tracksByArtists_df = pd.merge(songsbyplaylists_idx.reset_index(), tracksByArtist_idx.reset_index(),on=['track_id'], how='inner')
        
        # extract album of track
        print("Extract Album")
        tracksByAlbum = extract_album_list(tracks_df, 'track_id', 'album')
        tracksByAlbum_idx=tracksByAlbum.set_index('track_id')
        tracksByAlbums_df = pd.merge(songsbyplaylists_idx.reset_index(), tracksByAlbum_idx.reset_index(),on=['track_id'], how='inner')

        # extract duration of tracks
        print("Extract Duration")
        #tracksByDuration = extract_artistOrDuration_list(tracks_df, 'track_id', 'duration')
        #tracksByDuration_idx=tracksByDuration.set_index('track_id')
        #tracksByDurations_df = pd.merge(songsbyplaylists_idx.reset_index(), tracksByDuration_idx.reset_index(),on=['track_id'], how='inner')

        #tracksByDurations_df.to_csv('~/Documents/Polimi/RecommenderSystem/inputFiles/train_duration.csv', sep='\t' ,index=False)
        tracksByArtists_df.to_csv('~/Documents/Polimi/RecommenderSystem/inputFiles/train_artists.csv', sep='\t' ,index=False)
        tracksByAlbums_df.to_csv('~/Documents/Polimi/RecommenderSystem/inputFiles/train_albums.csv', sep='\t' ,index=False)
    
    tracksByAlbums_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/train_albums.csv')
    tracksByArtists_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/train_artists.csv')
    print(tracksByAlbums_df)
    print(tracksByArtists_df)
    #print(songsByAttributs_df)
 
    #playlist to recommend
    target_playlist = np.array(targetplaylists_df['playlist_id'])

    ############################################################
    isPredic = False 
    ############################################################
    isSLIM_BPR = False
    # SLIM BPR
    ############################################################
    if(isSLIM_BPR):
        print("Apply SLIM")
        songsbyplaylists_df['TAG_WT'] = 1
        URM_all, URM_train, URM_test = create_urm(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')

        similarity = compute(URM_train.tocsc())

        rec_SLIM_BPR = SLIM_BPR_Recommender(URM_train,similarity, epochs=1)
        rec_SLIM_BPR.fit()

        #playlist to recommend
        target_playlist = np.array(targetplaylists_df['playlist_id'])

        target_tracks = np.array(targettracks_df['track_id'])

        evaluate_algorithm(URM_test,rec_SLIM_BPR,target_playlist[0:1000],target_tracks)
    ############################################################
    isSLIM = True
    # SLIM
    ############################################################
    if(isSLIM):
        print("Apply SLIM")
        songsbyplaylists_df['TAG_WT'] = 1
        URM_all, URM_train, URM_test = create_urm(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')
        
        #playlist to recommend
        target_playlist = np.array(targetplaylists_df['playlist_id'])

        rec_SLIM = MultiThreadSLIM()
        rec_SLIM.fit(URM_all)

        if(isPredic):
            URM = URM_all
        else:
            URM = URM_train

        target_tracks = np.array(targettracks_df['track_id'])
        #predic
        if(isPredic):
            track_final = rec.predic()

        else:
            evaluate_algorithm(URM_test,rec,target_playlist[0:1000],target_tracks)

    ############################################################
    isCF = False
    # CF
    ############################################################
    if(isCF):
        print("Apply CF")
        songsbyplaylists_df['TAG_WT'] = 1
        URM_all, URM_train, URM_test = create_urm(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')
        
        #cosine similirity

        similarity = compute(URM_all.tocsc())

        #playlist to recommend
        target_playlist = np.array(targetplaylists_df['playlist_id'])

        if(isPredic):
            URM = URM_all
        else:
            URM = URM_train

        rec = ItemKNN_CF_Recommender(URM.tocsr(), similarity, target_playlist, k=50)
        rec.fit(URM_all)
        target_tracks = np.array(targettracks_df['track_id'])
        #predic
        if(isPredic):
            track_final = rec.predic()

        else:
            evaluate_algorithm(URM_test,rec,target_playlist[0:1000],target_tracks)

    ############################################################
    isIDF = False
    # IDF
    ############################################################
    if(isIDF):
        
        tracksByTags_df['TAG_WT'] = 1
        playlists = np.array(tracksByTags_df['playlist_id'])
        tracks = np.array(tracksByTags_df['track_id'])
        interactions = np.array(tracksByTags_df['TAG_WT'])    
        split=0.5
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

        print("Apply IDF")
        #songsbyplaylists_df['TAG_WT']=1
       
        #URM_all, URM_train, URM_test = create_urm(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')

        ICM_all_album = create_icm(tracksByArtists_df,'artist_id','track_id')
        ICM_all_artist = create_icm(tracksByAlbums_df,'album','track_id')
        #ICM_all_duration = create_icm(songsByAttributs_df,'duration','track_id')
        #ICM_all_playcount = create_icm(songsByAttributs_df,'playcount','track_id')
        ICM_all_tagid1 = create_icm(tracksByTags_df,'tags1','track_id')
        ICM_all_tagid2 = create_icm(tracksByTags_df,'tags2','track_id')
        ICM_all_tagid3 = create_icm(tracksByTags_df,'tags3','track_id')
        ICM_all_tagid4 = create_icm(tracksByTags_df,'tags4','track_id')
        ICM_all_tagid5 = create_icm(tracksByTags_df,'tags5','track_id')
        
        #numTags = len(tracksByTags_df['tags1']) + len(tracksByTags_df['tags2']) + len(tracksByTags_df['tags3']) + len(tracksByTags_df['tags4'])  + len(tracksByTags_df['tags5']) + len(tracksByArtists_df['artist_id']) + len(tracksByAlbums_df['album']) 
        from scipy.sparse import vstack

        print(" stack ")
        #ICM_all = vstack([ICM_all_album, ICM_all_artist])
        #ICM_all = ICM_all_tagid2
        target_tracks = np.array(targettracks_df['track_id'])

        ICM_all = vstack([ICM_all_tagid1, ICM_all_tagid2, ICM_all_tagid3, ICM_all_tagid4, ICM_all_tagid5])
       
        
        similarity = ICM_all.T * ICM_all #compute_ICM_idf(ICM_all)
        
        #similarity = compute(ICM_idf.tocsc())
        print(similarity)
        
        if(isPredic):
            URM = URM_all
        else:
            URM = URM_train
         
        rec_idf = ContentBasedFiltering_Recommender(URM.tocsr(), similarity, target_playlist, k=100)

        rec_idf.fit(ICM_all)

        if(isPredic):
            track_final = []  
            playlistCount = len(rec_idf.target_playlists)
            for user_id in range(playlistCount):     
                user_profile = rec_idf.dataset[user_id]
                scores = user_profile.dot(rec_idf.similarity).toarray().ravel()
                print("scores :",scores)
                # rank items
                ranking = scores.argsort()[::-1]
                
                print(ranking)
                seen = user_profile.indices
                print(seen)
                unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
                ranking = ranking[unseen_mask]    
                print("ranking after :", ranking)
                track_final.append(ranking[:5])
                print("track final :", track_final)
                aaaa
        else:
            evaluate_algorithm(URM_test,rec_idf,target_playlist[0:1000], target_tracks)

    #################################################################################
    #WRITE RESULT FILE
    #################################################################################
    if(isPredic):
        #unMap data
        recommend_df = unMap(track_final, map_playlistID, map_trackID, target_playlist)

        #export to csv file
        export_csv(recommend_df,'~/Documents/Polimi/RecommenderSystem/outputFiles/result.csv')


    



    
       


    








    
 
