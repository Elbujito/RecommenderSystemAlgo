from recSysFunctions import *
from evaluation import evaluate_algorithm
from dataset import *
from itemKNNRecommender import ItemKNN_CF_Recommender
from idf import *

if __name__ == "__main__":  # confirms that the code is under main function

    ###########################################################
    # CREATE DATASET
    ###########################################################
    # importing datasets (csv files) with separator \t
    playlists_df, tracks_df, targetplaylists_df, targettracks_df, songsbyplaylists_df, map_playlistID, map_trackID = init()

    ####################################################################
    # CREATE TRACKS PROFILE USING TF-IDF FOR OCCURENCE OF TAGS
    isNotExported = False
    ####################################################################
    print("Extract and Define Occurence Title")
    if(isNotExported):
        # content-based filtering using TF-IDF, extracting tags from tracks
        tracksByTags = extract_tags_list(tracks_df, 'track_id', 'tags')
        # set multiple index of tracksByTags to track_id and tags
        print("Set multiple index of tracksByTags to track_id and tags")
        tracksByTags=tracksByTags.set_index(['track_id', 'tags'])
        # set multiple index of songsbyplaylists_df to track_id and playlist_id
        print("Set multiple index of songsbyplaylists_df to track_id and playlist_id")
        songsbyplaylists_df=songsbyplaylists_df.set_index(['track_id', 'playlist_id'])
        # merging the 2 tables from the index track_id
        print("Merging the 2 tables from the index track_id")
        result = pd.merge(songsbyplaylists_df.reset_index(), tracksByTags.reset_index(),on=['track_id'], how='inner')
        # TF-IDF for songs by playlists
        TF_trackTags=define_tags_occurence(result, 'playlist_id', 'track_id', 'tags')
        TF_trackTags.set_index('track_id')
        # merging songsbyplaylists_df and TF_trackTags
        print("Merging songsbyplaylists_df and TF_trackTags")
        songsplayliststags_df = pd.merge(songsbyplaylists_df.reset_index(), TF_trackTags.reset_index(),on=['track_id'], how='inner')
     
        # write csv recommendation file
        songsplayliststags_df.to_csv('~/Documents/Polimi/RecommenderSystem/inputFiles/train_tags.csv', sep='\t' ,index=False)
    else: 
        songsplayliststags_df = read_file('~/Documents/Polimi/RecommenderSystem/inputFiles/train_tags.csv')

    # content-based filtering using TF-IDF
    # extracting tags from list
    # extract album of track
    print("Extract and Define Occurence Album")
    songsbyplaylists_df['album'] = songsbyplaylists_df.track_id.map(tracks_df.set_index('track_id')['album'])
    TF_tracksAlbum = define_tags_occurence(songsbyplaylists_df,'playlist_id','track_id','album')
    songsbyplaylists_df['TAG_WT_ALBUM'] = songsbyplaylists_df.track_id.map(TF_tracksAlbum.set_index('track_id')['TAG_WT'])

    # extract artist_id of tracks
    #print("Extract and Define Occurence Artist")
    #songsbyplaylists_df['artist'] = songsbyplaylists_df.track_id.map(tracks_df.set_index('track_id')['artist_id'])
    #TF_tracksArtist = define_tags_occurence(songsbyplaylists_df,'playlist_id','track_id','artist_id')
    #songsbyplaylists_df['TAG_WT_ARTIST'] = songsbyplaylists_df.track_id.map(TF_tracksArtist.set_index('track_id')['TAG_WT'])

    # extract playcount of tracks  
    print("Extract and Define Occurence Playcount")
    songsbyplaylists_df['playcount'] = songsbyplaylists_df.track_id.map(tracks_df.set_index('track_id')['playcount'])
    TF_tracksPlaycount = define_tags_occurence(songsbyplaylists_df,'playlist_id','track_id', 'playcount')
    songsbyplaylists_df['TAG_WT_PLAYCOUNT'] = songsbyplaylists_df.track_id.map(TF_tracksPlaycount.set_index('track_id')['TAG_WT'])

    # extract duration of tracks
    print("Extract and Define Occurence Duration")
    songsbyplaylists_df['duration'] = songsbyplaylists_df.track_id.map(tracks_df.set_index('track_id')['duration'])
    TF_tracksDuration = define_tags_occurence(songsbyplaylists_df,'playlist_id','track_id', 'duration')
    songsbyplaylists_df['TAG_WT_DURATION'] = songsbyplaylists_df.track_id.map(TF_tracksDuration.set_index('track_id')['TAG_WT'])

    print("Sum all weight")
    songsbyplaylists_df['WEIGHT'] = songsbyplaylists_df['TAG_WT_ALBUM'] + songsbyplaylists_df['TAG_WT_PLAYCOUNT'] + songsbyplaylists_df['TAG_WT_DURATION']

    #keep only usefull data
    weight_df = pd.DataFrame(songsplayliststags_df['playlist_id'])
    weight_df.insert(1,'track_id',songsplayliststags_df['track_id'])
    weight_df.insert(2,'TAG_WT',songsplayliststags_df['TAG_WT'])
    
    #sum of each tags for each songs in each playlists
    weight_df = weight_df.sort_values(by=['playlist_id','track_id'])
    weight_df = sum_weight(weight_df)

    #playlist to recommend
    target_playlist = np.array(targetplaylists_df['playlist_id'])

    ############################################################
    isPredic = True 
    ############################################################
    isIDF = True
    # IDF
    ############################################################
    if(isIDF):
        print("Apply IDF")
        weight_df['TAG_WT']=1

        ICM_all, URM_train, URM_test = create_urm(weight_df, 'playlist_id', 'track_id', 'TAG_WT')

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

        similarity = ICM_idf.T * ICM_idf

        if(isPredic):
            URM = ICM_all
        else:
            URM = URM_train

        rec_idf = ItemKNN_CF_Recommender(URM.tocsr(), similarity, target_playlist, k=100)
        rec_idf.fit(ICM_idf)

        if(isPredic):
            track_final = rec_idf.predic()
        else:
            evaluate_algorithm(URM_test, rec_idf, target_playlist[0:1000])

    ############################################################
    isCF = False
    # CF
    ############################################################
    if(isCF):
        print("Apply CF")
        songsbyplaylists_df['TAG_WT'] = 1
        URM_all, URM_train, URM_test = create_urm(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')

        #cosine similirity
        URM_all, similarity = compute_cosine(URM_all)

        #playlist to recommend
        target_playlist = np.array(targetplaylists_df['playlist_id'])

        if(isPredic):
            URM = URM_all
        else:
            URM = URM_train

        rec = ItemKNN_CF_Recommender(URM, similarity, target_playlist, k=100)
        rec.fit(URM_all)
        
        #predic
        if(isPredic):
            track_final = rec.predic()
        else:
            evaluate_algorithm(URM_test,rec,target_playlist[0:1000])

    #################################################################################
    #WRITE RESULT FILE
    #################################################################################
    if(isPredic):
        #unMap data
        recommend_df = unMap(track_final, map_playlistID, map_trackID, target_playlist)

        #export to csv file
        export_csv(recommend_df,'~/Documents/Polimi/RecommenderSystem/outputFiles/result.csv')


    



    
       


    








    
 
