from recSysFunctions import *
from evaluation import evaluate_algorithm
from cosine import compute_cosine
from dataset import create_urm
from itemKNNRecommender import ItemKNN_CF_Recommender
from idf import *

if __name__ == "__main__":  # confirms that the code is under main function

    ###########################################################
    # CREATE DATASET
    ###########################################################
    # importing datasets (csv files) with separator \t
    playlists_df, tracks_df, targetplaylists_df, targettracks_df, songsbyplaylists_df, map_playlistID, map_trackID = init()
       
    ############################################################
    # DATASET
    ############################################################
    #create urm all, train and test
    songsbyplaylists_df['TAG_WT'] = 1
    URM_all, URM_train, URM_test = create_urm(songsbyplaylists_df, 'playlist_id', 'track_id', 'TAG_WT')

    ############################################################
    isCF = True
    isPredic = False 
    # CF
    ############################################################
    if(isCF):

        #cosine similirity
        URM_all, similarity = compute_cosine(URM_all)

        #playlist to recommend
        target_playlist = np.array(targetplaylists_df['playlist_id'])

        rec = ItemKNN_CF_Recommender(URM_train, similarity, target_playlist, k=100)
        rec.fit(URM_all)

        #predic
        if(isPredic):
            track_final = rec.predic()

        #evaluation
        evaluate_algorithm(URM_test,rec,target_playlist[0:1000])

    #################################################################################
    isExportToCSV = False
    #WRITE RESULT FILE
    #################################################################################
    if(isExportToCSV):
        #unMap data
        recommend_df = unMap(track_final, map_playlistID, map_trackID)

        #export to csv file
        export_csv(recommend_df,'~/Documents/Polimi/RecommenderSystem/outputFiles/result.csv')


    



    
       


    








    
 
