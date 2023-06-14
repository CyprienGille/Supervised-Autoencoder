"""
This code is an implementation of the other methods used for comparison in the article :
    Learning a confidence score and the latent space of a new Supervised Autoencoder 
for diagnosis and prognosis in clinical metabolomic studies.
"""
#%%
import sys

if "../functions/" not in sys.path:
    sys.path.append("../functions/")
import os

import functions.functions_compare as fc
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


if __name__ == "__main__":

    # Set params :
    # filename = "LUNG.csv"
    # filename = "BRAIN_MID.csv"
    filename = "GC_Brest_D_MB.csv"

    outputPath = "results_compare/" + filename.split(".")[0] + "/"
    if not os.path.exists(outputPath):  # make the directory if it does not exist
        os.makedirs(outputPath)

    SEEDS = [5, 6, 7]
    algo_list = [
        "svm",
        "plsda",
        "RF",
        # "logreg",
    ]  # ML algorithms to compare to

    doScale = True  # scaling along rows
    doTopgenes = True  # feature ranking

    # Load data
    X, Yr, nbr_clusters, feature_names = fc.readData(filename)

    # Data Preprocessing
    X = np.log(abs(X + 1))
    X = X - np.mean(X, axis=0)
    if doScale:
        X = scale(X, axis=0)

    ######## Main #######

    print("Started training")
    for seed in SEEDS:
        # Processing
        print(f"------ Seed {seed} ------")
        (
            accTestCompare,
            df_timeElapsed,
            aucTestCompare,
            df_featureList,
        ) = fc.basic_run_other(
            X,
            Yr,
            nbr_clusters,
            algo_list,
            genenames=feature_names,
            clusternames=None,
            nfold=4,
            rng=seed,
            doTopGenes=True,
        )
        df_timeElapsed.to_csv(outputPath + "timeElapsed.csv")

        if seed == SEEDS[0]:
            accTestCompare_final = accTestCompare.iloc[:4, :]
            aucTestCompare_final = aucTestCompare.iloc[:4, :]
            if doTopgenes:
                df_featureList_final = df_featureList
        else:
            accTestCompare_final = pd.concat(
                [accTestCompare_final, accTestCompare.iloc[:4, :]]
            )
            aucTestCompare_final = pd.concat(
                [aucTestCompare_final, aucTestCompare.iloc[:4, :]]
            )
            if doTopgenes:
                for met in range(len(df_featureList_final)):
                    df_featureList_final[met] = df_featureList_final[met].join(
                        df_featureList[met]["weights"], rsuffix=" {}".format(seed)
                    )
    mean = pd.DataFrame(accTestCompare_final.mean(axis=0))

    if doTopgenes:
        for met in range(len(df_featureList_final)):
            mean_met = pd.DataFrame(df_featureList_final[met].iloc[:, 1:].mean(axis=1))
            std_met = pd.DataFrame(df_featureList_final[met].iloc[:, 1:].std(axis=1))
            mean_met.columns = ["Mean"]
            df_featureList_final[met] = df_featureList_final[met].join(mean_met)
            std_met.columns = ["Std"]
            df_featureList_final[met] = df_featureList_final[met].join(std_met)

    std = pd.DataFrame(accTestCompare_final.std(axis=0))
    mean.columns = ["Mean"]
    accTestCompare_final = accTestCompare_final.T.join(mean).T
    std.columns = ["Std"]
    accTestCompare_final = accTestCompare_final.T.join(std).T
    accTestCompare_final.to_csv(outputPath + "accCompare.csv")

    mean = pd.DataFrame(aucTestCompare_final.mean(axis=0))
    std = pd.DataFrame(aucTestCompare_final.std(axis=0))
    mean.columns = ["Mean"]
    aucTestCompare_final = aucTestCompare_final.T.join(mean).T
    std.columns = ["Std"]
    aucTestCompare_final = aucTestCompare_final.T.join(std).T
    aucTestCompare_final.to_csv(outputPath + "aucCompare.csv")

    if doTopgenes:
        for seed, algo in enumerate(algo_list):

            df_featureList_final[seed].to_csv(f"{outputPath}topgenes_{algo}.csv")
