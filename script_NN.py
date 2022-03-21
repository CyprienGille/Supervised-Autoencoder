# -*- coding: utf-8 -*-
"""
Copyright   I3S CNRS UCA 

This code is an implementation of statistical evaluation of our autoencoder discribe in the article :
An efficient diagnostic that uses the latent space of a Non-Parametric Supervised Autoencoder 
for metabolomic datasets of clinical studies.

When using this code , please cite

 Barlaud, Michel and  Guyard, Frederic
 Learning sparse deep neural networks using efficient structured projections on convex constraints for green ai. ICPR 2020 Milan Italy (2020)

@INPROCEEDINGS{9412162,  
               author={Barlaud, Michel and Guyard, Frédéric},  
               booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},   
               title={Learning sparse deep neural networks using efficient structured projections on convex constraints for green AI},  
               year={2021}, 
               volume={}, 
               number={}, 
               pages={1566-1573}, 
               doi={10.1109/ICPR48806.2021.9412162}}

and 

David Chardin, Cyprien Gille, Thierry Pourcher and Michel Barlaud :
    An efficient diagnostic that uses the latent space of a Non-Parametric Supervised Autoencoder 
for metabolomic datasets of clinical studies.

Parameters : 
    
    - Seed (line 80)
    - Database (line 106) (variable file_name)
    - Projection (line 145)
    - Constraint ETA (line 81)
    - Scaling (line 163)
    - Metabolomic selection (line 156)
    
Results_stat
    -accuracy
    -Probability
    -Top features 
    
    
    
"""
#%%
import os
import sys

if "../functions/" not in sys.path:
    sys.path.append("../functions/")


import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch import nn
import time
from sklearn import metrics

# lib in '../functions/'
import functions.functions_DNN as fd
import functions.functions_network_pytorch as fnp
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns


#################################

if __name__ == "__main__":
    # ------------ Parameters ---------

    ####### Set of parameters : ######

    # Set seed
    Seed = [5, 6, 7]

    # Set device (Gpu or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nfold = 4
    N_EPOCHS = 30
    N_EPOCHS_MASKGRAD = 30  # number of epochs for training masked gradient
    LR = 0.0005  # Learning rate
    BATCH_SIZE = 8  # Optimize the trade off between accuracy and Computational time
    LOSS_LAMBDA = 0.001  # Total loss =λ * loss_autoencoder +  loss_classification
    bW = 1  # Kernel size for distributions
    # Scaling
    doScale = True
    #    doScale = False
    # log transform
    doLog = True

    criterion_reconstruction = nn.SmoothL1Loss(reduction="sum")  # SmoothL1Loss

    # Loss functions for classification
    criterion_classification = nn.CrossEntropyLoss(reduction="sum")

    TIRO_FORMAT = True
    # file_name = "LUNG.csv"
    # file_name = "BRAIN_MID.csv"
    file_name = "GC_Brest_D_MB.csv"

    # Choose net
    # net_name = "LeNet"
    net_name = "DNN"
    n_hidden = 96  # nombre de neurones sur la couche cachee du DNN

    # Save Results or not
    SAVE_FILE = True
    # Output Path
    outputPath = "results_dnn/" + file_name.split(".")[0] + "/"
    if not os.path.exists(outputPath):  # make the directory if it does not exist
        os.makedirs(outputPath)

    # Do pca or t-SNE
    Do_pca = True
    Do_tSNE = True
    run_model = "No_proj"
    # Do projection at the middle layer or not
    DO_PROJ_middle = False

    # Do projection (True)  or not (False)
    #    GRADIENT_MASK = False
    GRADIENT_MASK = True
    if GRADIENT_MASK:

        run_model = "ProjectionLastEpoch"
    # Choose projection function
    if not GRADIENT_MASK:
        TYPE_PROJ = "No_proj"
        TYPE_PROJ_NAME = "No_proj"
    else:
        #        TYPE_PROJ = ft.proj_l1ball         # projection l1
        TYPE_PROJ = fd.proj_l11ball  # original projection l11 (les colonnes a zero)
        #        TYPE_PROJ = ft.proj_l21ball        # projection l21
        TYPE_PROJ_NAME = TYPE_PROJ.__name__

    AXIS = 0  #  for PGL21

    # Top genes params

    DoTopGenes = True
    #    DoTopGenes = False

    ETA = 10000  # Control feature selection
    # ------------ Main loop ---------
    # Load data

    X, Y, feature_name, label_name, patient_name, LFC_Rank = fd.ReadData(
        file_name, TIRO_FORMAT=TIRO_FORMAT, doScale=doScale, doLog=doLog
    )  # Load files datas

    # LFC_Rank.to_csv(outputPath+'/LFC_rank.csv')

    feature_len = len(feature_name)
    class_len = len(label_name)
    print("Number of feature: {}, Number of class: {}".format(feature_len, class_len))

    accuracy_train = np.zeros((nfold * len(Seed), class_len + 1))
    accuracy_test = np.zeros((nfold * len(Seed), class_len + 1))
    data_train = np.zeros((nfold * len(Seed), 7))
    data_test = np.zeros((nfold * len(Seed), 7))
    correct_prediction = []
    s = 0
    for SEED in Seed:

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        for i in range(nfold):

            train_dl, test_dl, train_len, test_len, Ytest = fd.CrossVal(
                X, Y, patient_name, BATCH_SIZE, i, SEED
            )
            print(
                "Len of train set: {}, Len of test set:: {}".format(train_len, test_len)
            )
            print("----------- Début iteration ", i, "----------------")
            # Define the SEED to fix the initial parameters
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)

            # run Classifier
            if net_name == "LeNet":
                net = fd.LeNet_300_100_DNN(
                    n_inputs=feature_len, n_outputs=class_len
                ).to(device)
            elif net_name == "DNN":
                net = fd.DNN(n_inputs=feature_len, n_outputs=class_len).to(
                    device
                )  # basic DNN

            weights_entry, spasity_w_entry = fnp.weights_and_sparsity(net)

            if GRADIENT_MASK:
                run_model = "ProjectionLastEpoch"

            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150, gamma=0.1)
            data_encoder, epoch_loss, best_test, net = fd.RunDNN(
                net,
                criterion_classification,
                optimizer,
                train_dl,
                train_len,
                test_dl,
                test_len,
                N_EPOCHS,
                outputPath,
                DO_PROJ_middle,
                run_model,
                TYPE_PROJ,
                ETA,
                AXIS=AXIS,
            )
            labelpredict = data_encoder[:, :-1].max(1)[1].cpu().numpy()
            # Do masked gradient

            if GRADIENT_MASK:
                print("\n--------Running with masked gradient-----")
                print("-----------------------")
                zero_list = []
                tol = 1.0e-3
                for index, param in enumerate(list(net.parameters())):
                    if index < len(list(net.parameters())) / 2 - 2 and index % 2 == 0:
                        ind_zero = torch.where(torch.abs(param) < tol)
                        zero_list.append(ind_zero)

                # Get initial network and set zeros
                # Recall the SEED to get the initial parameters
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)

                # run network
                if net_name == "LeNet":
                    net = fd.LeNet_300_100_DNN(
                        n_inputs=feature_len, n_outputs=class_len
                    ).to(device)
                elif net_name == "DNN":
                    net = fd.DNN(n_inputs=feature_len, n_outputs=class_len).to(
                        device
                    )  # basic DNN

                optimizer = torch.optim.Adam(net.parameters(), lr=LR)

                for index, param in enumerate(list(net.parameters())):
                    if index < len(list(net.parameters())) / 2 - 2 and index % 2 == 0:
                        param.data[zero_list[int(index / 2)]] = 0

                run_model = "MaskGrad"
                (data_encoder, epoch_loss, best_test, net,) = fd.RunDNN(
                    net,
                    criterion_classification,
                    optimizer,
                    train_dl,
                    train_len,
                    test_dl,
                    test_len,
                    N_EPOCHS_MASKGRAD,
                    outputPath,
                    zero_list,
                    run_model,
                    TYPE_PROJ,
                    ETA,
                    AXIS=AXIS,
                )
                print("\n--------Finised masked gradient-----")
                print("-----------------------")

            data_encoder = data_encoder.cpu().detach().numpy()

            (
                data_encoder_test,
                class_train,
                class_test,
                _,
                correct_pred,
                softmax,
                Ytrue,
                Ypred,
            ) = fd.runBestNet(
                train_dl,
                test_dl,
                best_test,
                outputPath,
                i,
                class_len,
                net,
                feature_name,
                test_len,
            )

            if SEED == Seed[-1]:
                if i == 0:
                    Ytruef = Ytrue
                    Ypredf = Ypred
                    LP_test = data_encoder_test.detach().cpu().numpy()
                else:
                    Ytruef = np.concatenate((Ytruef, Ytrue))
                    Ypredf = np.concatenate((Ypredf, Ypred))
                    LP_test = np.concatenate(
                        (LP_test, data_encoder_test.detach().cpu().numpy())
                    )

            accuracy_train[s * 4 + i] = class_train
            accuracy_test[s * 4 + i] = class_test
            # silhouette score
            X_encoder = data_encoder[:, :-1]
            labels_encoder = data_encoder[:, -1]
            data_encoder_test = data_encoder_test.cpu().detach()

            data_train[s * 4 + i, 0] = metrics.silhouette_score(
                X_encoder, labels_encoder, metric="euclidean"
            )

            X_encodertest = data_encoder_test[:, :-1]
            labels_encodertest = data_encoder_test[:, -1]
            data_test[s * 4 + i, 0] = metrics.silhouette_score(
                X_encodertest, labels_encodertest, metric="euclidean"
            )
            # ARI score

            data_train[s * 4 + i, 1] = metrics.adjusted_rand_score(
                labels_encoder, labelpredict
            )
            data_test[s * 4 + i, 1] = metrics.adjusted_rand_score(
                Ytest, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
            )

            # AMI Score
            data_train[s * 4 + i, 2] = metrics.adjusted_mutual_info_score(
                labels_encoder, labelpredict
            )
            data_test[s * 4 + i, 2] = metrics.adjusted_mutual_info_score(
                Ytest, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
            )

            # UAC Score
            if class_len == 2:
                data_train[s * 4 + i, 3] = metrics.roc_auc_score(
                    labels_encoder, labelpredict
                )
                data_test[s * 4 + i, 3] = metrics.roc_auc_score(
                    Ytest, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
                )

            # F1 precision recal
            data_train[s * 4 + i, 4:] = precision_recall_fscore_support(
                labels_encoder, labelpredict, average="macro"
            )[:-1]
            data_test[s * 4 + i, 4:] = precision_recall_fscore_support(
                Ytest, data_encoder_test[:, :-1].max(1)[1].numpy(), average="macro"
            )[:-1]

            # Recupération des labels corects
            correct_prediction += correct_pred

            # Get Top Genes of each class

            #         method = 'Shap'       # (SHapley Additive exPlanation) A nb_samples should be define
            nb_samples = 300  # Randomly choose nb_samples to calculate their Shap Value, time vs nb_samples seems exponential
            #        method = 'Captum_ig'   # Integrated Gradients
            method = "Captum_dl"  # Deeplift
            #        method = 'Captum_gs'  # GradientShap

            if DoTopGenes:
                tps1 = time.perf_counter()
                if i == 0:  # first fold, we never did topgenes before
                    print("Running topGenes...")
                    df_topGenes = fd.topGenes(
                        X,
                        Y,
                        feature_name,
                        class_len,
                        feature_len,
                        method,
                        nb_samples,
                        device,
                        net,
                    )
                    df_topGenes.index = df_topGenes.iloc[:, 0]
                    print("topGenes finished")
                    tps2 = time.perf_counter()
                else:
                    print("Running topGenes...")
                    df_topGenes = fd.topGenes(
                        X,
                        Y,
                        feature_name,
                        class_len,
                        feature_len,
                        method,
                        nb_samples,
                        device,
                        net,
                    )
                    print("topGenes finished")
                    df = pd.read_csv(
                        "{}{}_topGenes_{}_{}.csv".format(
                            outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                        ),
                        sep=";",
                        header=0,
                        index_col=0,
                    )
                    df_topGenes.index = df_topGenes.iloc[:, 0]
                    df_topGenes = df.join(df_topGenes.iloc[:, 1], lsuffix="_",)

                df_topGenes.to_csv(
                    "{}{}_topGenes_{}_{}.csv".format(
                        outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                    ),
                    sep=";",
                )
                tps2 = time.perf_counter()
                print("execution time topGenes  : ", tps2 - tps1)

        if SEED == Seed[0]:
            df_softmax = softmax
            df_softmax.index = df_softmax["Name"]
            # softmax.to_csv('{}softmax.csv'.format(outputPath),sep=';',index=0)
        else:
            softmax.index = softmax["Name"]
            df_softmax = df_softmax.join(softmax, rsuffix="_")

        # Moyenne sur les SEED
        if DoTopGenes:
            df = pd.read_csv(
                "{}{}_topGenes_{}_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                ),
                sep=";",
                header=0,
                index_col=0,
            )
            df_val = df.values[1:, 1:].astype(float)
            df_mean = df_val.mean(axis=1).reshape(-1, 1)
            df_std = df_val.std(axis=1).reshape(-1, 1)
            df = pd.DataFrame(
                np.concatenate((df.values[1:, :], df_mean, df_std), axis=1),
                columns=[
                    "Features",
                    "Fold 1",
                    "Fold 2",
                    "Fold 3",
                    "Fold 4",
                    "Mean",
                    "Std",
                ],
            )
            df_topGenes = df
            df_topGenes = df_topGenes.sort_values(by="Mean", ascending=False)
            df_topGenes = df_topGenes.reindex(
                columns=[
                    "Features",
                    "Mean",
                    "Fold 1",
                    "Fold 2",
                    "Fold 3",
                    "Fold 4",
                    "Std",
                ]
            )
            df_topGenes.to_csv(
                "{}{}_topGenes_{}_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                ),
                sep=";",
                index=0,
            )

            if SEED == Seed[0]:
                df_topGenes_mean = df_topGenes.iloc[:, 0:2]
                df_topGenes_mean.index = df_topGenes.iloc[:, 0]
            else:
                df = pd.read_csv(
                    "{}{}_topGenes_Mean_{}_{}.csv".format(
                        outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                    ),
                    sep=";",
                    header=0,
                    index_col=0,
                )
                df_topGenes.index = df_topGenes.iloc[:, 0]
                df_topGenes_mean = df.join(df_topGenes.iloc[:, 1], lsuffix="_",)

            df_topGenes_mean.to_csv(
                "{}{}_topGenes_Mean_{}_{}.csv".format(
                    outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                ),
                sep=";",
            )

        s += 1

        # accuracies
    df_accTrain, df_acctest = fd.packClassResult(
        accuracy_train, accuracy_test, nfold * len(Seed), label_name
    )
    print("\nAccuracy Train")
    print(df_accTrain)
    print("\nAccuracy Test")
    print(df_acctest)

    # metrics
    df_metricsTrain, df_metricsTest = fd.packMetricsResult(
        data_train, data_test, nfold * len(Seed)
    )

    # separation of the metrics in different dataframes
    clustering_metrics = ["Silhouette", "ARI", "AMI"]
    classification_metrics = ["AUC", "Precision", "Recall", "F1 score"]
    df_metricsTrain_clustering = df_metricsTrain[clustering_metrics]
    df_metricsTrain_classif = df_metricsTrain[classification_metrics]
    df_metricsTest_clustering = df_metricsTest[clustering_metrics]
    df_metricsTest_classif = df_metricsTest[classification_metrics]

    print("\nMetrics Train")
    # print(df_metricsTrain_clustering)
    print(df_metricsTrain_classif)
    print("\nMetrics Test")
    # print(df_metricsTest_clustering)
    print(df_metricsTest_classif)

    # Reconstruction by using the centers in laten space and datas after interpellation
    center_mean, center_distance = fd.Reconstruction(0.2, data_encoder, net, class_len)

    # Do pca,tSNE for encoder data
    if Do_pca and Do_tSNE:
        tit = "Latent Space"
        fd.ShowPcaTsne(X, Y, data_encoder, center_distance, class_len, tit)

    if DoTopGenes:
        df = pd.read_csv(
            "{}{}_topGenes_Mean_{}_{}.csv".format(
                outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
            ),
            sep=";",
            header=0,
            index_col=0,
        )
        df_val = df.values[:, 1:].astype(float)
        df_mean = df_val.mean(axis=1).reshape(-1, 1)
        df_std = df_val.std(axis=1).reshape(-1, 1)
        df_meanstd = df_std / df_mean
        col_seed = ["Seed " + str(i) for i in Seed]
        df = pd.DataFrame(
            np.concatenate((df.values[:, :], df_mean, df_std, df_meanstd), axis=1),
            columns=["Features"] + col_seed + ["Mean", "Std", "Mstd"],
        )
        df_topGenes = df
        df_topGenes = df_topGenes.sort_values(by="Mean", ascending=False)
        df_topGenes = df_topGenes.reindex(
            columns=["Features", "Mean"] + col_seed + ["Std", "Mstd"]
        )
        df_topGenes.to_csv(
            "{}{}_topGenes_Mean_{}_{}.csv".format(
                outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
            ),
            sep=";",
            index=0,
        )

    plt.figure()
    plt.title("Kernel Density")
    plt.plot([0.5, 0.5, 0.5], [-1, 0, 3])
    lab = 0
    for col in softmax.iloc[:, 2:]:
        distrib = softmax[col].where(softmax["Labels"] == lab).dropna()
        if lab == 0:
            sns.kdeplot(1 - distrib, bw=0.1, shade=True, color="tab:blue")
            # sns.kdeplot(1 - distrib, bw=0.1, fill=True, shade="True")
        else:
            sns.kdeplot(distrib, bw=0.1, shade=True, color="tab:orange")
            # sns.kdeplot(distrib, bw=0.1, fill=True, shade="True")

        lab += 1

    spasity_percentage_entry = {}
    for keys in spasity_w_entry.keys():
        spasity_percentage_entry[keys] = spasity_w_entry[keys] * 100
    print("spasity % of all layers entry \n", spasity_percentage_entry)
    print("-----------------------")
    weights, spasity_w = fnp.weights_and_sparsity(net.encoder)
    spasity_percentage = {}
    for keys in spasity_w.keys():
        spasity_percentage[keys] = spasity_w[keys] * 100
    print("spasity % of all layers \n", spasity_percentage)
    print("-----------------------")

    mat_in = net.state_dict()["encoder.0.weight"]

    mat_col_sparsity = fd.sparsity_col(mat_in, device=device)
    print(" Colonnes sparsity sur la matrice d'entrée: \n", mat_col_sparsity)
    mat_in_sparsity = fd.sparsity_line(mat_in, device=device)
    print(" ligne sparsity sur la matrice d'entrée: \n", mat_in_sparsity)
    layer_list = [x for x in weights.values()]
    titile_list = [x for x in spasity_w.keys()]
    fd.show_img(layer_list, file_name)

    # Loss figure
    if os.path.exists(file_name.split(".")[0] + "_Loss_No_proj.npy") and os.path.exists(
        file_name.split(".")[0] + "_Loss_MaskGrad.npy"
    ):
        loss_no_proj = np.load(file_name.split(".")[0] + "_Loss_No_proj.npy")
        loss_with_proj = np.load(file_name.split(".")[0] + "_Loss_MaskGrad.npy")
        plt.figure()
        plt.title(file_name.split(".")[0] + " Loss")
        plt.xlabel("Epoch")
        plt.ylabel("TotalLoss")
        plt.plot(loss_no_proj, label="No projection")
        plt.plot(loss_with_proj, label="With projection ")
        plt.legend()
        plt.show()
    if SAVE_FILE:
        df_acctest.to_csv(
            "{}{}_acctest.csv".format(outputPath, str(TYPE_PROJ_NAME)), sep=";"
        )

        df_metricsTest_classif.to_csv(
            "{}{}_auctest.csv".format(outputPath, str(TYPE_PROJ_NAME)), sep=";"
        )

        print("Save topGenes results to: ' {} ' ".format(outputPath))


# %%
