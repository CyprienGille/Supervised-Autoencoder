# Written by i3s
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def proj_l1ball(y, eta):
    """
    Note that the y should be better 1D or after some element-wise operation, the results will turn to be un predictable.
    This function will automatically reshape the y as (m,), where m is the y.size, or the y.shape[0]*y.shape[1].
    """
    if type(y) is not np.ndarray:
        y = np.array(y)
    if y.ndim > 1:
        y = np.reshape(y, (-1,))
    return np.maximum(
        np.absolute(y)
        - np.amax(
            [
                np.amax(
                    (np.cumsum(np.sort(np.absolute(y), axis=0)[::-1], axis=0) - eta)
                    / (np.arange(y.shape[0]) + 1)
                ),
                0,
            ]
        ),
        0,
    ) * np.sign(y)


def drop_cells(X, Y, n_fold):
    """
    # ====================================================================
    # This function will detect whether the size of the first dimension of
    # X is divisible by n_fold. If not, it will remove the n_diff rows from
    # the biggest class(with the largest size in Y) where n_diff=len(Y)%n_fold
    #
    # ---- Input
    # X         : The data
    # Y         : The label
    # n_fold    : The number of fold
    # --- Output
    # X_new, Y_new : The new data and the new label
    # =====================================================================
    """
    m, d = X.shape
    if m % n_fold == 0:
        return X, Y
    n_diff = m % n_fold
    # choose in the biggest class to delete
    #  Find the biggest class
    lst_count = []
    for i in np.unique(Y):
        lst_count.append(np.where(Y == i, 1, 0).sum())
    ind_max = np.unique(Y)[np.argmax(lst_count)]
    lst_inds = np.where(Y == ind_max)[0]

    # Delete n_diff elements in the biggest class
    lst_del = np.random.choice(lst_inds, n_diff)
    X_new = np.delete(X, lst_del, 0)
    Y_new = np.delete(Y, lst_del, 0)
    return X_new, Y_new


def getPredLabel(Ypred):
    for i in range(Ypred.shape[0]):
        if Ypred[i] > 1.5:
            Ypred[i] = 2
        if Ypred[i] <= 1.5:
            Ypred[i] = 1
    return Ypred


def rankFeatureHelper(alg, coef, feature_names):
    df1 = pd.DataFrame(feature_names, columns=[alg])
    df1["weights"] = coef
    df1["sort_helper"] = df1["weights"].abs()
    df2 = df1.sort_values(by="sort_helper", ascending=False).drop("sort_helper", axis=1)
    return df2


import warnings


def readData(filename):
    DATADIR = "data/"
    df_X = pd.read_csv(
        DATADIR + str(filename),
        delimiter=";",
        decimal=",",
        header=0,
        encoding="ISO-8859-1",
        low_memory=False,
    )

    df_names = df_X["Name"]
    feature_names = df_names[1:].values.astype(str)
    X = df_X.iloc[1:, 1:].values.astype(float).transpose()
    Yr = df_X.iloc[0, 1:].values.astype(float)
    nbr_clusters = len(np.unique(Yr))
    feature_names = df_names.values.astype(str)[1:]
    label_name = df_names

    for index, label in enumerate(
        label_name
    ):  # convert string labels to numero (0,1,2....)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            Yr = np.where(Yr == label, index, Yr)
    Yr = Yr.astype(np.int64)
    if not Yr.all():  # if labels are 0/1 instead of 1/2
        Yr += 1
    return X, Yr, nbr_clusters, feature_names


def basic_run_other(
    X,
    YR,
    k,
    alglist,
    genenames=None,
    clusternames=None,
    nfold=4,
    rng=6,
    doTopGenes=False,
):

    np.random.seed(rng)  # reproducible

    _, d = X.shape

    # parameter checking
    if genenames is None:
        genenames = ["Gene {}".format(i + 1) for i in range(d)]
    if clusternames is None:
        clusternames = ["Class {}".format(i + 1) for i in range(k)]

    if YR.ndim == 1:  # In case that OneHotEncoder get 1D array and raise a TypeError
        YR = YR.reshape(-1, 1)

    # Dropping the cells randomly if the n%d is not zero
    # For more details please see instructions in drop_cells
    X, YR = drop_cells(X, YR, nfold)

    # Initialization

    kf = KFold(n_splits=nfold, random_state=rng, shuffle=True)

    numalg = len(alglist)
    accuracy_train_comp = np.zeros((nfold, numalg))
    accuracy_test_comp = np.zeros((nfold, numalg))
    AUC_train_comp = np.zeros((nfold, numalg * 4))
    AUC_test_comp = np.zeros((nfold, numalg * 4))
    timeElapsedMatrix = np.zeros((nfold, numalg))
    top_features_list = []

    # 4-flod cross validation
    for i, (train_ind, test_ind) in enumerate(kf.split(YR)):
        print("{:-<30}".format(""))
        print("{message:^6} {f1} / {f2}".format(message="fold", f1=i + 1, f2=nfold))

        # ========== Training =========
        Xtrain = X[train_ind]
        Xtest = X[test_ind]
        Ytrain = YR[train_ind]
        Ytest = YR[test_ind]

        # start loop of other algorithms comparison

        for j, alg in enumerate(alglist):
            get_features = lambda m: None  # default get_features, will be overriden
            if alg == "svm":

                tuned_parameters = [
                    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
                ]
                clf = GridSearchCV(SVC(), tuned_parameters)
                get_features = lambda m: m.best_estimator_.coef_.transpose()

            if alg == "RF":
                clf = RandomForestClassifier(n_estimators=100)
                get_features = lambda m: m.feature_importances_

            if alg == "plsda":
                clf = PLSRegression(n_components=4, scale=False)
                get_features = lambda m: m.coef_

            if alg == "logreg":
                clf = LogisticRegression(C=10)
                get_features = lambda m: m.coef_.transpose()

            if alg == "NN":
                clf = KNeighborsClassifier(n_neighbors=50)

            if alg == "GaussianNB":
                clf = GaussianNB(var_smoothing=1e-9)  # var smoothing to be tuned

            if alg == "Adaboost":
                clf = AdaBoostClassifier(n_estimators=100)  # parameters to be tuned
                get_features = lambda m: m.feature_importances_

            if alg == "Lasso":
                lasso = Lasso(random_state=0, max_iter=10000)
                alphas = np.logspace(-4, -0.5, 20)
                tuned_parameters = [{"alpha": alphas}]
                n_folds = 5
                clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds)
                get_features = lambda m: m.best_estimator_.coef_

            # build the model
            startTime = time.perf_counter()
            model = clf.fit(Xtrain, Ytrain.ravel())
            endTime = time.perf_counter()
            timeElapsedMatrix[i][j] = endTime - startTime

            if k > 2:
                Ypred_test = np.around(
                    model.predict(Xtest)
                ).ravel()  # getPredLabel(model.predict(Xtest))
                Ypred_train = np.around(
                    model.predict(Xtrain)
                ).ravel()  # getPredLabel(model.predict(Xtrain))
            else:
                Ypred_test = getPredLabel(model.predict(Xtest)).ravel()
                Ypred_train = getPredLabel(model.predict(Xtrain)).ravel()

            accuracy_test_comp[i][j] = accuracy_score(Ypred_test.astype("int64"), Ytest)
            accuracy_train_comp[i][j] = accuracy_score(
                Ypred_train.astype("int64"), Ytrain
            )

            if (
                np.unique(Ytest).shape[0] == 2
                and np.unique(Ypred_test.astype("int64")).shape[0] == 2
            ):
                AUC_test_comp[i][j * 4] = roc_auc_score(
                    Ypred_test.astype("int64"), Ytest
                )
                AUC_train_comp[i][j * 4] = roc_auc_score(
                    Ypred_train.astype("int64"), Ytrain
                )

            # F1 precision recall
            # Note: for some models, these are not defined
            # (for example, the Lasso)
            # In those cases, the undefined scores are set to 0,
            # And no warning is raised
            # Cf. the zero_division=0 parameter.
            AUC_train_comp[i][
                j * 4 + 1 : j * 4 + 4
            ] = metrics.precision_recall_fscore_support(
                Ytrain, Ypred_train.astype("int64"), average="macro", zero_division=0
            )[
                :-1
            ]

            AUC_test_comp[i][
                j * 4 + 1 : j * 4 + 4
            ] = metrics.precision_recall_fscore_support(
                Ytest, Ypred_test.astype("int64"), average="macro", zero_division=0
            )[
                :-1
            ]

            # get the topgenes from the first fold
            if i == 0 and doTopGenes:
                coef = get_features(clf)
                if coef is not None:
                    df_rankFeature = rankFeatureHelper(alg, coef, genenames)
                else:
                    df_rankFeature = rankFeatureHelper(
                        alg, [0] * len(genenames), genenames
                    )
                top_features_list.append(df_rankFeature)

    # Data accuracy1
    ind_df_comp = []
    for i_fold in range(nfold):
        ind_df_comp.append("Fold {}".format(i_fold + 1))

    df_comp = pd.DataFrame(accuracy_test_comp, index=ind_df_comp, columns=alglist)
    df_comp.loc["Mean"] = df_comp.mean()

    colauc = []
    for met in alglist:
        colauc.append(met + " AUC")
        colauc.append(met + " Precision")
        colauc.append(met + " Recall")
        colauc.append(met + " F1 score")
    df_compauc = pd.DataFrame(AUC_test_comp, index=ind_df_comp, columns=colauc)
    df_compauc.loc["Mean"] = df_compauc.mean()

    alglen = len(alglist)
    alglist1 = []
    for i in range(alglen):
        alglist1.append(alglist[i])
    df_timeElapsed = pd.DataFrame(
        timeElapsedMatrix, index=ind_df_comp, columns=alglist1
    )
    df_timeElapsed.loc["Mean"] = df_timeElapsed.mean()

    return df_comp, df_timeElapsed, df_compauc, top_features_list


# =============================================================================
if __name__ == "__main__":

    print("This is just a storage file for functions. So... nothing happened.")
