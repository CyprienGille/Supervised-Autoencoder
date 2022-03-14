import torch
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import time
from sklearn.model_selection import KFold

import matplotlib as mpl
from torch import nn

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import shap
except ImportError:
    print("Use '!pip install shap' to install shap")

try:
    import captum
except ImportError:
    print("Use '!pip install captum' to install captum")

from captum.attr import (
    GradientShap,
    DeepLift,
    IntegratedGradients,
)

# ===========================================================================
# Basic functions
# ===========================================================================


def proj_l1ball(w0, eta, device="cpu"):
    # To help you understand, this function will perform as follow:
    #    a1 = torch.cumsum(torch.sort(torch.abs(y),dim = 0,descending=True)[0],dim=0)
    #    a2 = (a1 - eta)/(torch.arange(start=1,end=y.shape[0]+1))
    #    a3 = torch.abs(y)- torch.max(torch.cat((a2,torch.tensor([0.0]))))
    #    a4 = torch.max(a3,torch.zeros_like(y))
    #    a5 = a4*torch.sign(y)
    #    return a5

    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    init_shape = w.size()

    if w.dim() > 1:
        init_shape = w.size()
        w = w.reshape(-1)

    Res = torch.sign(w) * torch.max(
        torch.abs(w)
        - torch.max(
            torch.cat(
                (
                    (
                        torch.cumsum(
                            torch.sort(torch.abs(w), dim=0, descending=True)[0],
                            dim=0,
                            dtype=torch.get_default_dtype(),
                        )
                        - eta
                    )
                    / torch.arange(
                        start=1,
                        end=w.numel() + 1,
                        device=device,
                        dtype=torch.get_default_dtype(),
                    ),
                    torch.tensor([0.0], dtype=torch.get_default_dtype(), device=device),
                )
            )
        ),
        torch.zeros_like(w),
    )

    Q = Res.reshape(init_shape).clone().detach()

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()
    return Q


def proj_l21ball(w0, eta, axis=1, device="cpu"):

    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)
    init_shape = w.size()

    x = torch.as_tensor(w, dtype=torch.get_default_dtype(), device=device)

    if axis is None:
        axis = tuple(range(x.dim()))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception:
            raise TypeError("'axis' must be None, an integer or a tuple of integers")

    if axis > x.dim() - 1:
        axis = x.dim() - 1
    Y = torch.norm(x, 2, dim=axis)
    T = proj_l1ball(Y, eta, device=device).reshape(Y.shape)
    max_TY = torch.max(T, Y)
    x0 = torch.where(max_TY == 0, torch.zeros_like(T), torch.div(T, max_TY))
    if axis == 0:
        x = torch.mul(x, x0)
    else:
        order = tuple(np.arange(x.dim()))
        new_order = (order[axis],) + order[:axis] + order[axis + 1 :]
        reverse_order = (order[axis],) + (0,) + order[axis + 1 :]
        x = torch.mul(x.permute(new_order), x0)
        x = x.permute(reverse_order)

    Q = x.reshape(init_shape).clone().detach().requires_grad_(True)

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()

    return Q


## fold in ["local","full",partial"]
def proj_nuclear(w0, eta_star, fold="local", device="cpu"):

    w1 = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)
    init_shape = w1.size()

    if fold == "full":
        w = full_fold_conv(w0)
    elif fold == "partial":
        w = partial_fold_conv(w0)
    else:
        w = w1

    if w.dim() == 1:
        v = proj_l1ball(w, eta_star, device=device)
    elif w.dim() == 2:
        L, S0, R = torch.svd(w, some=True)  #'economy-size decomposition'
        # norm_nuclear = S0.sum().item() # Note that the S will be a vector but not a diagonal matrix
        v_star = proj_l1ball(S0, eta_star, device=S0.device)
        v = torch.matmul(L, torch.matmul(v_star.diag(), R.t()))
    elif w.dim() > 2:  # occurs only in the case of local folding
        L, S0, R = np.linalg.svd(w.data.numpy(), full_matrices=False)
        # norm_nuclear = S0.sum()
        v_star = proj_l1ball(S0.reshape((-1,)), eta_star, device=device)
        S1 = v_star.reshape(S0.shape)
        v_temp = np.matmul(L, S1[..., None] * R)
        v = torch.as_tensor(v_temp, device=device)

    if fold == "full":
        v = full_unfold_conv(v, init_shape)
    elif fold == "partial":
        v = partial_unfold_conv(v, init_shape)

    Q = v.reshape(init_shape).clone().detach().requires_grad_(True)

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()

    return Q


def proj_l11ball(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l1ball(w[:, i], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def proj_l11ball_line(w2, eta, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[i, :])).data.item() for i in range(nrow)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(nrow):
            Res[i, :] = proj_l1ball(w[i, :], PW[i].data.item(), device=device)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()

    return Q


def proj_l12ball(V, eta, axis=1, threshold=0.001, device="cpu"):

    V = torch.as_tensor(V, dtype=torch.get_default_dtype(), device=device)

    tol = 0.001
    lst_f = []
    test = eta * eta

    if V.dim() == 1:
        return proj_l1ball(V, eta, device=device)

    if axis == 0:
        V = V.T
    Vshape = V.shape
    # m,d = Vshape
    lmbda = 0.0
    p = np.ones(Vshape[0], dtype=int) * (Vshape[1] - 1)  # to change in case of tensor
    delta = np.zeros(Vshape[0])
    V_abs = np.abs(V)  # maybe transposed if change the value of axis
    sgn = np.sign(V)
    #    V0 = np.sort(V_abs,axis=1)[:,::-1]
    #    V_sum = np.cumsum(V0,axis=1)
    V_sum = np.cumsum(np.sort(V_abs, axis=1)[:, ::-1], axis=1)

    q = np.arange(0, Vshape[1])
    sum_q = np.power(np.array([V_sum[:, qi] for qi in q]), 2)
    sum_q = np.sqrt(sum_q.sum(axis=1))
    lmbda_init = np.max((sum_q / eta - 1) / (q + 1))
    lmbda = lmbda_init
    # lmbda=0
    p = np.argmax(V_sum / (1 + lmbda * np.arange(1, Vshape[1] + 1)), axis=1)

    while np.abs(test) > tol:
        # update lambda
        sum0 = np.array(list(map(lambda x, y: y[x], p, V_sum)))
        sum1 = np.sum(np.power(sum0 / (1 + lmbda * p), 2))
        sum2 = np.sum(p * (np.power(sum0, 2) / np.power(1 + lmbda * p, 3)))
        test = sum1 - eta * eta
        lmbda = lmbda + test / (2 * sum2)
        lst_f.append(test)
        # update p
        p = np.argmax(V_sum / (1 + lmbda * np.arange(1, Vshape[1] + 1)), axis=1)

    delta = lmbda * (np.array(list(map(lambda x, y: y[x], p, V_sum))) / (1 + lmbda * p))
    W = V_abs - delta.reshape((-1, 1))
    W[W < 0] = 0
    W = W * sgn
    W[np.where(np.abs(W) < threshold)] = 0
    if axis == 0:
        W = W.T

    return W.float()


def full_fold_conv(M):

    if M.dim() > 2:
        M2 = M.clone().detach()
        init_shape = M2.shape

        row, col = init_shape[0:2]
        N = list(M2.reshape(-1).size())[0]

        Q = torch.transpose(torch.transpose(M2, 0, 1).reshape(N).reshape(col, -1), 0, 1)
    else:
        Q = M

    return Q


def full_unfold_conv(M, original_shape):

    if len(list(original_shape)) > 2:
        M2 = M.clone().detach()
        init_shape = original_shape

        inverse_shape = [init_shape[1], init_shape[0]]

        if len(list(init_shape)) > 2:
            last_shape = list(init_shape[2:])
            inverse_shape = inverse_shape + last_shape

        inverse_shape = tuple(inverse_shape)

        row, col = init_shape[0:2]
        N = list(M2.reshape(-1).size())[0]

        Q = torch.transpose(
            torch.transpose(M2, 0, 1).reshape(N).reshape(inverse_shape), 0, 1
        )
    else:
        Q = M

    return Q


def partial_fold_conv(M):

    if M.dim() > 2:
        M2 = M.clone().detach()
        init_shape = list(M2.shape)

        L = len(init_shape)

        Q = torch.cat(
            tuple(
                [
                    torch.cat(tuple([M2[i, j] for j in range(init_shape[1])]), 1)
                    for i in range(init_shape[0])
                ]
            ),
            0,
        )
    else:
        Q = M

    return Q


def partial_unfold_conv(M, original_shape):

    if len(list(original_shape)) > 2:
        M2 = M.clone().detach()
        init_shape = list(original_shape)

        Z = torch.empty(original_shape)

        for i in range(init_shape[0]):
            for j in range(init_shape[1]):
                di = init_shape[2]
                dj = init_shape[3]
                Z[i, j] = M2[
                    i * di : (i * di + init_shape[2]), j * dj : (j * dj + init_shape[3])
                ]
            # print('row: {}-{}, col: {}-{}'.format(i*di,(i*di+init_shape[2]),j*dj,(j*dj+init_shape[3])))
    else:
        Z = M
    return Z


def sort_weighted_projection(y, eta, w, n=None, device="cpu"):
    if type(y) is not torch.Tensor:
        y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    if type(w) is not torch.Tensor:
        w = torch.as_tensor(w, dtype=torch.get_default_dtype())
    if y.dim() > 1:
        y = y.view(-1)
    if w.dim() > 1:
        w = w.view(-1)
    if device is not None and "cuda" in device:
        y = y.cuda()
        w = w.cuda()
    elif y.is_cuda:
        w = w.cuda()
    elif w.is_cuda:
        y = y.cuda()
    if any(w < 0):
        raise ValueError("sort_weighted_projection: The weight should be positive")
    y0 = y * torch.sign(y)
    w = w.type(dtype=y.dtype)
    y0 = y0.type(dtype=y.dtype)
    x = torch.zeros_like(y)
    if n is None:
        n = len(x)
    z = torch.div(y0, w)
    p = torch.argsort(z, descending=True)
    WYs = 0.0
    Ws = 0.0
    for j in p:
        WYs += w[j] * y0[j]
        Ws += w[j] * w[j]
        if ((WYs - eta) / Ws) > z[j]:
            break
    WYs -= w[j] * y0[j]
    Ws -= w[j] * w[j]
    L = (WYs - eta) / Ws
    if n == len(x):
        x = torch.max(torch.zeros_like(y), y0 - w * L)
    else:
        for i in range(n):
            x[i] = max(torch.zeros_like(y), y0[i] - w[i] * L)
    x *= torch.sign(y)
    return x


def sort_weighted_proj(y, eta, w, n=None, device="cpu"):
    """
    Weighted projection on l1 ball. V2
    When w = ones(y.shape) this function is equivalent to proj_l1ball(y,eta)
    It's not a simple 
    Instead of using for loop, we use np.cumsum
    """
    if type(y) is not torch.Tensor:
        y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    if type(w) is not torch.Tensor:
        w = torch.as_tensor(w, dtype=torch.get_default_dtype())
    if y.dim() > 1:
        y = y.view(-1)
    if w.dim() > 1:
        w = w.view(-1)
    if device is not None and "cuda" in device:
        y = y.cuda()
        w = w.cuda()
    elif y.is_cuda:
        w = w.cuda()
    elif w.is_cuda:
        y = y.cuda()
    if any(w < 0):
        raise ValueError("sort_weighted_projection: The weight should be positive")

    y0 = y * torch.sign(y)
    w = w.type(dtype=y.dtype)
    y0 = y0.type(dtype=y.dtype)
    x = torch.zeros_like(y)
    if n is None:
        n = len(x)
    z = torch.div(y0, w)
    p = torch.argsort(z, descending=True)
    yp = y0[p]
    wp = w[p]
    Wys = torch.cumsum(yp * wp, dim=0)
    Ws = torch.cumsum(wp * wp, dim=0)
    L = torch.div((Wys - eta), Ws)
    ind = (L > z[p]).nonzero()
    if len(ind) == 0:  # all elements of (Wys -eta) /Ws are <z[p], so L is the last one
        L = L[-1]
    else:
        L = L[ind[0]]
    if n == len(x):
        x = torch.max(torch.zeros_like(y), y0 - w * L)
    else:
        x[0:n] = torch.max(torch.zeros_like(y), y0[0:n] - w[0:n] * L)
    x *= torch.sign(y)
    return x


def centroids(XW, Y, k, device="cpu"):
    """
    ==========================================================================
    Return the number of selected genes from the matrix w
    #----- INPUT
        XW              : (Tensor) X*W
        Y               : (Tensor) the labels
        k               : (Scaler) number of clusters
    #----- OUTPUT
        mu              : (Tensor) the centroids of each cluster
    ===========================================================================   
    """
    Y = Y.view(-1)
    d = XW.shape[1]
    mu = torch.zeros((k, d), device=device)
    """
    since in python the index starts from 0 not from 1, 
    here the Y==i will be change to Y==(i+1)
    Or the values in Y need to be changed
    """
    for i in range(k):
        C = XW[Y == (i + 1), :]
        mu[i, :] = torch.mean(C, dim=0)
    return mu


def nb_Genes(w, device="cpu"):
    """
    ==========================================================================  \n
    Return the number of selected genes from the matrix w                       \n
    #----- INPUT                                                                \n
        w               : (Tensor) weight matrix                                \n
    #----- OUTPUT                                                               \n
        nbG             : (Scalar) the number of genes                          \n
        indGene_w       : (array) the index of the genes                        \n
    ===========================================================================   
    """
    #
    d = w.shape[0]
    ind_genes = torch.zeros((d, 1), device="cpu")

    for i in range(d):
        if torch.norm(w[i, :]) > 0:
            ind_genes[i] = 1

    indGene_w = (ind_genes == 1).nonzero()[:, 0]
    nbG = ind_genes.sum().int()
    return nbG, indGene_w.numpy()


def select_feature_w(w, featurenames, device="cpu"):
    """
    ==========================================================================
    #----- INPUT
        w               : (Tensor) weight matrix
        featurenames    : (List or array of Tensor) the feature names 
    #----- OUTPUT
        features        : (array) the chosen features
        normW           : (array) the norm of the features
    
    Note that for torch algorithm, the w will be torch.Tensor()
    while the featurenames is usually of type array,
    so this function will take w as type torch.Tensor and perform exactly 
    the same way of that in functions.py and it will produce then the same
    output, even the with same type.
    
    Simply speaking, this function will perform the same way as select_feature_w
    in functions.py exccept that the input w will be of type torch.Tensor
    ===========================================================================   
    """
    if type(w) is not torch.Tensor:
        w = torch.as_tensor(w, device=device)
    d, k = w.shape
    lst_features = []
    lst_norm = []

    for i in range(k):
        s_tmp = w[:, i]
        f_tmp = torch.abs(s_tmp)
        f_tmp, ind = torch.sort(f_tmp, descending=True)
        nonzero_inds = torch.nonzero(f_tmp)
        lst_f = []
        lst_n = []
        if len(nonzero_inds) > 0:
            nozero_ind = nonzero_inds[-1]  # choose the last nonzero index
            if nozero_ind == 0:
                lst_f.append(featurenames[ind[0]])
                lst_n.append(s_tmp[ind[0]])
            else:
                for j in range(nozero_ind + 1):
                    lst_f.append(featurenames[ind[j]])
                    lst_n = s_tmp[ind[0 : (nozero_ind + 1)]]
        lst_features.append(lst_f)
        lst_norm.append(lst_n)

    # Then the last part is exactly the same with orginal function
    n_cols_f = len(lst_features)
    n_rows_f = max(map(len, lst_features))
    n_cols_n = len(lst_norm)
    n_rows_n = max(map(len, lst_norm))

    for i in range(n_cols_f):
        ft = np.array(lst_features[i])
        ft.resize(n_rows_f, refcheck=False)
        nt = np.array(lst_norm[i])
        nt.resize(n_rows_n, refcheck=False)
        if i == 0:
            features = ft
            normW = nt
            continue
        features = np.vstack((features, ft))
        normW = np.vstack((normW, nt))
    features = features.T
    normW = normW.T
    return features, normW


def compute_accuracy(idxR, idx, k, device="cpu"):
    """
    =================================               \n
    #----- INPUT                                    \n
      idxR : (Tensor) real labels                   \n
      idx  : (Tensor) estimated labels              \n
      k    : (Scalar) number of class               \n
    #----- OUTPUT                                   \n
      ACC_glob : (Scalar) global accuracy           \n
      tab_acc  : (Numpy.array) accuracy per class   \n
    =================================
    """
    if type(idxR) is not torch.Tensor:
        idxR = torch.as_tensor(idxR, dtype=torch.int, device=device)
    else:
        idxR = idxR.int()
    if type(idx) is not torch.Tensor:
        idx = torch.as_tensor(idx, dtype=torch.int, device=device)
    else:
        idx = idx.int()

    # Global accuracy
    y = (idx == idxR).nonzero()[:, 0].numel()
    ACC_glob = y / idxR.numel()

    # Accuracy per class
    tab_acc = torch.zeros((1, k), device=device)
    """
    since in python the index starts from 0 not from 1, 
    here the idx(ind)==j in matlab will be change to idx[ind]==(j+1)
    """
    for j in range(k):
        ind = (idxR == (j + 1)).nonzero()
        if len(ind) == 0:
            tab_acc[0, j] = 0.0
        else:
            tab_acc[0, j] = (idx[ind] == (j + 1)).nonzero()[:, 0].numel() / ind.numel()
    return ACC_glob, tab_acc.numpy()


def predict_L1(Xtest, W, mu, device="cpu"):
    """
    =================================           \n
    #----- INPUT                                \n
      Xtest : (Tensor) the data                 \n
      w     : (Tensor) the weight matrix        \n
      mu    : (Tensor) the centroids            \n
    #----- OUTPUT                               \n
      Ytest : (Tensor) the predictions          \n
    =================================
    """
    if type(Xtest) is not torch.Tensor:
        Xtest = torch.as_tensor(Xtest, device=device)
    if type(W) is not torch.Tensor:
        W = torch.as_tensor(W, device=device)
    if type(mu) is not torch.Tensor:
        mu = torch.as_tensor(mu, device=device)
    # Chambolle_Predict
    k = mu.shape[0]
    m = Xtest.shape[0]
    Ytest = torch.zeros((m, 1), device=device)

    for i in range(m):
        distmu = torch.zeros((1, k), device=device)
        XWi = torch.matmul(Xtest[i, :], W)
        for j in range(k):
            distmu[0, j] = torch.norm(XWi - mu[j, :], p=1)
        # Not like np.argmin, the torch.argmin returns the last index of the
        # minimum not the first one, we will then get the latter manually
        Ytest[i] = (distmu == distmu.min()).nonzero()[0, 1] + 1
    return Ytest


def sparsity(M, tol=1.0e-3, device="cpu"):
    """
    ==========================================================================  \n
    Return the spacity for the input matrix M                                   \n
    ----- INPUT                                                                 \n
        M               : (Tensor) the matrix                                   \n
        tol             : (Scalar,optional) the threshold to select zeros       \n
    ----- OUTPUT                                                                \n
        spacity         : (Scalar) the spacity of the matrix                    \n
    ===========================================================================   
    """
    if type(M) is not torch.Tensor:
        M = torch.as_tensor(M, device=device)
    M1 = torch.where(torch.abs(M) < tol, torch.zeros_like(M), M)
    nb_nonzero = len(M1.nonzero())
    return 1.0 - nb_nonzero / M1.numel()


class LoadDataset(torch.utils.data.Dataset):
    """Load data in Pytorch 

    Attributes:
        X: numpy array - input datas.
        Y: numpy array - labels.
        ind: string - patient id
    """

    def __init__(self, X, Y, ind):
        super().__init__()
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.ind = ind

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.ind[i]


def SpiltData(X, Y, patient_name, BATCH_SIZE=32, split_rate=0.2):
    """ Spilt Data randomly  
    Attributes:
        X,Y: data and label 
        BATCH_SIZE: BATCH_SIZE
        split_rate : % of test data 
    Return:
        train_dl: Loaded train set 
        test_dl: Loaded test set 
        len(train_set), len(test_set): length of train set and test set
    """
    dataset = LoadDataset(X, Y, patient_name)
    N_test_samples = round(split_rate * len(dataset))
    train_set, test_set = torch.utils.data.random_split(
        dataset, [len(dataset) - N_test_samples, N_test_samples]
    )
    train_dl = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=1)
    return train_dl, test_dl, len(train_set), len(test_set)


def SpiltData_unsupervised(X, Y, patient_name, BATCH_SIZE=32, split_rate=0.2):
    """ Spilt Data randomly  
    Attributes:
        X,Y: data and label 
        BATCH_SIZE: BATCH_SIZE
        split_rate : % of test data 
    Return:
        train_dl: Loaded train set 
        test_dl: Loaded test set 
        len(train_set), len(test_set): length of train set and test set
    """
    dataset = LoadDataset(X, Y, patient_name)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_dl = torch.utils.data.DataLoader(dataset, batch_size=1)
    return train_dl, len(train_dl.dataset)


def CrossVal(X, Y, patient_name, BATCH_SIZE=32, nfold=0, seed=1):
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    i = 0
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        ind_train, ind_test = patient_name[train_index], patient_name[test_index]
        dtrain = LoadDataset(X_train, y_train, ind_train)
        # train_set, _ = torch.utils.data.random_split(dtrain, [1])
        train_dl = torch.utils.data.DataLoader(
            dtrain, batch_size=BATCH_SIZE, shuffle=True
        )
        dtest = LoadDataset(X_test, y_test, ind_test)
        # _, test_set = torch.utils.data.random_split(dtest, [0])
        test_dl = torch.utils.data.DataLoader(dtest, batch_size=1)
        if i == nfold:
            return train_dl, test_dl, len(dtrain), len(dtest), y_test
        i += 1


class LeNet_300_100_DNN(nn.Module):
    def __init__(self, n_inputs, n_outputs=2):

        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, n_outputs),
        )

    def forward(self, x):
        return self.encoder(x)


class DNN(nn.Module):
    def __init__(self, n_inputs, n_outputs=2, n_hidden=300):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x):
        return self.encoder(x)


def RunDNN(
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
    ETA_STAR=100,
    AXIS=0,
):
    """ Main loop for Neural Network, run autoencoder and return the encode and decode matrix
    
    Args:
        net, criterion_classification, optimizer: class - net configuration 
        train_dl,  train_len: pytorch Dataset type - used full data as train set
        N_EPOCHS : int - number of epoch 
        outputPath: string - patch to store the best net
        DO_PROJ_middle: bool - Do projection at middle layer or not(default is No) 
        run_model: string-
                    'ProjectionLastEpoch' and 'MaskGrad' for double descend
                    'None': original training
            
            
    Return: 
        data_encoder: tensor - encoded data
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_loss, epoch_acc, train_time = (
        [],
        [],
        [],
    )
    (epoch_val_loss, epoch_val_acc,) = ([], [])
    best_test = 0
    for e in range(N_EPOCHS):
        t1 = time.perf_counter()
        print("EPOCH:", e)
        running_loss, running_accuracy = 0, 0
        net.train()
        for i, batch in enumerate(tqdm(train_dl)):
            x = batch[0]
            labels = batch[1]

            if torch.cuda.is_available():
                x = x.cuda()
                labels = labels.cuda()

            encoder_out = net(x)

            # Compute the loss
            loss = criterion_classification(encoder_out, labels.long())

            optimizer.zero_grad()
            loss.backward()

            # Set the gradient as 0
            if run_model == "MaskGrad":
                for index, param in enumerate(list(net.parameters())):
                    if index < len(list(net.parameters())) / 2 - 2 and index % 2 == 0:
                        param.grad[DO_PROJ_middle[int(index / 2)]] = 0
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
                running_accuracy += (encoder_out.max(1)[1] == labels).sum().item()

            if e == N_EPOCHS - 1:
                # labels = encoder_out.max(1)[1].float()
                if i == 0:
                    data_out = torch.cat((encoder_out, labels.view(-1, 1)), dim=1)
                else:
                    tmp2 = torch.cat((encoder_out, labels.view(-1, 1)), dim=1)
                    data_out = torch.cat((data_out, tmp2), dim=0)

        t2 = time.perf_counter()
        train_time.append(t2 - t1)
        print("Total loss:", running_loss / float(train_len))
        epoch_loss.append(running_loss / train_len)
        epoch_acc.append(running_accuracy / train_len)

        # Do projection at last epoch (GRADIENT_MASK)
        if run_model == "ProjectionLastEpoch" and e == (N_EPOCHS - 1):
            net_parameters = list(net.parameters())
            for index, param in enumerate(net_parameters):
                if (
                    DO_PROJ_middle == False and index != len(net_parameters) / 2 - 2
                ):  # Do no projection at middle layer
                    param.data = Projection(
                        param.data, TYPE_PROJ, ETA, ETA_STAR, AXIS, device
                    ).to(device)

        # testing our model
        running_loss, running_accuracy = 0, 0
        net.eval()

        for i, batch in enumerate(tqdm(test_dl)):
            with torch.no_grad():
                x = batch[0]
                labels = batch[1]
                if torch.cuda.is_available():
                    x = x.cuda()
                    labels = labels.cuda()
                encoder_out = net(x)

                # Compute the loss
                loss = criterion_classification(encoder_out, labels.long())
                running_loss += loss.item()
                running_accuracy += (encoder_out.max(1)[1] == labels).sum().item()
        print(
            "test accuracy : ",
            running_accuracy / test_len,
            "Total loss:",
            running_loss / float(test_len),
        )
        if running_accuracy > best_test:
            best_net_it = e
            best_test = running_accuracy
            torch.save(net.state_dict(), str(outputPath) + "best_net")
        epoch_val_loss.append(running_loss / test_len)
        epoch_val_acc.append(running_accuracy / test_len)

    print("Epoch du best net = ", best_net_it)
    # Plot
    if str(run_model) != "ProjectionLastEpoch":
        # plt.figure()
        # plt.plot( epoch_acc )
        # plt.plot( epoch_val_acc )
        # plt.title('Total accuracy classification')
        # plt.show()
        print(
            f"{N_EPOCHS} epochs trained for {sum(train_time)}s, {np.mean(train_time)} s/epoch"
        )
    return data_out, epoch_loss, best_test, net


def selectf(x, feature_name):
    x = x.cpu()
    n, d = x.shape
    mat = []
    for i in range(d):
        mat.append([feature_name[i] + "", np.linalg.norm(x[:, i])])
    mat = sorted(mat, key=lambda norm: norm[1], reverse=True)
    columns = ["Genes", "Weights"]
    res = pd.DataFrame(mat)

    res = res.sort_values(1, axis=0, ascending=False)
    res.columns = columns
    # res.to_csv('{}topGenesCol.csv'.format(outputPath) , sep =';')
    return mat


def runBestNet(
    train_dl,
    test_dl,
    best_test,
    outputPath,
    nfold,
    class_len,
    net,
    feature_name,
    test_len,
):
    """ Load the best net and test it on your test set 
    Attributes:
        train_dl, test_dl: train(test) sets
        best_test: the testing accuracy of best model 
        outputPath: patch to load the net weights 
    Return:

        class_test: accuracy of each class for testing       
    """
    class_test_correct = np.zeros(class_len)
    class_test_total = np.zeros(class_len)
    class_train_correct = np.zeros(class_len)
    class_train_total = np.zeros(class_len)
    correct_pred = []
    Y_predit = []
    Y_true = []
    L = (
        []
    )  # [np.array(["Name", "Labels"] + ["Proba class "+ str(i) for i in range(class_len)])]
    best_value = np.zeros((1, 1))
    net.load_state_dict(torch.load(str(outputPath) + "best_net"))
    net.eval()
    for i, batch in enumerate(tqdm(train_dl)):
        x = batch[0]
        labels = batch[1]
        if torch.cuda.is_available():
            x = x.cuda()
            labels = labels.cuda()
        network_out = net(x)
        with torch.no_grad():
            c = (network_out.max(1)[1] == labels).squeeze()
            for i in range(len(x)):

                label = int(labels[i].item())
                if c.dim() == 0:

                    class_train_correct[label] += int(c.item())
                else:
                    class_train_correct[label] += int(c[i].item())
                class_train_total[label] += 1
    First = True
    for i, batch in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            x = batch[0]
            labels = batch[1]
            index = batch[2]
            if torch.cuda.is_available():
                x = x.cuda()
                labels = labels.cuda()
            network_out = net(x)
            m = nn.Softmax(dim=1)
            L.append(
                [index[0], labels.item()]
                + m(network_out).detach().cpu().numpy().tolist()[0]
            )
            Y_predit.append(network_out.max(1)[1].item())
            Y_true.append(labels.item())
            c = (network_out.max(1)[1] == labels).squeeze()
            for i in range(len(x)):
                label = int(labels[i].item())
                if c.dim() == 0:
                    class_test_correct[label] += int(c.item())
                    if c.item():
                        correct_pred.append(index[0])
                else:
                    class_test_correct[label] += int(c[i].item())
                    if c[i].item():
                        correct_pred.append(index[i][0])
                class_test_total[label] += 1

            if First:
                data_encoder = torch.cat((network_out, labels.view(-1, 1)), dim=1)

                First = False
            else:

                tmp2 = torch.cat((network_out, labels.view(-1, 1)), dim=1)
                data_encoder = torch.cat((data_encoder, tmp2), dim=0)

    if best_test != sum(class_test_correct):
        print("!!!!!!! Problem !!!!!!!")
    class_train = (class_train_correct / class_train_total).reshape(1, -1)
    best_value[0] = sum(class_train_correct) / sum(class_train_total)
    class_train = np.hstack((best_value, class_train))
    class_test = (class_test_correct / class_test_total).reshape(1, -1)
    best_value[0] = sum(class_test_correct) / sum(class_test_total)
    class_test = np.hstack((best_value, class_test))

    try:
        if nfold != 0:
            df = pd.read_csv(
                "{}Labelspred_softmax.csv".format(outputPath), sep=";", header=0
            )

            soft = pd.DataFrame(L)
            soft = pd.DataFrame(
                np.concatenate((df.values, soft.values[:, :])),
                columns=["Name", "Labels"]
                + ["Proba class " + str(i) for i in range(class_len)],
            )
            soft.to_csv("{}Labelspred_softmax.csv".format(outputPath), sep=";", index=0)
        else:
            soft = pd.DataFrame(
                L,
                columns=["Name", "Labels"]
                + ["Proba class " + str(i) for i in range(class_len)],
            )
            soft.to_csv("{}Labelspred_softmax.csv".format(outputPath), sep=";", index=0)
    except FileNotFoundError:
        soft = pd.DataFrame(
            L,
            columns=["Name", "Labels"]
            + ["Proba class " + str(i) for i in range(class_len)],
        )
        soft.to_csv("{}Labelspred_softmax.csv".format(outputPath), sep=";", index=0)

    normGenes = selectf(net.state_dict()["encoder.0.weight"], feature_name)

    return (
        data_encoder,
        class_train,
        class_test,
        normGenes,
        correct_pred,
        soft,
        Y_true,
        Y_predit,
    )


def packClassResult(accuracy_train, accuracy_test, fold_nb, label_name):
    """ Transform the accuracy of each class in different fold to DataFrame
    Attributes:
        accuracy_train: List, class_train in different fold
        accuracy_test: List, class_test in different fold 
        fold_nb: number of fold  
        label_name: name of different classes(Ex: Class 1， Class 2)
    Return:
        df_accTrain: dataframe, training accuracy per Class in different fold 
        df_acctest: dataframe, testing accuracy per Class in different fold     
    """
    columns = ["Global"] + ["Class " + str(x) for x in label_name]
    ind_df = ["Fold " + str(x + 1) for x in range(fold_nb)]

    df_accTrain = pd.DataFrame(accuracy_train, index=ind_df, columns=columns)
    df_accTrain.loc["Mean"] = df_accTrain.apply(lambda x: x.mean())
    df_accTrain.loc["Std"] = df_accTrain.apply(lambda x: x.std())

    df_acctest = pd.DataFrame(accuracy_test, index=ind_df, columns=columns)
    df_acctest.loc["Mean"] = df_acctest.apply(lambda x: x.mean())
    df_acctest.loc["Std"] = df_acctest.apply(lambda x: x.std())
    return df_accTrain, df_acctest


def packMetricsResult(data_train, data_test, fold_nb):
    """ Transform the accuracy of each class in different fold to DataFrame
    Attributes:
        accuracy_train: List, class_train in different fold
        accuracy_test: List, class_test in different fold 
        fold_nb: number of fold  
        label_name: name of different classes(Ex: Class 1， Class 2)
    Return:
        df_accTrain: dataframe, training accuracy per Class in different fold 
        df_acctest: dataframe, testing accuracy per Class in different fold     
    """
    columns = (
        ["Silhouette"]
        + ["ARI"]
        + ["AMI"]
        + ["AUC"]
        + ["Precision"]
        + ["Recall"]
        + ["F1 score"]
    )
    ind_df = ["Fold " + str(x + 1) for x in range(fold_nb)]

    df_metricsTrain = pd.DataFrame(data_train, index=ind_df, columns=columns)
    df_metricsTrain.loc["Mean"] = df_metricsTrain.apply(lambda x: x.mean())
    df_metricsTrain.loc["Std"] = df_metricsTrain.apply(lambda x: x.std())

    df_metricsTest = pd.DataFrame(data_test, index=ind_df, columns=columns)
    df_metricsTest.loc["Mean"] = df_metricsTest.apply(lambda x: x.mean())
    df_metricsTest.loc["Std"] = df_metricsTest.apply(lambda x: x.std())
    return df_metricsTrain, df_metricsTest


def showClassResult_unsupervised(accuracy_train, fold_nb, label_name):
    """ Transform the accuracy of each class in different fold to DataFrame
    Attributes:
        accuracy_train: List, class_train in different fold
        accuracy_test: List, class_test in different fold 
        fold_nb: number of fold  
        label_name: name of different classes(Ex: Class 1， Class 2)
    Return:
        df_accTrain: dataframe, training accuracy per Class in different fold 
        df_acctest: dataframe, testing accuracy per Class in different fold     
    """
    columns = ["Global"] + ["Class " + str(x) for x in label_name]
    ind_df = ["Fold " + str(x + 1) for x in range(fold_nb)]
    df_accTrain = pd.DataFrame(accuracy_train, index=ind_df, columns=columns)
    df_accTrain.loc["Mean"] = df_accTrain.apply(lambda x: x.mean())
    print("\nAccuracy Train")
    print(df_accTrain)

    return df_accTrain


def showMetricsResult_unsupervised(data_train, fold_nb):
    """ Transform the accuracy of each class in different fold to DataFrame
    Attributes:
        accuracy_train: List, class_train in different fold
        accuracy_test: List, class_test in different fold 
        fold_nb: number of fold  
        label_name: name of different classes(Ex: Class 1， Class 2)
    Return:
        df_accTrain: dataframe, training accuracy per Class in different fold 
        df_acctest: dataframe, testing accuracy per Class in different fold     
    """
    columns = ["Silhouette"] + ["ARI"] + ["AMI"]
    ind_df = ["Fold " + str(x + 1) for x in range(fold_nb)]
    df_accTrain = pd.DataFrame(data_train, index=ind_df, columns=columns)
    df_accTrain.loc["Mean"] = df_accTrain.apply(lambda x: x.mean())

    print("\nMetrics Train")
    print(df_accTrain)

    return df_accTrain


def Projection(W, TYPE_PROJ=proj_l11ball, ETA=100, AXIS=0, ETA_STAR=100, device="cpu"):
    """ For different projection, give the correct args and do projection
    Args:
        W: tensor - net weight matrix
        TYPE_PROJ: string and funciont- use which projection  
        ETA: int - only for Proximal_PGL1 or Proximal_PGL11 projection 
        ETA_STAR: int - only for Proximal_PGNuclear or Proximal_PGL1_Nuclear projection 
        AXIS: int 0,1 - only for Proximal_PGNuclear or Proximal_PGL1_Nuclear projection 
        device: parameters of projection 
    Return:
        W_new: tensor - W after projection 
    """

    # global TYPE_PROJ, ETA, ETA_STAR, AXIS, device
    if TYPE_PROJ == "No_proj":
        W_new = W
    if (
        TYPE_PROJ == proj_l1ball
        or TYPE_PROJ == proj_l11ball
        or TYPE_PROJ == proj_l11ball_line
    ):
        W_new = TYPE_PROJ(W, ETA, device)
    if TYPE_PROJ == proj_l21ball or TYPE_PROJ == proj_l12ball:
        W_new = TYPE_PROJ(W, ETA, AXIS, device=device)
    if TYPE_PROJ == proj_nuclear:
        W_new = TYPE_PROJ(W, ETA_STAR, device=device)
    return W_new


class CMDS_Loss(nn.Module):
    """Equation(1) in Self-calibrating Neural Networks for Dimensionality Reduction

    Attributes:
        X: tensor - original datas.
        Y: tensor - encoded datas.
    Returns:
        cmds: float - The cmds loss.
    """

    def __init__(self):
        super(CMDS_Loss, self).__init__()

    def forward(self, y, x):
        XTX = Covariance(x.T, bias=True)
        YTY = Covariance(y.T, bias=True)
        cmds = torch.norm(XTX - YTY) ** 2
        return cmds


def ShowPcaTsne(X, Y, data_encoder, center_distance, class_len, tit):
    """ Visualization with PCA and Tsne
    Args:
        X: numpy - original imput matrix
        Y: numpy - label matrix  
        data_encoder: tensor  - latent sapce output, encoded data  
        center_distance: numpy - center_distance matrix
        class_len: int - number of class 
    Return:
        Non, just show results in 2d space  
    """

    # Define the color list for plot
    color = [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#BCBD22",
        "#17BECF",
        "#40004B",
        "#762A83",
        "#9970AB",
        "#C2A5CF",
        "#E7D4E8",
        "#F7F7F7",
        "#D9F0D3",
        "#A6DBA0",
        "#5AAE61",
        "#1B7837",
        "#00441B",
        "#8DD3C7",
        "#FFFFB3",
        "#BEBADA",
        "#FB8072",
        "#80B1D3",
        "#FDB462",
        "#B3DE69",
        "#FCCDE5",
        "#D9D9D9",
        "#BC80BD",
        "#CCEBC5",
        "#FFED6F",
    ]
    color_original = [color[i] for i in Y]

    # Do pca for original data
    pca = PCA(n_components=2)
    X_pca = X if class_len == 2 else pca.fit(X).transform(X)
    X_tsne = X if class_len == 2 else TSNE(n_components=2).fit_transform(X)

    # Do pca for encoder data if cluster>2
    if data_encoder.shape[1] != 3:  # layer code_size >2  (3= 2+1 data+labels)
        data_encoder_pca = data_encoder[:, :-1]
        X_encoder_pca = pca.fit(data_encoder_pca).transform(data_encoder_pca)
        X_encoder_tsne = TSNE(n_components=2).fit_transform(data_encoder_pca)
        Y_encoder_pca = data_encoder[:, -1].astype(int)
    else:
        X_encoder_pca = data_encoder[:, :-1]
        X_encoder_tsne = X_encoder_pca
        Y_encoder_pca = data_encoder[:, -1].astype(int)
    color_encoder = [color[i] for i in Y_encoder_pca]

    # Do pca for center_distance
    labels = np.unique(Y)
    center_distance_pca = pca.fit(center_distance).transform(center_distance)
    color_center_distance = [color[i] for i in labels]

    # Plot
    title2 = tit

    plt.figure()
    plt.title(title2)
    plt.scatter(X_encoder_pca[:, 0], X_encoder_pca[:, 1], c=color_encoder)

    plt.show()


def CalculateDistance(x):
    """ calculate columns pairwise distance
    Args:
         x: matrix - with shape [m, d]
    Returns:
         dist: matrix - with shape [d, d]
    """
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist


def Covariance(m, bias=False, rowvar=True, inplace=False):
    """ Estimate a covariance matrix given data(tensor).
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: numpy array - A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: bool - If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """

    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1) if not bias else 1.0 / (m.size(1))
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def Reconstruction(INTERPELLATION_LAMBDA, data_encoder, net, class_len):
    """ Reconstruction the images by using the centers in laten space and datas after interpellation
    Args:
         INTERPELLATION_LAMBDA: float - [0,1], interpolated_datas = (1-λ)*x + λ*y
         data_encoder: tensor - data in laten space (output of encoder)
         net: autoencoder net
         
    Returns:
         center_mean: numpy - with shape[class_len, class_len], center of each cluster
         interpellation_latent: numpy - with shape[class_len, class_len], interpolated datas
         
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # For interpellation
    interpellation_latent = np.zeros((class_len, class_len))
    # center of encoder data
    center_mean = np.zeros((class_len, class_len))
    center_latent = np.zeros((class_len, class_len))
    center_Y = np.unique(data_encoder[:, -1])
    for i in range(class_len):
        # For interpellation
        data_i = (data_encoder[data_encoder[:, -1] == center_Y[i]])[:, :-1]
        index_x, index_y = np.random.randint(0, data_i.shape[0], 2)
        interpellation_latent[i] = (
            INTERPELLATION_LAMBDA * data_i[index_x, :]
            + (1 - INTERPELLATION_LAMBDA) * data_i[index_y, :]
        )
        # center of encoder data
        center_mean[i] = data_i.mean(axis=0)

    #    # Decode interpellation data
    #    interpellation_decoded = net.decoder(torch.from_numpy(interpellation_latent).float().to(device))

    # Distance of each center
    center_distance = CalculateDistance(center_mean)

    return center_mean, center_distance


def topGenes(
    X, Y, feature_name, class_len, feature_len, method, nb_samples, device, net
):
    """ Get the rank of features for each class, depends on it's contribution 
    Attributes:
        X,Y,feature_name,class_len, feature_len,  device : datas
        method: 'Shap' is very slow; 'Captum_ig', 'Captum_dl', Captum_gs' give almost the same results
        nb_samples: only for 'Shap', we used a part of the original data, other methods used all original data 
    Return:
        res: dataframe, ranked features (a kind of interpretation of neural networks) 
    """

    input_x = torch.from_numpy(X).float().to(device)
    if method == "Shap":
        print("Running Shap Model... (It may take a long time)")
        nb_samples = nb_samples
        rand_index = np.random.choice(input_x.shape[0], nb_samples, replace=True)
        background = input_x[rand_index]
        Y_rand = Y[rand_index].reshape(-1, 1)
        Y_unique, Y_counts = np.unique(Y_rand, return_counts=True)
        # Create object that can calculate shap values and explain predictions of the model
        explainer = shap.DeepExplainer(net.encoder, background)
        # Calculate Shap values, with dimension (y*N*x) y:number of labels, N number of background samples, x number of features
        shap_values = explainer.shap_values(background)
    if method == "Captum_ig":
        baseline = torch.zeros((X.shape)).to(device)
        ig = IntegratedGradients(net.encoder)
        attributions, delta = ig.attribute(
            input_x, baseline, target=0, return_convergence_delta=True
        )
    if method == "Captum_dl":
        baseline = torch.zeros((X.shape)).to(device)
        dl = DeepLift(net.encoder)
        attributions, delta = dl.attribute(
            input_x, baseline, target=0, return_convergence_delta=True
        )
    if method == "Captum_gs":
        baseline_dist = (torch.randn((X.shape)) * 0.001).to(device)
        gs = GradientShap(net.encoder)
        attributions, delta = gs.attribute(
            input_x,
            stdevs=0.09,
            n_samples=10,
            baselines=baseline_dist,
            target=0,
            return_convergence_delta=True,
        )

    # Use the weight differences to do rank
    if class_len == 2:
        class_len = 1
    feature_rank = np.empty(
        (feature_len, 2 * class_len), dtype=object
    )  # save ranked features and weights
    # one class vs others
    for class_index in range(class_len):
        attributions_mean_list = []
        Y_i = Y.copy()
        Y_i[Y_i != class_index] = class_index + 1  # change to 2 class
        Y_unique, Y_counts = np.unique(Y_i, return_counts=True)
        # repeat 2 times
        for i in Y_unique:
            if method == "Shap":
                attributions_i = torch.from_numpy(shap_values[i]).float().to(device)
            else:
                attributions_i = attributions[Y_i == i]  # find all X of each class
            attributions_mean = torch.mean(attributions_i, dim=0)
            attributions_mean_list.append(attributions_mean)
        # class_weight differences
        class_weight = attributions_mean_list[0] - attributions_mean_list[1]
        attributions_weight, index_sorted = torch.sort(class_weight, descending=True)
        attributions_name = np.array([feature_name[x] for x in index_sorted])
        attributions_weight = attributions_weight.detach().cpu()
        feature_rank[:, class_index * 2] = attributions_name
        feature_rank[:, class_index * 2 + 1] = attributions_weight

    # Save results as DAtaFrame
    mat_head = np.array(
        ["topGenes" if x % 2 == 0 else "Weights" for x in range(class_len * 2)]
    )
    mat_head = mat_head.reshape(1, -1)
    mat = np.r_[mat_head, feature_rank]
    mat[1:, 1] = mat[1:, 1] / float(mat[1, 1])
    columns = ["Class" + str(int(x / 2) + 1) for x in range(class_len * 2)]
    ind_df = ["Attributes"] + [str(x) for x in range(feature_len)]
    res = pd.DataFrame(mat, index=ind_df, columns=columns)
    return res


def show_img(x_list, file_name):
    """Visualization of Matrix, color map
    
    Attributes:
        x_list: list - list of matrix to be shown.
        titile: list - list of figure title.
      
    Returns:
        None
    """

    # En valeur absolue
    x = x_list[0]
    d = np.zeros((x.shape[0] + 1, x.shape[1]))
    d[: x.shape[0], : x.shape[1]] = x
    d = np.where(d > 0, d, abs(d))
    d[-1, :] = np.linalg.norm(x, axis=0)

    x = np.array(sorted(d.T, key=lambda d: d[-1], reverse=True))

    x = x[:, :-1].T

    plt.figure()
    plt.plot()
    plt.title(file_name[:-4] + " without ReLU sorted")
    im = plt.imshow(
        x,
        cmap=plt.cm.jet,
        norm=mpl.colors.Normalize(vmin=x.min(), vmax=x.max()),
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(im)
    plt.tight_layout()
    plt.xlabel("Features")
    plt.ylabel("Neurons")
    plt.show()


def sparsity_line(M, tol=1.0e-3, device="cpu"):
    """Get the line sparsity(%) of M
    
    Attributes:
        M: Tensor - the matrix.
        tol: Scalar,optional - the threshold to select zeros.
        device: device, cpu or gpu
      
    Returns:
        spacity: Scalar (%)- the spacity of the matrix.

    """
    if type(M) is not torch.Tensor:
        M = torch.as_tensor(M, device=device)
    M1 = torch.where(torch.abs(M) < tol, torch.zeros_like(M), M)
    M1_sum = torch.sum(M1, 1)
    nb_nonzero = len(M1_sum.nonzero())
    return (1.0 - nb_nonzero / M1.shape[0]) * 100


def sparsity_col(M, tol=1.0e-3, device="cpu"):
    """Get the line sparsity(%) of M
    
    Attributes:
        M: Tensor - the matrix.
        tol: Scalar,optional - the threshold to select zeros.
        device: device, cpu or gpu
      
    Returns:
        spacity: Scalar (%)- the spacity of the matrix.

    """
    if type(M) is not torch.Tensor:
        M = torch.as_tensor(M, device=device)
    M1 = torch.where(torch.abs(M) < tol, torch.zeros_like(M), M)
    M1_sum = torch.sum(M1, 0)
    nb_nonzero = len(M1_sum.nonzero())
    return (1.0 - nb_nonzero / M1.shape[1]) * 100


def Recal_minibatch(X_encoder_pca, Y_encoder_pca, center_mean, data_encoder):
    clust1, clust2, clust12, clust21 = [], [], [], []
    for i in range(len(Y_encoder_pca)):

        if Y_encoder_pca[i] == 0 and sum(data_encoder[i, :-1] - center_mean[0]) > 0:
            clust1.append(data_encoder[i, :-1])
        #        X_encoder_pca[i , 0] = data_encoder[i,0] - center_mean[0][0]
        #        X_encoder_pca[i , 1] = data_encoder[i,1] + center_mean[0][1]
        if Y_encoder_pca[i] == 0 and sum(data_encoder[i, :-1] - center_mean[0]) < 0:
            clust12.append(data_encoder[i, :-1])
        #        X_encoder_pca[i , 0] = data_encoder[i,0] + center_mean[0][0]
        #        X_encoder_pca[i , 1] = data_encoder[i,1] - center_mean[0][1]
        #
        if Y_encoder_pca[i] == 1 and sum(data_encoder[i, :-1] - center_mean[1]) > 0:
            clust2.append(data_encoder[i, :-1])
        #        X_encoder_pca[i , 0] = data_encoder[i,0] + center_mean[1][0]
        #        X_encoder_pca[i , 1] = data_encoder[i,1] - center_mean[1][1]
        if Y_encoder_pca[i] == 1 and sum(data_encoder[i, :-1] - center_mean[1]) < 0:
            clust21.append(data_encoder[i, :-1])
    #        X_encoder_pca[i , 0] = data_encoder[i,0] - center_mean[1][0]
    #        X_encoder_pca[i , 1] = data_encoder[i,1] + center_mean[1][1]
    clust1 = np.array(clust1)
    mean1 = clust1.mean(axis=0)
    clust2 = np.array(clust2)
    mean2 = clust2.mean(axis=0)
    clust12 = np.array(clust12)
    mean12 = clust12.mean(axis=0)
    clust21 = np.array(clust21)
    mean21 = clust21.mean(axis=0)

    for i in range(len(Y_encoder_pca)):

        if Y_encoder_pca[i] == 0 and sum(data_encoder[i, :-1] - center_mean[0]) < 0:
            X_encoder_pca[i, 0] = data_encoder[i, 0] + np.linalg.norm(
                mean12[0] - mean1[0]
            )
            X_encoder_pca[i, 1] = data_encoder[i, 1] + np.linalg.norm(
                mean12[1] - mean1[1]
            )
            Y_encoder_pca[i] = 3
        if Y_encoder_pca[i] == 1 and sum(data_encoder[i, :-1] - center_mean[1]) < 0:
            X_encoder_pca[i, 0] = data_encoder[i, 0] + np.linalg.norm(
                mean21[0] - mean2[0]
            )
            X_encoder_pca[i, 1] = data_encoder[i, 1] + np.linalg.norm(
                mean21[1] - mean2[1]
            )
            Y_encoder_pca[i] = 4
    return X_encoder_pca, Y_encoder_pca


def RankFeature(w, feature_names):

    df1 = pd.DataFrame(feature_names, columns=["pd"])
    df1["weights"] = w
    # =====Sort the difference based on the absolute value=========
    df1["sort_helper"] = df1["weights"].abs()
    df2 = df1.sort_values(by="sort_helper", ascending=False).drop("sort_helper", axis=1)
    df2["weights"] = df2["weights"] / df2.iloc[0, 1]
    # ==== end_sort=============
    return df2


from sklearn.preprocessing import scale as scale


def ReadData(
    file_name="MyeloidProgenitors.csv",
    model="",
    TIRO_FORMAT=True,
    doScale=True,
    doLog=True,
):
    """Read different data(csv, npy, mat) files  
    * csv has two format, one is data of facebook, another is TIRO format.
    
    Args:
        file_name: string - file name, default directory is "datas/FAIR/"
        
    Returns:
        X(m*n): numpy array - m samples and n features, normalized   
        Y(m*1): numpy array - label of all samples (represented by int number, class 0,class 1，2，...)
        feature_name(n*1): string -  name of features
        label_name(m*1): string -  name of each class
    """
    global RankFeature
    if file_name.split(".")[-1] == "csv":
        if model == "autoencoder":
            data_pd = pd.read_csv(
                str(file_name),
                delimiter=";",
                decimal=",",
                header=0,
                encoding="ISO-8859-1",
            )
            X = (data_pd.iloc[1:, 1:].values.astype(float)).T
            Y = data_pd.iloc[0, 1:].values.astype(float).astype(int)
            feature_name = data_pd["Name"].values.astype(str)[1:]
            label_name = np.unique(Y)
        elif not TIRO_FORMAT:
            data_pd = pd.read_csv(
                "datas/" + str(file_name), delimiter=",", header=None, dtype="unicode"
            )
            index_root = data_pd[data_pd.iloc[:, -1] == "root"].index.tolist()
            data = data_pd.drop(index_root).values
            X = data[1:, :-1].astype(float)
            Y = data[1:, -1]
            feature_name = data[0, :-1]
            patient_name = data_pd.columns[1:]
            label_name = np.unique(data[1:, -1])
            X1 = X[np.where(Y == label_name[0])[0], :]
            X2 = X[np.where(Y == label_name[1])[0], :]

            difference = np.mean(X1, axis=0) - np.mean(X2, axis=0)
            # Do standardization
            RankFeature = RankFeature(difference, feature_name)
            X = X - np.mean(X, axis=0)
            # X = scale(X,axis=0)

        elif TIRO_FORMAT:
            data_pd = pd.read_csv(
                "datas/" + str(file_name),
                delimiter=";",
                decimal=",",
                header=0,
                encoding="ISO-8859-1",
            )
            # data_pd = pd.read_csv('datas/FAIR/'+ str(file_name),delimiter=',', decimal=".", header=0, encoding = 'ISO-8859-1')
            X = (data_pd.iloc[1:, 1:].values.astype(float)).T
            Y = data_pd.iloc[0, 1:].values.astype(float).astype(np.int64)
            col = data_pd.columns.to_list()
            if col[0] != "Name":
                col[0] = "Name"
            data_pd.columns = col
            feature_name = data_pd["Name"].values.astype(str)[1:]
            label_name = np.unique(Y)
            patient_name = data_pd.columns[1:]
            # Do standardization
            if doLog:
                X = np.log(abs(X + 1))  # Transformation

            X1 = X[np.where(Y == label_name[0])[0], :]
            X2 = X[np.where(Y == label_name[1])[0], :]

            difference = np.mean(X1, axis=0) - np.mean(X2, axis=0)

            RankFeature = RankFeature(difference, feature_name)
            X = X - np.mean(X, axis=0)
            if doScale:
                X = scale(X, axis=0)  # Standardization along rows
        for index, label in enumerate(
            label_name
        ):  # convert string labels to numero (0,1,2....)
            Y = np.where(Y == label, index, Y)
        Y = Y.astype(np.int64)

    return X, Y, feature_name, label_name, patient_name, RankFeature


if __name__ == "__main__":
    print("This is just a file containing functions, so nothing happened.")

