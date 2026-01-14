import numba
import numpy as np
from _code_ import kmeans_cost_label, algo1, k_means_cost, detAlg, samplingResult, hard_noisy_oracle, unpickle
from sklearn.cluster import KMeans
import random
# import csv
import math
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.cluster import kmeans_plusplus as kpp
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import BallTree
from numba import jit
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment

import time


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.array(linear_assignment(np.max(w) - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


@jit(nopython=True)
def find_minimum(dim_j_points, sample_id, n_neibor):
    minimum = 1E20
    best_point = -1
    dim_j_points = np.sort(dim_j_points)
    sum_l1 = dim_j_points ** 2
    for j1 in range(0, len(sample_id)):
        l = sample_id[j1]
        points_l = dim_j_points[l:l + n_neibor]
        sum_l = np.sum(points_l)
        cost = np.sum(sum_l1[l:l + n_neibor]) - sum_l ** 2 / points_l.shape[0]
        if (cost < minimum):
            minimum = cost
            best_point = sum_l / n_neibor
    return best_point


@jit(nopython=True)
def find_center(dim_j_points, sample_id, n_neibor):
    minimum = 1E20
    best_point = -1
    sum_l1 = dim_j_points ** 2
    for j1 in range(0, len(sample_id)):
        dis = np.abs(dim_j_points - dim_j_points[sample_id[j1]])
        nearest_id = np.argpartition(dis, n_neibor)[0:n_neibor]
        nearest_points = dim_j_points[nearest_id]
        sum_l = np.sum(nearest_points)
        cost = np.sum(sum_l1[nearest_id]) - sum_l ** 2 / n_neibor
        if (cost < minimum):
            minimum = cost
            best_point = sum_l / n_neibor
    return best_point


def Ours(points, oracle_labels, k, p_ours):
    n, d = points.shape

    # print("Method", k*d, n/100)

    centers = np.zeros((k, d))
    # sample_range = [i for i in range(0, n)]
    for i in range(0, k):
        R = 2
        points_i = points[np.where(oracle_labels == i)[0]]
        n_neibor = math.floor((1 - p_ours) * points_i.shape[0])
        if(points_i.shape[1] * k > points_i.shape[0]/25):
            sample_id1 = random.sample(range(0, points_i.shape[0] - n_neibor), min(R, points_i.shape[0] - n_neibor))
        else:
            sample_id1 = random.sample(range(0, points_i.shape[0]), min(R, points_i.shape[0]))
        sample_id1 = np.array(sample_id1)
        for j in range(0, d):
            dim_j_points = points_i[:, j]
            if (points_i.shape[1] * k > dim_j_points.shape[0]/25):
                best_point = find_minimum(dim_j_points, sample_id1, n_neibor)
            else:
                best_point = find_center(dim_j_points, sample_id1, n_neibor)

            centers[i][j] = best_point

    return centers

@jit(nopython=True)
def find_minimum1(dim_j_points, omega_j, sample_id, outliers, n_neibor_1):
    minimum = 1E20
    # best_point = dim_j_points[0].copy()
    for j1 in range(0, len(sample_id)):
        dis = np.sum((omega_j - dim_j_points[sample_id[j1]]) ** 2, axis=1)
        nearest = np.argpartition(dis, kth=- (omega_j.shape[0] - outliers))[0:omega_j.shape[0] - outliers]
        cost_j1 = (dis[nearest]).sum()
        if (cost_j1 < minimum):
            minimum = cost_j1
            best_point = dim_j_points[sample_id[j1]]
            best_point = best_point.reshape(1, best_point.shape[0])
            stop = 1
    dis_j1 = np.sum((dim_j_points - best_point) ** 2, axis=1)
    n_id = np.argpartition(dis_j1, n_neibor_1)[0:n_neibor_1]
    nearest_points = dim_j_points[n_id]
    center = np.sum(nearest_points, axis=0) / nearest_points.shape[0]
    return center


def Ours1(points, oracle_labels, k, p_ours):
    # print("Check", p_ours)
    n, d = points.shape
    centers = np.zeros((k, d))
    epsilon = 0.2
    for i in range(0, k):
        points_i = points[np.where(oracle_labels == i)[0]]
        n_neibor_1 = math.floor((1 - p_ours) * points_i.shape[0])
        R = 10
        epsilon = 1
        sample_size = math.log10(
            (points_i.shape[0] ** 3) * d * (math.log10(n * 1E4 / (epsilon ** 2))) ** 3) * math.log10(
            points_i.shape[0] * 1E4) / (epsilon ** 4)
        sample_size = min(int(sample_size), int(points_i.shape[0] / 20))
        sample_size = max(sample_size, 2)

        outliers = math.floor((p_ours * 1.3 * math.ceil(sample_size)))
        outliers = max(outliers, 1)

        dim_j_points = points_i
        omega_j = random.sample(range(0, points_i.shape[0]), sample_size)
        omega_j = dim_j_points[omega_j]
        sample_id = random.sample(range(0, dim_j_points.shape[0]), min(dim_j_points.shape[0], R))
        sample_id = np.array(sample_id)

        best_center = find_minimum1(dim_j_points.copy(), omega_j, sample_id, outliers, n_neibor_1)
        centers[i] = best_center

    return centers

@jit(nopython=True)
def generate_center_candidates(dim_j_points, sample_id, p_ours):
    lower = 1e-2
    upper = dim_j_points.shape[0] ** 2
    q = lower

    # Estimate a maximum size for the candidate array
    max_candidates = len(sample_id) * 10 * int(math.log2(upper / lower))
    center_candidates = np.zeros(max_candidates)
    count = 0  # Tracks the number of populated rows in center_candidates

    # Initial population of center_candidate with original sample points
    for j1 in range(len(sample_id)):
        if count < max_candidates:
            center_candidates[count] = dim_j_points[sample_id[j1]]
            count += 1

    # Iterative loop to populate center_candidate with shifted points
    while q < upper:
        lij = math.sqrt(q / ((1 - p_ours) * dim_j_points.shape[0]))
        shifts = np.array([-2 * lij, -lij, lij, 2 * lij])
        for j1 in range(len(sample_id)):
            base_point = dim_j_points[sample_id[j1]]
            for shift in shifts:
                if count < max_candidates:
                    center_candidates[count] = base_point + shift
                    count += 1

        q *= 10  # Double q in each iteration
    center_candidates = center_candidates[:count]
    center_candidates = np.sort(center_candidates)
    id_new = np.zeros(len(center_candidates), dtype=numba.int64)
    now = center_candidates[0]
    id_now = 1
    for i in range(1, len(center_candidates)):
        if (center_candidates[i] - now < 1E-2):
            continue
        else:
            id_new[id_now] = i
            id_now += 1
            now = center_candidates[i]
    id_new = id_new[:id_now]
    center_candidates = center_candidates[id_new]
    return center_candidates[:count]


@jit(nopython=True)
def find_minimum2(dim_j_points, omega_j, center_candidates, outliers, n_neibor_1):
    minimum = 1E20
    for j1 in range(0, len(center_candidates)):
        dis = np.abs(omega_j - center_candidates[j1])
        nearest = np.argpartition(dis, kth=- (omega_j.shape[0] - outliers))[0:omega_j.shape[0] - outliers]
        cost_j1 = (dis[nearest]).sum()
        if (cost_j1 < minimum):
            minimum = cost_j1
            best_point = center_candidates[j1]
    dis_j1 = np.abs(dim_j_points - best_point)
    n_id = np.argpartition(dis_j1, n_neibor_1)[0:n_neibor_1]
    nearest_points = dim_j_points[n_id]
    center = np.sum(nearest_points) / nearest_points.shape[0]
    return center


def Ours2(points, oracle_labels, k, p_ours):
    n, d = points.shape
    centers = np.zeros((k, d))
    epsilon = 0.2
    for i in range(0, k):
        points_i = points[np.where(oracle_labels == i)[0]]
        n_neibor_1 = math.floor((1 - 2 * p_ours) * points_i.shape[0])
        R = 5
        epsilon = 1
        sample_size = math.log10(
            (points_i.shape[0] ** 3) * d * (math.log10(n * 1E4 / (epsilon ** 2))) ** 3) * math.log10(
            points_i.shape[0] * 1E4) / (epsilon ** 4)
        sample_size = min(int(sample_size), int(points_i.shape[0] / 20))
        sample_size = max(sample_size, 2)
        weights = np.ones(sample_size) * points_i.shape[0] / sample_size
        outliers = math.floor((p_ours * 1.3 * math.ceil(sample_size)))
        outliers = max(outliers, 1)

        for j in range(0, d):
            dim_j_points = points_i[:, j]
            omega_j = random.sample(range(0, points_i.shape[0]), sample_size)
            omega_j = dim_j_points[omega_j]
            sample_id = random.sample(range(0, dim_j_points.shape[0]), min(dim_j_points.shape[0], R))

            center_candidates = generate_center_candidates(dim_j_points, sample_id, p_ours)
            centers[i][j] = find_minimum2(dim_j_points.copy(), omega_j, center_candidates, outliers, n_neibor_1)
            stop = 1

    return centers


if __name__ == '__main__':
    result = pd.DataFrame(columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])

    # data_name = ['phy']
    # data_name = ['cifar10','mnist']
    data_name = ['usps']

    # data_name = ['A1','A2','A3','S1','S2','S3','S4']
    # data_name = ['SUSY']
    err_range = [0.2]
    # err_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    nTrials = 15
    nIters = 10
    k_range = [10, 20, 30, 40, 50]
    # k_range = [20]
    nPortion = 1

    for i11 in range(0, len(data_name)):
        dataset = data_name[i11]

        print('loading data')
        if dataset == 'cifar10':
            data = unpickle("test1.dat")
            data = data[b'data'].astype(float)
            np.random.shuffle(data)
            nPortion = int(len(data) * nPortion)
            test = data[-nPortion:]
        elif dataset == 'phy':
            data = np.loadtxt("phy.dat")
            nPortion = int(len(data) * nPortion)
            np.random.shuffle(data)
            test = data[-nPortion:, :]
        elif dataset == 'mnist':
            data = load_digits().data
            nPortion = int(len(data) * nPortion)
            test = data[-nPortion:]
        elif dataset == 'fashion_mnist':
            print("Fetching Fashion-MNIST...")
            data = fetch_openml(name="Fashion-MNIST", as_frame=False).data
            nPortion = int(len(data) * nPortion)
            np.random.shuffle(data)
            test = data[-nPortion:]
        elif dataset == 'usps':
            print("Fetching USPS...")
            data = fetch_openml(name="USPS", version=2, as_frame=False).data
            nPortion = int(len(data) * nPortion)
            np.random.shuffle(data)
            test = data[-nPortion:]
        elif dataset == 'S1':
            k = 15
            data = np.loadtxt('s1.txt')
            gt = np.loadtxt('s1-cb.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'S2':
            data = np.loadtxt('s2.txt')
            gt = np.loadtxt('s2-cb.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            k = 15
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'S3':
            k = 15
            data = np.loadtxt('s3.txt')
            gt = np.loadtxt('s3-cb.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'S4':
            k = 15
            data = np.loadtxt('s4.txt')
            gt = np.loadtxt('s4-cb.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'A1':
            k = 20
            data = np.loadtxt('a1.txt')
            gt = np.loadtxt('a1-ga-cb.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'A2':
            k = 35
            data = np.loadtxt('a2.txt')
            gt = np.loadtxt('a2-ga-cb.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'A3':
            k = 50
            data = np.loadtxt('a3.txt')
            gt = np.loadtxt('a3-ga-cb.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'birch1':
            data = np.loadtxt('birch1.txt')
            gt = np.loadtxt('b1-gt.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'birch2':
            data = np.loadtxt('birch2.txt')
            gt = np.loadtxt('b2-gt.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'birch3':
            data = np.loadtxt('birch3.txt')
            gt = np.loadtxt('b3-gt.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'unbalance':
            data = np.loadtxt('unbalance.txt')
            gt = np.loadtxt('unbalance-gt.txt')
            Tree = BallTree(gt, leaf_size=40)
            dist, ind = Tree.query(data, k=1)
            test = data.copy()
            noisy_orc_labels = ind[:, 0]
            cost_opt = (dist[:, 0] ** 2).sum()
        elif dataset == 'kdd':
            data = np.loadtxt("kdd.txt", delimiter=",", usecols=(
                0, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34, 35,
                36, 37, 38, 39, 40))
            data = data[:, 0:data.shape[1] - 1]
            nPortion = int(len(data) * nPortion)
            np.random.shuffle(data)
            test = data[-nPortion:, :]
        elif dataset == "SUSY":
            data = np.loadtxt('data/SUSY.csv', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
                              delimiter=",")
            data = data[:, 0:data.shape[1] - 1]
            nPortion = int(len(data) * nPortion)
            np.random.shuffle(data)
            test = data[-nPortion:, :]
        elif dataset == 'HIGGS':
            data = np.loadtxt('HIGGS.csv', delimiter=",")
            data = data[:, 1:data.shape[1]]
            nPortion = int(len(data) * nPortion)
            np.random.shuffle(data)
            test = data[-nPortion:, :]
        elif dataset == 'SIFT':
            x = np.memmap("../../homeb/huangjy/learn.bvecs", dtype='uint8', mode='r')
            d11 = x[:4].view('int32')[0]
            data = x.reshape(-1, d11 + 4)[:, 4:]
            data = np.array(data)
            nPortion = int(len(data) * nPortion)
            np.random.shuffle(data)
            test = data[-nPortion:, :]

        for i12 in range(0, len(err_range)):

            err = err_range[i12]
            for i13 in range(0, len(k_range)):
                k = k_range[i13]

                print("Dataset size:", len(test), "k:", k, "error:", err)

                # test, _ = make_blobs(n_samples=1000000, n_features=10, centers=10)

                # pairwise_distances(X, kwargs)

                start = time.time()
                time_OPT = time.time() - start
                tkmeans = time.time()
                kmeans_scikit_10 = KMeans(n_clusters=k).fit(test)
                true_labels_10 = kmeans_scikit_10.labels_
                tkmeans1 = time.time()

                # #PlanB
                # CRT = KMeans(n_clusters=k,init="k-means++",n_init=1,max_iter=1)
                # CRT.fit(data)
                # centers = CRT.cluster_centers_

                # centers = Projection(data, centers)
                # LS_FAST = LS(n_clusters=k, rounds=500)
                # centers_f = LS_FAST.Fast_LS(data.copy(), centers)

                # "Calculating the Final Clustering Centers and Labels"
                # CRT1 = KMeans(n_clusters = k, init = centers_f, n_init=1, max_iter=300)
                # CRT1.fit(data)
                # centers_f = CRT1.cluster_centers_
                # true_labels_10 = CRT1.labels_

                print('Predictor: {} corruption (avg over {} trials)'.format(err, nIters))

                pvals_alg1 = np.linspace(.01, 0.5, nTrials)
                pvals_det = np.linspace(.01, .5, nTrials)
                pvals_ours = np.linspace(.01, .5, nTrials)
                pvals_ours1 = np.linspace(.01, .3, nTrials)
                pvals_ours2 = np.linspace(.01, .3, nTrials)

                # print(pvals_alg1)

                cost_oracle = []
                cost_sampling = []
                cost_algo = []
                det_cost_algo = []
                baseline_10 = []

                cost_ours = []
                cost_ours1 = []
                cost_ours2 = []
                ours_time = []
                ours_time1 = []
                ours_time2 = []
                baseline_time = []

                cost_opt = [kmeans_cost_label(test, true_labels_10, k)[1]]
                cost_opt = cost_opt[0]
                noisy_orc_labels = hard_noisy_oracle(test, true_labels_10, err)

                time_sampling = []
                time_alg = []
                det_time_alg = []
                time_OPT = [time_OPT]

                nmi_algo = []
                nmi_det = []
                nmi_ours = []
                nmi_ours1 = []
                nmi_ours2 = []

                ari_algo = []
                ari_det = []
                ari_ours = []
                ari_ours1 = []
                ari_ours2 = []

                cost_kmeans_plus = []
                time_kmeans_plus = []
                nmi_kmeans_plus = []
                ari_kmeans_plus = []

                for i in range(nIters):
                    kpp_start = time.time()
                    kpp_center = kpp(test, k)[0]
                    labels_kpp, kpp_cost = k_means_cost(test, kpp_center)
                    kpp_ari = adjusted_rand_score(true_labels_10, labels_kpp)
                    kpp_nmi = normalized_mutual_info_score(true_labels_10, labels_kpp)

                    cost_kmeans_plus.append(kpp_cost)
                    time_kmeans_plus.append(time.time() - kpp_start)
                    nmi_kmeans_plus.append(kpp_nmi)
                    ari_kmeans_plus.append(kpp_ari)

                    cost_oracle.append(kmeans_cost_label(test, noisy_orc_labels, k)[1])

                    start = time.time()
                    # SR = samplingResult(test, noisy_orc_labels, k)
                    SR = -1
                    time_sampling.append(time.time() - start)
                    cost_sampling.append(SR)
                    lowest = float('inf')
                    det_lowest = float('inf')
                    ours_lowest = float('inf')
                    ours_lowest1 = float('inf')
                    ours_lowest2 = float('inf')
                    alg1_time_count = 0
                    det_time_count = 0
                    ours_time_count = 0
                    ours_time_count1 = 0
                    ours_time_count2 = 0

                    algo_nmi = 0
                    det_nmi = 0
                    ours_nmi = 0
                    ours1_nmi = 0
                    ours2_nmi = 0

                    algo_ari = 0
                    det_ari = 0
                    ours_ari = 0
                    ours1_ari = 0
                    ours2_ari = 0

                    for p_alg1, p_det, p_ours, p_ours1, p_ours2 in zip(pvals_alg1, pvals_det, pvals_ours, pvals_ours1,
                                                                       pvals_ours2):
                        start = time.time()
                        # print(p_alg1)
                        cr = algo1(test, noisy_orc_labels.copy(), k, p_alg1)
                        alg1_time_count += time.time() - start

                        start = time.time()
                        dr = detAlg(test, noisy_orc_labels.copy(), k, p_det)
                        det_time_count += time.time() - start

                        start = time.time()
                        orr = Ours(test, noisy_orc_labels.copy(), k, p_ours)
                        ours_time_count += time.time() - start

                        start = time.time()
                        orr1 = Ours1(test, noisy_orc_labels.copy(), k, p_ours1)
                        ours_time_count1 += time.time() - start

                        start = time.time()
                        orr2 = Ours2(test, noisy_orc_labels.copy(), k, p_ours2)
                        ours_time_count2 += time.time() - start

                        labels_curr, curr_cost = k_means_cost(test, cr)
                        labels_det, det_curr_cost = k_means_cost(test, dr)
                        labels_ours, ours_cost = k_means_cost(test, orr)
                        labels_ours1, ours_cost1 = k_means_cost(test, orr1)
                        labels_ours2, ours_cost2 = k_means_cost(test, orr2)

                        if det_curr_cost < det_lowest:
                            det_lowest = det_curr_cost
                            det_nmi = normalized_mutual_info_score(true_labels_10, labels_det)
                            det_ari = adjusted_rand_score(true_labels_10, labels_det)

                        if curr_cost < lowest:
                            lowest = curr_cost
                            algo_nmi = normalized_mutual_info_score(true_labels_10, labels_curr)
                            algo_ari = adjusted_rand_score(true_labels_10, labels_curr)

                        if ours_cost < ours_lowest:
                            ours_lowest = ours_cost
                            ours_nmi = normalized_mutual_info_score(true_labels_10, labels_ours)
                            ours_ari = adjusted_rand_score(true_labels_10, labels_ours)

                        if ours_cost1 < ours_lowest1:
                            ours_lowest1 = ours_cost1
                            ours1_nmi = normalized_mutual_info_score(true_labels_10, labels_ours1)
                            ours1_ari = adjusted_rand_score(true_labels_10, labels_ours1)

                        if ours_cost2 < ours_lowest2:
                            ours_lowest2 = ours_cost2
                            ours2_nmi = normalized_mutual_info_score(true_labels_10, labels_ours2)
                            ours2_ari = adjusted_rand_score(true_labels_10, labels_ours2)

                    cost_algo.append(lowest)
                    det_cost_algo.append(det_lowest)
                    nmi_det.append(det_nmi)
                    ari_det.append(det_ari)
                    # acc_det.append(det_acc)

                    time_alg.append(alg1_time_count)
                    det_time_alg.append(det_time_count)
                    nmi_algo.append(algo_nmi)
                    ari_algo.append(algo_ari)
                    # acc_algo.append(algo_acc)

                    cost_ours.append(ours_lowest)
                    ours_time.append(ours_time_count)
                    nmi_ours.append(ours_nmi)
                    ari_ours.append(ours_ari)
                    # acc_ours.append(ours_acc)

                    cost_ours1.append(ours_lowest1)
                    ours_time1.append(ours_time_count1)
                    nmi_ours1.append(ours1_nmi)
                    ari_ours1.append(ours1_ari)

                    cost_ours2.append(ours_lowest2)
                    ours_time2.append(ours_time_count2)
                    nmi_ours2.append(ours2_nmi)
                    ari_ours2.append(ours2_ari)
                    # acc_ours1.append(ours1_acc)

                print('kmeans++:', np.average(cost_kmeans_plus), np.std(cost_kmeans_plus), "Time",
                      np.average(time_kmeans_plus), np.std(time_kmeans_plus), "NMI", np.average(nmi_kmeans_plus),
                      np.std(nmi_kmeans_plus), "ARI", np.average(ari_kmeans_plus), np.std(ari_kmeans_plus))
                print('Algo1:', np.average(cost_algo), np.std(cost_algo), "Time", np.average(time_alg),
                      np.std(time_alg), "NMI", np.average(nmi_algo), np.std(nmi_algo), "ARI", np.average(ari_algo),
                      np.std(ari_algo))
                print('Det:', np.average(det_cost_algo), np.std(det_cost_algo), "Time", np.average(det_time_alg),
                      np.std(det_time_alg), "NMI", np.average(nmi_det), np.std(nmi_det), "ARI", np.average(ari_det),
                      np.std(ari_det))
                print('Ours:', np.average(cost_ours), np.std(cost_ours), "Time", np.average(ours_time),
                      np.std(ours_time), "NMI", np.average(nmi_ours), np.std(nmi_ours), "ARI", np.average(ari_ours),
                      np.std(ari_ours))
                print('Ours1:', np.average(cost_ours1), np.std(cost_ours1), "Time", np.average(ours_time1),
                      np.std(ours_time1), "NMI", np.average(nmi_ours1), np.std(nmi_ours1), "ARI", np.average(ari_ours1),
                      np.std(ari_ours1))
                print('Ours2:', np.average(cost_ours2), np.std(cost_ours2), "Time", np.average(ours_time2),
                      np.std(ours_time2), "NMI", np.average(nmi_ours2), np.std(nmi_ours2), "ARI", np.average(ari_ours2),
                      np.std(ari_ours2))
                print('Optimal', cost_opt)

                new = pd.DataFrame(
                    [[dataset, k, err, "kmeans++", np.average(cost_kmeans_plus), 0, 0, 0]],
                    columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])
                result = pd.concat([result, new])
                
                new = pd.DataFrame(
                    [[dataset, k, err, "Algo1", np.average(cost_algo), np.std(cost_algo), np.average(time_alg),
                      np.std(time_alg)]],
                    columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])
                result = pd.concat([result, new])
                
                new = pd.DataFrame(
                    [[dataset, k, err, "Det", np.average(det_cost_algo), np.std(det_cost_algo),
                      np.average(det_time_alg), np.std(det_time_alg)]],
                    columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])
                result = pd.concat([result, new])
                
                new = pd.DataFrame(
                    [[dataset, k, err, "Ours", np.average(cost_ours), np.std(cost_ours), np.average(ours_time),
                      np.std(ours_time)]],
                    columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])
                result = pd.concat([result, new])
                
                new = pd.DataFrame(
                    [[dataset, k, err, "Ours1", np.average(cost_ours1), np.std(cost_ours1), np.average(ours_time1), np.std(ours_time1)]],
                    columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])
                result = pd.concat([result, new])

                new = pd.DataFrame(
                    [[dataset, k, err, "Ours2", np.average(cost_ours2), np.std(cost_ours2), np.average(ours_time2), np.std(ours_time2)]],
                    columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])
                result = pd.concat([result, new])
                
                new = pd.DataFrame(
                    [[dataset, k, err, "OPT", cost_opt, 0, 0, 0]],
                    columns=['dataset', 'k', 'alpha', 'method', 'cost', 'cost_dev', 'time', 'time_dev'])
                result = pd.concat([result, new])

    resultFileFormatted = dataset + 'KmeansResult.csv'
    resultFileFormattedTime = dataset + 'KmeansResultTime.csv'

    # load data

    result.to_csv(f"{dataset}_large_k.csv", index=False)
    '''
    cost_oracle = np.array(cost_oracle) 
    cost_sampling = np.array(cost_sampling) 
    cost_algo = np.array(cost_algo) 
    det_cost_algo = np.array(det_cost_algo)
    baseline_10 = np.array(baseline_10)
    result = np.array([baseline_10, cost_oracle, cost_sampling, cost_algo, det_cost_algo]).T
    mean = np.mean(result, axis = 0)
    std = np.std(result, axis = 0)
    result = np.expand_dims(np.append(mean, std), 0)
    result = np.append(result, [[np.average(cost_opt)]], axis = 1)

    header = ["Params", "k++" ,"Oracle", "Sampling" , "Ergun, Jon, et al.", "Ours", "k++EB" , "OracleEB", "SamplingEB" ,"Ergun, Jon, et al. EB", "OursEB", "OPT"]
    if args.overwrite:
        w = 'w'
    else:
        w = 'a'

    params = ["Error {} K {} Num Trials {}".format(args.err, args.k, args.nTrials)]
    params_result = list(map(str, result[0].tolist()))
    params.extend(params_result)

    with open(resultFileFormatted,w) as fd:
        writer = csv.writer(fd, delimiter=',')
        if args.overwrite:
            writer.writerows([header, params])
        else:
            writer.writerows([params])


    #######################################

    result = np.array([cost_sampling, cost_algo, det_cost_algo]).T
    mean = [np.mean(x) for x in [time_sampling, time_alg, det_time_alg]]
    std = [np.std(x) for x in [time_sampling, time_alg, det_time_alg]]
    result = np.expand_dims(np.append(mean, std), 0)
    result = np.append(result, [[np.average(time_OPT)]], axis = 1)

    header = ["Params",  "Sampling" , "Ergun, Jon, et al.", "Ours", "Sampling++EB" ,"Ergun, Jon, et al. EB", "OursEB", "OPT"]
    if args.overwrite:
        w = 'w'
    else:
        w = 'a'

    params = ["Error {} K {} Num Trials {} Dataset Portion {}".format(args.err, args.k, args.nTrials, args.nPortion)]
    params_result = list(map(str, result[0].tolist()))
    params.extend(params_result)

    with open(resultFileFormattedTime,w) as fd:
        writer = csv.writer(fd, delimiter=',')
        if args.overwrite:
            writer.writerows([header, params])
        else:
            writer.writerows([params])
    '''