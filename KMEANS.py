import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

from PROJECT import *


def show(data, k_means, k):
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, k)]

    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(data, k_means_cluster_centers)

    plt.figure(figsize=(8, 6))
    for n, col in zip(range(k), colors):
        class_member_mask = (k_means_labels == n)
        center = k_means_cluster_centers[n]
        xy = data[class_member_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)
        plt.plot(center[1], center[0], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    # plt.xlim(116.28, 116.33)
    # plt.ylim(39.98, 40.02)
    plt.title('[KMEANS] Estimated number of clusters: %d' % k)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='DBSCAN in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-k', '--k', help='The number of clusters', required=True, type=int)
    args = parser.parse_args()

    filename = args.filename
    k = args.k

    df = pd.read_csv(filename, converters={'date_time': parse_dates})
    df = df.drop('date_time', 1)

    start = time.time()
    k_means = KMeans(init='random', n_clusters=k, n_init=10).fit(df)  # init='k-means++'
    print("[KMEANS] Finish all in {} seconds".format(time.time() - start))

    show(df.values, k_means, k)


if __name__ == '__main__':
    main()
