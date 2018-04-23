import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import Birch

from PROJECT import *


def show(data, birch_model, threshold):
    # Plot result
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    colors_ = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clusters)]

    plt.figure(figsize=(8, 6))

    for centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = (labels == k)
        xy = data[mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)
        if birch_model.n_clusters is None:
            plt.plot(centroid[1], centroid[0], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    plt.title('Birch with global clustering')

    # plt.xlim(116.28, 116.33)
    # plt.ylim(39.98, 40.02)
    plt.title('[BIRCH] Estimated number of clusters: %d Threshold: %.3f' % (n_clusters, threshold))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='DBSCAN in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-t', '--threshold', help='Threshold', required=False, type=float, default=0.004)
    args = parser.parse_args()

    filename = args.filename
    threshold = args.threshold

    df = pd.read_csv(filename, sep=",", converters={'date_time': parse_dates})
    df = df.drop('date_time', 1)

    start = time.time()
    birch_model = Birch(threshold=threshold, n_clusters=None).fit(df)  # Birch(threshold=1.7, n_clusters=100)
    print("[BIRCH] Finish all in {} seconds".format(time.time() - start))

    show(df.values, birch_model, threshold)


if __name__ == '__main__':
    main()
