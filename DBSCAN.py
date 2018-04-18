import argparse
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def show(data, db):
    labels = db.labels_  # labels_ 表示各个点所属的簇的编号，-1表示该点为噪音点
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # continue
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def parse_dates(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='DBSCAN in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-s', '--eps', help='Radius Threshold', required=False, type=float, default=0.002855)
    parser.add_argument('-p', '--minPts', help='Minimum number of points', required=False, type=int, default=10)
    args = parser.parse_args()

    filename = args.filename
    eps = args.eps
    min_pts = args.minPts

    df = pd.read_csv(filename, sep=",", converters={'date_time': parse_dates})
    df = df.drop('date_time', 1)

    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(df)
    print("Finish all in {} seconds".format(time.time() - start))

    show(df.values, db)


if __name__ == '__main__':
    main()
