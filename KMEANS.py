import argparse
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

from PROJECT import *


def main():
    parser = argparse.ArgumentParser(description='K-Means in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-k', '--k', help='The number of clusters', required=True, type=int)
    args = parser.parse_args()

    filename = args.filename
    k = args.k

    df = pd.read_csv(filename, converters={'date_time': parse_dates})
    date_time = df['date_time']
    df = df.drop('date_time', 1)

    start = time.time()
    k_means = KMeans(init='random', n_clusters=k).fit(df)  # init='k-means++'
    print("[KMEANS] Finish all in {} seconds".format(time.time() - start))

    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(df.values, k_means_cluster_centers)

    df['date_time'] = date_time
    df['cluster'] = k_means_labels

    output_name = "/var/www/project/k_means_result_{}.txt".format(k)
    transform_save(df, output_name)


if __name__ == '__main__':
    main()
