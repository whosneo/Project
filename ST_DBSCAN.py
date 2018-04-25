# coding=UTF-8
# !/usr/bin/python

import argparse
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PROJECT import *


def ST_DBSCAN(df, spatial_threshold, temporal_threshold, min_pts):
    labels_ = 0
    noise = -1
    unmarked = -2
    stack = []

    df['cluster'] = unmarked

    for index, point in df.iterrows():
        if df.loc[index]['cluster'] == unmarked:
            neighbors = find_neighbors(index, df, spatial_threshold, temporal_threshold)

            if len(neighbors) < min_pts:
                df.at[index, 'cluster'] = noise
            else:
                labels_ += 1
                df.at[index, 'cluster'] = labels_

                for neighbor in neighbors:
                    df.at[neighbor, 'cluster'] = labels_
                    stack.append(neighbor)

                while len(stack) > 0:
                    current_point_index = stack.pop()
                    new_neighbors = find_neighbors(current_point_index, df, spatial_threshold, temporal_threshold)

                    if len(new_neighbors) >= min_pts:
                        for new_neighbor in new_neighbors:
                            neighbor_cluster = df.loc[new_neighbor]['cluster']
                            if neighbor_cluster == unmarked:
                                df.at[new_neighbor, 'cluster'] = labels_
                                stack.append(new_neighbor)
                            elif neighbor_cluster == noise:
                                df.at[new_neighbor, 'cluster'] = labels_
    return df


# find neighbors by two thresholds
def find_neighbors(index_center, df, spatial_threshold, temporal_threshold):
    neighbors = []

    center_point = df.loc[index_center]

    min_time = center_point['date_time'] - timedelta(seconds=temporal_threshold)
    max_time = center_point['date_time'] + timedelta(seconds=temporal_threshold)
    df = df[(df['date_time'] >= min_time) & (df['date_time'] <= max_time)]

    for index, point in df.iterrows():
        if index != index_center:
            distance = geo_distance((center_point['latitude'], center_point['longitude']),
                                    (point['latitude'], point['longitude']))
            if distance <= spatial_threshold:
                neighbors.append(index)

    return neighbors


# show cluster
def show(data, spatial, temporal, min_pts):
    plt.figure(figsize=(8, 6))
    labels = data['cluster']
    data = np.mat(data)
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # continue
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)

    # plt.xlim(116.28, 116.33)
    # plt.ylim(39.98, 40.02)
    plt.xlim(116.2924, 116.33125)
    plt.ylim(39.9475, 40.02)
    plt.title('[ST-DBSCAN] Estimated number of clusters: %d spatial: %.1f temporal: %.1f minPts: %d' % (
        n_clusters_, spatial, temporal, min_pts))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='ST-DBSCAN in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-p', '--minPts', help='Minimum number of points', required=False, type=int, default=100)
    parser.add_argument('-s', '--spatial', help='Spatial Threshold (in meters)', required=False, type=float,
                        default=100)
    parser.add_argument('-t', '--temporal', help='Temporal Threshold (in seconds)', required=False, type=float,
                        default=900)
    args = parser.parse_args()

    filename = args.filename
    min_pts = args.minPts
    spatial_threshold = args.spatial
    temporal_threshold = args.temporal

    df = pd.read_csv(filename, converters={'date_time': parse_dates})

    start = time.time()
    st_db = ST_DBSCAN(df, spatial_threshold, temporal_threshold, min_pts)
    print("[ST-DBSCAN] Finish all in {} seconds".format(time.time() - start))

    show(st_db.values, spatial_threshold, temporal_threshold, min_pts)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    output_name = "st_dbscan_result_{}_{}_{}_{}.csv".format(spatial_threshold, temporal_threshold, min_pts, time_str)
    st_db.to_csv(output_name, index=False)


if __name__ == "__main__":
    main()
    # result = pd.read_csv("./st_dbscan_result_100_900_100_20180425-161956.csv", converters={'date_time': parse_dates})
    # show(result, 100, 900, 100)
