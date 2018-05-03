# coding=UTF-8
# !/usr/bin/python

import argparse
import time
from datetime import timedelta

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

    output_name = "/var/www/project/st_dbscan_result_{}_{}_{}.csv".format(spatial_threshold, temporal_threshold, min_pts)
    transform_save(st_db, output_name)


if __name__ == "__main__":
    main()
