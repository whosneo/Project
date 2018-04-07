# coding=UTF-8
# !/usr/bin/python

import argparse
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from geopy.distance import great_circle
from numpy import *

"""
    The minimal command to run this algorithm is:
    $ python main.py -f sample.csv
    Or could be executed with advanced configurations:
    $ python main.py -f sample.csv -p 5 -s 500 -t 60
    In the current moment, the data set must have the
    'latitude', 'longitude' and 'date_time' columns, but
    if you want, can be easily changed.
"""


def st_dbscan(df, spatial_threshold, temporal_threshold, min_neighbors):
    """
    Python st-dbscan implementation.
    INPUTS:
        df={o1,o2,...,on} Set of objects
        spatial_threshold = Maximum geographical coordinate (spatial) distance
        value
        temporal_threshold = Maximum non-spatial distance value
        min_neighbors = Minimum number of points within Eps1 and Eps2 distance
    OUTPUT:
        C = {c1,c2,...,ck} Set of clusters
    """
    cluster_label = 0
    noise = -1
    unmarked = 777777
    stack = []

    # initialize each point with unmarked
    df['cluster'] = unmarked

    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index]['cluster'] == unmarked:
            neighborhood = retrieve_neighbors(index, df, spatial_threshold,
                                              temporal_threshold)

            if len(neighborhood) < min_neighbors:
                df.at[index, 'cluster'] = noise
            else:  # found a core point
                cluster_label += 1
                # assign a label to core point
                df.at[index, 'cluster'] = cluster_label

                # assign core's label to its neighborhood
                for neig_index in neighborhood:
                    df.at[neig_index, 'cluster'] = cluster_label
                    stack.append(neig_index)  # append neighborhood to stack

                # find new neighbors from core point neighborhood
                while len(stack) > 0:
                    current_point_index = stack.pop()
                    new_neighborhood = retrieve_neighbors(
                        current_point_index, df, spatial_threshold,
                        temporal_threshold)

                    # current_point is a new core
                    if len(new_neighborhood) >= min_neighbors:
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index]['cluster']
                            if all([neig_cluster != noise,
                                    neig_cluster == unmarked]):
                                # TODO: verify cluster average
                                # before add new point
                                df.at[neig_index, 'cluster'] = cluster_label
                                stack.append(neig_index)
    return df


def retrieve_neighbors(index_center, df, spatial_threshold, temporal_threshold):
    neighborhood = []

    center_point = df.loc[index_center]

    # filter by time
    min_time = center_point['date_time'] - timedelta(seconds=temporal_threshold)
    max_time = center_point['date_time'] + timedelta(seconds=temporal_threshold)
    df = df[(df['date_time'] >= min_time) & (df['date_time'] <= max_time)]

    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            distance = great_circle(
                (center_point['latitude'], center_point['longitude']),
                (point['latitude'], point['longitude'])).meters
            if distance <= spatial_threshold:
                neighborhood.append(index)

    return neighborhood


def parse_dates(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


# show cluster
def show_cluster(data):
    data = mat(data)
    n = data.shape[0]
    fig = plt.figure()
    scatter_colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(n):
        if data[i, 4] < 0:
            continue
        color_style = scatter_colors[data[i, 4] % len(scatter_colors)]
        ax.scatter(data[i, 2], data[i, 1], c=color_style, s=50)
    plt.show()


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='ST-DBSCAN in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-p', '--minPts', help='Minimum number of points', required=False, type=int, default=60)
    parser.add_argument('-s', '--spatial', help='Spatial Threshold (in meters)', required=False, type=float,
                        default=200)
    parser.add_argument('-t', '--temporal', help='Temporal Threshold (in seconds)', required=False, type=float,
                        default=900)
    args = parser.parse_args()

    filename = args.filename
    min_pts = args.minPts
    spatial_threshold = args.spatial
    temporal_threshold = args.temporal

    df = pd.read_csv(filename, sep=",", converters={'date_time': parse_dates})

    result = st_dbscan(df, spatial_threshold, temporal_threshold, min_pts)
    print("Time Elapsed: {} seconds".format(time.time() - start))

    time_str = time.strftime("%Y%m%d-%H%M%S")

    show_cluster(result)
    output_name = "result_{}_{}_{}_{}.csv".format(spatial_threshold, temporal_threshold, min_pts, time_str)
    result.to_csv(output_name, index=False, sep=',')


def test_time(filename):
    df = pd.read_csv(filename, sep=",", converters={'date_time': parse_dates})
    result_t600 = st_dbscan(df, spatial_threshold=500, temporal_threshold=600, min_neighbors=5)

    df = pd.read_csv(filename, sep=",", converters={'date_time': parse_dates})
    result_t6 = st_dbscan(df, spatial_threshold=500, temporal_threshold=0.6, min_neighbors=5)

    assert not result_t600.equals(result_t6)


if __name__ == "__main__":
    main()
