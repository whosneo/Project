import argparse
import time

import pandas as pd
from sklearn.cluster import DBSCAN

from PROJECT import *


def main():
    parser = argparse.ArgumentParser(description='DBSCAN in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-s', '--eps', help='Radius Threshold', required=False, type=float, default=100)
    parser.add_argument('-p', '--minPts', help='Minimum number of points', required=False, type=int, default=200)
    args = parser.parse_args()

    filename = args.filename
    eps = args.eps
    min_pts = args.minPts

    df = pd.read_csv(filename, converters={'date_time': parse_dates})
    date_time = df['date_time']
    df = df.drop('date_time', 1)

    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_pts, metric=geo_distance).fit(df)
    print("[DBSCAN] Finish all in {} seconds".format(time.time() - start))

    df['date_time'] = date_time
    df['cluster'] = db.labels_

    output_name = "/var/www/project/dbscan_result_{}_{}.csv".format(eps, min_pts)
    transform_save(df, output_name)


if __name__ == '__main__':
    main()
