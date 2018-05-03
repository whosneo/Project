import argparse
import time

import pandas as pd
from sklearn.cluster import Birch

from PROJECT import *


def main():
    parser = argparse.ArgumentParser(description='BIRCH in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    parser.add_argument('-t', '--threshold', help='Threshold', required=False, type=float, default=0.004)
    args = parser.parse_args()

    filename = args.filename
    threshold = args.threshold

    df = pd.read_csv(filename, converters={'date_time': parse_dates})
    date_time = df['date_time']
    df = df.drop('date_time', 1)

    start = time.time()
    birch_model = Birch(threshold=threshold, n_clusters=None).fit(df)  # Birch(threshold=1.7, n_clusters=100)
    print("[BIRCH] Finish all in {} seconds".format(time.time() - start))

    df['date_time'] = date_time
    df['cluster'] = birch_model.labels_

    output_name = "/var/www/project/birch_result_{}.txt".format(threshold)
    transform_save(df, output_name)


if __name__ == '__main__':
    main()
