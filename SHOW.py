import argparse

import matplotlib.pyplot as plt
import pandas as pd

from PROJECT import *


def show(data):
    plt.figure(figsize=(8, 6))

    col = [0, 0, 0, 1]
    plt.plot(data[:, 1], data[:, 0], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)

    # plt.xlim(116.28, 116.33)
    # plt.ylim(39.98, 40.02)
    plt.title('Original data')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='DBSCAN in Python')
    parser.add_argument('-f', '--filename', help='Name of the file', required=True)
    args = parser.parse_args()

    filename = args.filename

    df = pd.read_csv(filename, sep=",", converters={'date_time': parse_dates})
    df = df.drop('date_time', 1)

    show(df.values)


if __name__ == '__main__':
    main()
