import time
from numpy import *
import matplotlib.pyplot as plt


# read data from file
def load_data(file_name, split_char='\t'):
    load_data_set = []
    with open(file_name) as FR:
        for line in FR.readlines():
            line_arr = line.strip().split(split_char)
            load_data_set.append(list(map(float, line_arr)))
    return load_data_set


# calculate Euclidean distance 计算欧式距离
def euclidean_distance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random points
def init_centroids(init_data_set, init_k):
    n_points, dim = init_data_set.shape
    centroid = zeros((init_k, dim))
    for i in range(init_k):
        index = int(random.uniform(0, n_points))
        centroid[i, :] = init_data_set[index, :]
    return centroid


# k-means cluster
def k_means(cluster_data_set, cluster_k):
    n_points = cluster_data_set.shape[0]
    # first column stores which cluster this point belongs to,
    # second column stores the error between this point and its centroid
    cluster_assment = mat(zeros((n_points, 2)))
    cluster_changed = True

    # step 1: init centroids
    cluster_centroids = init_centroids(cluster_data_set, cluster_k)

    while cluster_changed:
        cluster_changed = False
        # for each point
        for i in range(n_points):
            min_dist = 100000.0
            min_index = 0
            # for each centroid
            # step 2: find the centroid who is closest
            for j in range(cluster_k):
                distance = euclidean_distance(cluster_centroids[j, :], cluster_data_set[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j

            # step 3: update its cluster
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
                cluster_assment[i, :] = min_index, min_dist ** 2

        # step 4: update centroids
        for j in range(cluster_k):
            points_in_cluster = cluster_data_set[nonzero(cluster_assment[:, 0].A == j)[0]]
            cluster_centroids[j, :] = mean(points_in_cluster, axis=0)

    print('Done.')
    return cluster_centroids, cluster_assment


# show cluster
def show_cluster(show_data_set, show_k, show_centroids, show_cluster_assment):
    n_points = show_data_set.shape[0]

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if show_k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

    # draw all points
    for i in range(n_points):
        mark_index = int(show_cluster_assment[i, 0])
        plt.plot(show_data_set[i, 0], show_data_set[i, 1], mark[mark_index])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(show_k):
        plt.plot(show_centroids[i, 0], show_centroids[i, 1], mark[i], markersize=12)

    plt.show()


if __name__ == '__main__':
    # step 1: load data
    print("step 1: loading data...")
    data_set = load_data('./dataSet/testSet.txt', split_char='\t')
    data_set = mat(data_set)  # 转换为矩阵

    # step 2: clustering...
    print("step 2: clustering...")
    k = 4  # testSet.txt
    # k = 7  # 788points.txt

    start = time.clock()
    centroids, assment = k_means(data_set, k)
    end = time.clock()

    print('finish all in %s' % str(end - start))

    # step 3: show the result
    print("step 3: show the result...")
    show_cluster(data_set, k, centroids, assment)
