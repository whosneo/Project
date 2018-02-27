import time
from numpy import *
import matplotlib.pyplot as plt


# calculate Euclidean distance 计算欧式距离
def euclidean_distance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
def init_centroids(data_set, k):
    num_samples, dim = data_set.shape
    centroid = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, num_samples))
        centroid[i, :] = data_set[index, :]
    return centroid


# k-means cluster
def k_means(data_set, k):
    num_samples = data_set.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assment = mat(zeros((num_samples, 2)))
    cluster_changed = True

    # step 1: init centroids
    centroids = init_centroids(data_set, k)

    while cluster_changed:
        cluster_changed = False
        # for each sample
        for i in range(num_samples):
            min_dist = 100000.0
            min_index = 0
            # for each centroid
            # step 2: find the centroid who is closest
            for j in range(k):
                distance = euclidean_distance(centroids[j, :], data_set[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j

            # step 3: update its cluster
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
                cluster_assment[i, :] = min_index, min_dist ** 2

        # step 4: update centroids
        for j in range(k):
            points_in_cluster = data_set[nonzero(cluster_assment[:, 0].A == j)[0]]
            centroids[j, :] = mean(points_in_cluster, axis=0)

    print('Done.')
    return centroids, cluster_assment


# show cluster
def show_cluster(data_set, k, centroids, cluster_assment):
    num_samples = data_set.shape[0]

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

    # draw all samples
    for i in range(num_samples):
        mark_index = int(cluster_assment[i, 0])
        plt.plot(data_set[i, 0], data_set[i, 1], mark[mark_index])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


def main():
    # step 1: load data
    print("step 1: loading data...")
    data_set = []
    file_in = open('./dataSet/testSet.txt')
    # file_in = open('./dataSet/788points.txt')
    for line in file_in.readlines():  # 逐行读取
        line_arr = line.strip().split('\t')
        # line_arr = line.strip().split(',')
        data_set.append([float(line_arr[0]), float(line_arr[1])])  # 二维数据

    # step 2: clustering...
    print("step 2: clustering...")
    data_set = mat(data_set)  # 转换为矩阵
    k = 4  # testSet.txt
    # k = 7  # 788points.txt
    centroids, cluster_assment = k_means(data_set, k)

    # step 3: show the result
    print("step 3: show the result...")
    show_cluster(data_set, k, centroids, cluster_assment)


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))
