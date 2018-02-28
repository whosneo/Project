from numpy import *
import matplotlib.pyplot as plt
import time

UNCLASSIFIED = False
NOISE = 0


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


# get points whose distance to selected point smaller than eps
def region_query(data, point_id, eps):
    n_points = data.shape[1]
    seeds = []
    for i in range(n_points):
        if euclidean_distance(data[:, point_id], data[:, i]) < eps:
            seeds.append(i)
    return seeds


def expand_cluster(data, cluster_result, point_id, cluster_id, eps, min_pts):
    """
    输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
    输出：能否成功分类
    """
    seeds = region_query(data, point_id, eps)
    if len(seeds) < min_pts:  # 不满足minPts条件的为噪声点
        cluster_result[point_id] = NOISE
        return False
    else:
        cluster_result[point_id] = cluster_id  # 划分到该簇
        for seed_id in seeds:
            cluster_result[seed_id] = cluster_id

        while len(seeds) > 0:  # 持续扩张
            current_point = seeds[0]
            query_results = region_query(data, current_point, eps)
            if len(query_results) >= min_pts:
                for result_point in query_results:
                    if cluster_result[result_point] == UNCLASSIFIED:
                        seeds.append(result_point)
                        cluster_result[result_point] = cluster_id
                    elif cluster_result[result_point] == NOISE:
                        cluster_result[result_point] = cluster_id
            seeds = seeds[1:]
        return True


def dbscan(cluster_data_set, eps, min_pts):
    """
    输入：数据集, 半径大小, 最小点个数
    输出：分类簇id
    """
    cluster_id = 1
    n_points = cluster_data_set.shape[1]
    cluster_result = [UNCLASSIFIED] * n_points
    for point_id in range(n_points):
        if cluster_result[point_id] == UNCLASSIFIED:
            if expand_cluster(cluster_data_set, cluster_result, point_id, cluster_id, eps, min_pts):
                cluster_id = cluster_id + 1
    return cluster_result, cluster_id - 1


# show cluster
def show_cluster(data, show_clusters, show_cluster_num):
    mat_clusters = mat(show_clusters).transpose()
    fig = plt.figure()
    scatter_colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(show_cluster_num + 1):
        color_sytle = scatter_colors[i % len(scatter_colors)]
        sub_cluster = data[:, nonzero(mat_clusters[:, 0].A == i)]
        ax.scatter(sub_cluster[0, :].flatten().A[0], sub_cluster[1, :].flatten().A[0], c=color_sytle, s=50)


if __name__ == '__main__':
    # step 1: load data
    print("step 1: loading data...")
    data_set = load_data('./dataSet/788points.txt', split_char=',')
    data_set = mat(data_set).transpose()
    # print(dataSet)

    # step 2: clustering...
    print("step 2: clustering...")

    start = time.clock()
    clusters, cluster_num = dbscan(data_set, 2, 15)
    end = time.clock()

    print('finish all in %s' % str(end - start))

    # step 3: show the result
    print("step 3: show the result...")
    print("cluster Numbers = ", cluster_num)
    # print(clusters)
    show_cluster(data_set, clusters, cluster_num)
    plt.show()
