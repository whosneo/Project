import os
import matplotlib.pyplot as plt
from numpy import *


def convert(file_name, write_file_name, split_char=','):
    file = open(file_name)
    write_file = open(write_file_name, mode="w")
    write_file.write("ID_RECORD,latitude,longitude,date_time\n")
    # for i in range(6):
    #     file.readline()
    i = 0
    while True:
        line = file.readline()
        if not line:
            break
        line_arr = line.strip().split(split_char)
        i = i + 1
        write_file.write(str(i) + "," + line_arr[0] + "," + line_arr[1] + "," + line_arr[5] + " " + line_arr[6] + "\n")
        # write_file.write("{lat:" + line_arr[1] + ", lng:" + line_arr[2] + "},\n")
    file.close()
    write_file.close()


def get_every_file(dir_path, suffix='.plt'):
    new_file_list = []
    file_list = os.listdir(dir_path)

    for file in file_list:
        full_file = os.path.join(dir_path, file)
        if os.path.isfile(full_file):
            if os.path.splitext(full_file)[1] == suffix:
                new_file_list.append(full_file)

    return new_file_list


def main(dir_path):
    file_list = get_every_file(dir_path, suffix='.plt')
    file_list.sort()

    for file in file_list:
        file_path = os.path.split(file)[0]
        file_name = os.path.split(file)[1]
        convert(file_path + "/" + file_name, "/home/question/converted/" + file_name)
        # break


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


if __name__ == "__main__":
    path = r'/home/question/Trajectory'
    main(path)
    # convert("/home/question/save.plt", "/home/question/result.txt")
    # "/home/question/converted/20090702022530.plt"
    # show_cluster()
