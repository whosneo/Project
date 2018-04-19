from numpy import *
import matplotlib.pyplot as plt


def convert(file_name, write_file_name, split_char=','):
    file = open(file_name)
    write_file = open(write_file_name, mode="a")
    write_file.write("latitude,longitude,date_time\n")
    write_map_file = open("./dataSet/data_converted_map.csv", mode="a")
    for i in range(6):
        file.readline()
    while True:
        line = file.readline()
        if not line:
            break
        line_arr = line.strip().split(split_char)
        write_file.write(line_arr[0] + "," + line_arr[1] + "," + line_arr[5] + " " + line_arr[6] + "\n")
        # write_file.write(line_arr[0] + "," + line_arr[1] + "\n")
        write_map_file.write("{lat:" + line_arr[0] + ", lng:" + line_arr[1] + "},\n")
    file.close()
    write_file.close()
    write_map_file.close()


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


if __name__ == "__main__":
    # path = r'/home/question/Trajectory'
    # main(path)
    # convert("/home/question/save.plt", "/home/question/result.txt")
    # "/home/question/converted/20090702022530.plt"
    # convert("./dataSet/20081028003826.plt", "./dataSet/data_converted.csv")

    convert("./dataSet/20081111001704.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081112023003.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081112091400.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081113034608.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081114015255.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081114101436.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081115010133.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081116085532.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081117051133.plt", "./dataSet/data_converted.csv")
    convert("./dataSet/20081117155223.plt", "./dataSet/data_converted.csv")

    # plt.figure(figsize=(8, 6))
    # x = [1, 2, 3]
    # plt.plot(x, x, 'o')
    # plt.ylim(0, 4)
    # plt.title('Estimated number of clusters')
    # plt.show()
