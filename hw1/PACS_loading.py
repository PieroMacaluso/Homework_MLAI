from PIL import Image
import numpy as np
import threading
import os


def gen_matrix(path_dir):
    o_dog = [np.empty((0, 154587)), np.empty((0, 1))]
    o_guitar = [np.empty((0, 154587)), np.empty((0, 1))]
    o_house = [np.empty((0, 154587)), np.empty((0, 1))]
    o_person = [np.empty((0, 154587)), np.empty((0, 1))]

    threads = [
        threading.Thread(target=add_data_to_matrix,
                         args=(o_dog, path_dir, "dog", 0)),
        threading.Thread(target=add_data_to_matrix,
                         args=(o_guitar, path_dir, "guitar", 1)),
        threading.Thread(target=add_data_to_matrix,
                         args=(o_house, path_dir, "house", 2)),
        threading.Thread(target=add_data_to_matrix,
                         args=(o_person, path_dir, "person", 3))
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    x = np.vstack((o_dog[0], o_guitar[0], o_house[0], o_person[0]))
    y = np.vstack((o_dog[1], o_guitar[1], o_house[1], o_person[1])).ravel()
    np.save(path_dir + "data.npy", x)
    np.save(path_dir + "label.npy", y)
    return x, y


def load_matrix(path_data, path_label):
    x = np.load(path_data)
    y = np.load(path_label)
    return x, y


def add_data_to_matrix(list_to_fill, pacs_dir, folder_name, color):
    print('Loading ' + folder_name + '...')
    path_dir = pacs_dir + folder_name
    directory = os.fsencode(path_dir)
    lst = os.listdir(directory)
    lst.sort()
    for file in lst:
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            img_data = np.asarray(Image.open(os.path.join(path_dir, filename)))
            list_to_fill[0] = np.vstack((list_to_fill[0], img_data.ravel()))
            list_to_fill[1] = np.vstack((list_to_fill[1], color))
            # print(filename)
            continue
        else:
            continue
    print(folder_name + ' loaded correctly!')
