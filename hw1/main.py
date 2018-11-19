from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def gen_matrix(path_dir):

    o_dog = [np.empty((0, 154587)), np.empty((0, 1))]
    o_guitar = [np.empty((0, 154587)), np.empty((0, 1))]
    o_house = [np.empty((0, 154587)), np.empty((0, 1))]
    o_person = [np.empty((0, 154587)), np.empty((0, 1))]

    threads = [
        threading.Thread(target=add_data_to_matrix,
                         args=(o_dog, path_dir, "dog", "red")),
        threading.Thread(target=add_data_to_matrix,
                         args=(o_guitar, path_dir, "guitar", "yellow")),
        threading.Thread(target=add_data_to_matrix,
                         args=(o_house, path_dir, "house", "blue")),
        threading.Thread(target=add_data_to_matrix,
                         args=(o_person, path_dir, "person", "green"))
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    X = np.vstack((o_dog[0], o_guitar[0], o_house[0], o_person[0]))
    y = np.vstack((o_dog[1], o_guitar[1], o_house[1], o_person[1]))
    np.save(path_dir + "data.npy", X)
    np.save(path_dir + "label.npy", y)
    return X, y


def load_matrix(path_data, path_label):
    X = np.load(path_data)
    y = np.load(path_label)
    return X, y


def add_data_to_matrix(list, PACS_dir, folder_name, color):
    path_dir = PACS_dir + folder_name
    directory = os.fsencode(path_dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            img_data = np.asarray(Image.open(os.path.join(path_dir, filename)))
            list[0] = np.vstack((list[0], img_data.ravel()))
            list[1] = np.vstack((list[1], color))
            continue
        else:
            continue


def normalize_data(df: np.array) -> np.array:
    """ 
    Normalize NumPy array feature by feature 

    Parameters
    ----------
    df : np.array
        The dataset which has to be normalized
    Return
    ------
    out : np.array
        The dataset normalized
    """
    df = (df - np.mean(df, axis=0)) / np.std(df, axis=0)
    return df


if (len(sys.argv) != 3 and len(sys.argv) != 4):
    print("Error: wrong number of parameter")
    exit(-1)

if (sys.argv[1] == '-d' and len(sys.argv) == 3):
    X, y = gen_matrix(sys.argv[2])
elif (sys.argv[1] == '-l' and len(sys.argv) == 4):
    X, y = load_matrix(sys.argv[2], sys.argv[3])
else:
    print("Error: wrong number of parameter")
    exit(-1)

# Standardize the features
scaler = StandardScaler()
X_n = scaler.fit_transform(X)
X_n = normalize_data(X)

pca_60 = PCA(60)
X_60 = pca_60.fit_transform(X_n)
app_60 = pca_60.inverse_transform(X_60)

pca_6 = PCA(6)
X_6 = pca_6.fit_transform(X_n)
app_6 = pca_6.inverse_transform(X_6)

pca_2 = PCA(2)
X_2 = pca_2.fit_transform(X_n)
app_2 = pca_2.inverse_transform(X_2)

pca_l6 = PCA(154587)
X_l6 = np.transpose(pca_l6.fit_transform(X_n)[-6:])
app_l6 = pca_l6.inverse_transform(X_l6)

plt.figure(figsize=(8, 4))
image_id = 1000

# Original Image
plt.subplot(1, 5, 1)
orig = scaler.inverse_transform(X_n).astype(np.uint8)
plt.imshow(orig[image_id].reshape(227, 227, 3))
plt.title('Original Image', fontsize=20)

# 60PC Image
plt.subplot(1, 5, 2)
orig = scaler.inverse_transform(app_60).astype(np.uint8)
plt.imshow(orig[image_id].reshape(227, 227, 3))
plt.title('60PC', fontsize=20)

# 6PC Image
plt.subplot(1, 5, 3)
orig = scaler.inverse_transform(app_6).astype(np.uint8)
plt.imshow(orig[image_id].reshape(227, 227, 3))
plt.title('6PC', fontsize=20)

# 2PC Image
plt.subplot(1, 5, 4)
orig = scaler.inverse_transform(app_2).astype(np.uint8)
plt.imshow(orig[image_id].reshape(227, 227, 3))
plt.title('2PC', fontsize=20)

# Last 6PC Image
plt.subplot(1, 5, 5)
orig = scaler.inverse_transform(app_l6).astype(np.uint8)
plt.imshow(orig[image_id].reshape(227, 227, 3))
plt.title('2PC', fontsize=20)

# plt.scatter(X_red[:, 10], X_red[:, 11], c=y)
plt.show()
print("end")
