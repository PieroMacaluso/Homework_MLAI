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


class Standardize(object):
    __mean = None
    __std = None

    def standardize(self, data):
        new = np.array(data)
        self.__mean = np.mean(new, axis=0)
        self.__std = np.std(new, axis=0)
        new = (new - self.__mean) / self.__std
        return new

    def destandardize(self, data):
        new = np.array(data)
        new = new * self.__std + self.__mean
        return new


# Start of the program
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
st = Standardize()
X_n = st.standardize(X)

pca = PCA()
pca.fit(X_n)

# Create the figure and set id of the image
plt.figure(figsize=(8, 4))
image_id = 1

X_t = pca.transform(X_n)

# Plot Original Image
plt.subplot(1, 5, 1)
orig = st.destandardize(X_n)/255
orig.astype(np.float32)
plt.imshow(orig[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
plt.title('Original Image')

# Plot 60PC Image
plt.subplot(1, 5, 2)
nComp = 60
X_b = np.dot(X_t[:, :nComp], pca.components_[:nComp, :])
orig = st.destandardize(X_b)/255
orig.astype(np.float32)
plt.imshow(orig[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
plt.title('60PC')

# Plot 6PC Image
plt.subplot(1, 5, 3)
nComp = 6
X_b = np.dot(X_t[:, :nComp], pca.components_[:nComp, :])
orig = st.destandardize(X_b)/255
orig.astype(np.float32)
plt.imshow(orig[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
plt.title('6PC')

# Plot 2PC Image
plt.subplot(1, 5, 4)
nComp = 2
X_b = np.dot(X_t[:, :nComp], pca.components_[:nComp, :])
orig = st.destandardize(X_b)/255
orig.astype(np.float32)
plt.imshow(orig[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
plt.title('2PC')

# Plot Last 6PC Image
plt.subplot(1, 5, 5)
nComp = -6
X_b = np.dot(X_t[:, nComp:], pca.components_[nComp:, :])
orig = st.destandardize(X_b)/255
orig.astype(np.float32)
plt.imshow(orig[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
plt.title('Last 6PC')

plt.show()
print("end")
