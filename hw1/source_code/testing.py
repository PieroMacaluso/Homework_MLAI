import os
import sys
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# TODO: Change this to 252894 to obtain results of the report
r_state = None


def gen_matrix(path_dir, next_faster):
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
    """
    Thread function which fills a list with images as flattened arrays
    :param list_to_fill: list to fill
    :param pacs_dir: directory of PACS_homework
    :param folder_name: folder/category name of images
    :param color: label number
    :return: None
    """
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


def show_scatterplot(axs, first_pc, second_pc, x, y, data, color, str_x, str_y):
    """
    Prints a Scatter Plot
    :param axs: Axes
    Subplot Axes to modify
    :param first_pc: index of first PC to show
    :param second_pc: index of second PC to show
    :param x: All PCs
    :param y: All labels
    :param data: legend data
    :param color: legend color
    :param str_x: x labels
    :param str_y: y labels
    :return: None
    """
    for i in range(0, 4):
        axs.scatter(x[y == i, first_pc], x[y == i, second_pc],
                    c=color[i], label=data[i], s=10)
    axs.set_xlabel(str_x)
    axs.set_ylabel(str_y)
    axs.legend()
    print(str_x + " %d " % (max(x[:, first_pc]) - min(x[:, first_pc])) + str_y + " %d " % (
            max(x[:, second_pc]) - min(x[:, second_pc])))
    axs.grid(True)


def pca_custom(x):
    """
    Principal Component Analysis
    :param x: matrix with training data stored as flattened arrays in rows
    :return: projection matrix (with important dimensions first), variance
    and mean.
    """

    # get dimensions
    # num_data, dim = x.shape
    # center data
    mean_x = x.mean(axis=0)
    x = x - mean_x

    # PCA - compact trick used
    m = np.dot(x, x.T)  # covariance matrix, AA', not the A'A like usual
    e, ev = np.linalg.eigh(m)  # compute eigenvalues and eigenvectors
    tmp = np.dot(x.T, ev).T  # this is the compact trick
    v = tmp[::-1]  # reverse since last eigenvectors are the ones we want
    s = np.sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
    for i in range(v.shape[1]):
        v[:, i] /= s  # What for?

    # return the projection matrix, the variance and the mean
    return v, s, mean_x


def reconstruction(x_t, st, n_comp, comps):
    """
    Re-projection with n_comp
    :param x_t: Projection normalized
    :param st: Standard Scaler
    :param n_comp: number of components (negative for last n_comp components)
    :param comps: components
    :return: re-projected image
    """
    if n_comp > 0:
        x_b = np.dot(x_t[:, :n_comp], comps[:n_comp, :])
    else:
        # We want the last non-trivial components
        x_b = np.dot(x_t[:, n_comp:], comps[n_comp:, :])

    orig = st.inverse_transform(x_b) / 255
    return orig.astype(np.float64)


def show_reconstruction(ax, x_t, st, pca: PCA, pca_n, image_id):
    """
    Plot image reconstruction specifying the number of PC
    :param ax: axes to plot
    :param x_t: Projection
    :param st: Standar Scaler
    :param pca: pca
    :param pca_n: number of components (negative for last n_comp components)
    :param image_id: id of the image to re-project
    :return: None
    """
    if pca_n > 0:
        var = float(np.sum(pca.explained_variance_ratio_[:pca_n]) * 100)
        ax.set_title("%dPC var: %2.2f%%" % (pca_n, var))
    else:
        var = float(np.sum(pca.explained_variance_ratio_[pca_n:]) * 100)
        ax.set_title("Last %dPC var: %2.2f%%" % (pca_n * -1, var))
    x_pc = reconstruction(x_t, st, pca_n, pca.components_)
    ax.axis('off')
    ax.imshow(x_pc[image_id].reshape(227, 227, 3), vmin=0, vmax=1)


def plot_decision_boundaries_gnb(x_t, y, r_s, r_e, x_train, y_train, x_test, y_test, title, legend_data, legend_color):
    """
    Useful to plot decision boundaries of Gaussian Naive Bayes Classifier
    :param x_t: projection
    :param y: labels
    :param r_s: start of range of components
    :param r_e: end of range of components
    :param x_train: training data
    :param y_train: training label
    :param x_test: test data
    :param y_test: test label
    :param title: title to plot
    :param legend_data: legend data
    :param legend_color: legend color
    :return: None
    """
    clf = GaussianNB()
    clf.fit(x_train[:, r_s:r_e], y_train)
    x_min, x_max = x_t[:, r_s].min() - 1, x_t[:, r_s].max() + 1
    y_min, y_max = x_t[:, (r_e - 1)].min() - 1, x_t[:, (r_e - 1)].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))

    fig, axarr = plt.subplots(1, 1)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
    cmap_plot = matplotlib.colors.ListedColormap(legend_color)

    axarr.contourf(xx, yy, z, alpha=0.5, norm=norm, cmap=cmap_plot)
    for i in range(0, 4):
        axarr.scatter(x_t[y == i, r_s], x_t[y == i, (r_e - 1)], c=legend_color[i], label=legend_data[i], s=20)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    axarr.set_title(
        "Accuracy over (%d points): %2.2f%%" % (x_test.shape[0], clf.score(x_test[:, r_s:r_e], y_test) * 100))
    axarr.set_xlabel("%d° PC" % (r_s + 1))
    axarr.set_ylabel("%d° PC" % (r_e - 1))
    axarr.legend()
    fig.show()
    print(title)
    print("- Accuracy over (%d points): %2.2f%%" % (x_test.shape[0], clf.score(x_test[:, r_s:r_e], y_test) * 100))


def classification(x_n, x_t, y):
    # 1.3 CLASSIFICATION
    x_train, x_test, y_train, y_test = train_test_split(x_n, y, random_state=r_state)
    # Classifier on original data
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    res1 = clf.fit(x_train, y_train).score(x_test, y_test) * 100

    x_train, x_test, y_train, y_test = train_test_split(x_t, y, random_state=r_state)
    clf = GaussianNB()
    res2 = clf.fit(x_train[:, 0:2], y_train).score(x_test[:, 0:2], y_test) * 100
    clf = GaussianNB()
    res3 = clf.fit(x_train[:, 2:4], y_train).score(x_test[:, 2:4], y_test) * 100
    return res1, res2, res3


def main():
    # 1.1 DATA PREPARATION
    if len(sys.argv) == 3 and sys.argv[1] == '-s':
        x, y = gen_matrix(sys.argv[2], True)
    elif len(sys.argv) == 3 and sys.argv[1] == '-n':
        x, y = gen_matrix(sys.argv[2], False)
    elif len(sys.argv) == 4 and sys.argv[1] == '-l':
        x, y = load_matrix(sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("$ python main.py -n <PACS_homework folder>")
        print("-- Loads Image from the specified Folder and execute the code")
        print("$ python main.py -s <PACS_homework folder>")
        print("-- Loads Image from the specified Folder and save files in data.npy and label.npy before executing the "
              "code")
        print("$ python main.py -l <data.npy file> <label.npy file>")
        print("-- Loads Image from data.npy and label.npy files and execute the code")
        exit(-1)

    legend_data = ["Dog", "Guitar", "House", "Person"]
    legend_color = ["red", "blue", "green", "orange"]

    # 1.2 PRINCIPAL COMPONENT VISUALIZATION
    # Standardize the features
    st = StandardScaler()
    st.fit(x)
    x_n = st.transform(x)
    pca = PCA()
    pca.fit(x_n)
    x_t = pca.transform(x_n)

    times = 500
    res1 = np.empty(times)
    res2 = np.empty(times)
    res3 = np.empty(times)
    for index in range(0, times):
        (res1[index], res2[index], res3[index]) = classification(x_n, x_t, y)
        print(index, "\t", res1[index], "\t", res2[index], "\t", res3[index])
    print("NORMALIZED DATA")
    print("Average : %f%%" % np.mean(res1))
    print("Variance: %f" % np.std(res1, ddof=1))
    print("PROJECTION 1-2PC")
    print("Average: %f%%" % np.mean(res2))
    print("Variance: %f" % np.std(res2, ddof=1))
    print("PROJECTION 3-4PC")
    print("Average: %f%%" % np.mean(res3))
    print("Variance: %f" % np.std(res3, ddof=1))


if __name__ == "__main__":
    main()
