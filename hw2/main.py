"""
Homework 2 Code
Author: Piero Macaluso s252894

Course: Machine Learning and Artificial Intelligence
Academic Year: 2018/2019
University: Polytechnic University of Turin
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, datasets
import pandas as pd
from IPython.display import display, HTML

# TODO: Change this to 252894 to obtain results of the report
r_state = 252894


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def main():
    # Section 1
    (x, y) = datasets.load_iris(return_X_y=True)
    x = x[:, :2]  # we only take the first two features. Skip PCA
    # train:val:test 5:2:3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=r_state)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4, random_state=r_state)

    fig = plt.figure()  # 1
    acc = np.empty(7)
    c = 1e-3
    c_best = 1e-3
    a_best = 0
    i = 1
    xx, yy = make_meshgrid(x[:, 0], x[:, 1])
    while c <= 1e3:
        clf = svm.LinearSVC(C=c, random_state=r_state)
        acc[i - 1] = clf.fit(x_train, y_train).score(x_val, y_val)
        if acc[i - 1] > a_best:
            c_best = c
            a_best = acc[i - 1]
        ax = fig.add_subplot(3, 3, i)
        plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train,
                   cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('C=%2.2E A=%.2f ' % (c, acc[i - 1]))

        c = c * 10
        i = i + 1
    fig.suptitle("Linear SVM - C tuning - C_best = %2.2E A_best = %2.2f" %
                 (c_best, a_best), fontsize=14, fontweight='bold')
    plt.rcParams["figure.figsize"][0] = 6.25
    plt.savefig("report/img/fig01d.png", transparent=False, dpi=300, bbox_inches="tight")
    fig.show()

    # TODO: Add value on top of bar
    fig = plt.figure()  # 2
    fig.suptitle("Linear SVM - C tuning - C_best = %2.2E A_best = %2.2f" %
                 (c_best, a_best), fontsize=14, fontweight='bold')
    ax = fig.add_subplot(1, 1, 1)
    V = np.array(['1e-3', '1e-2', '1e-1', '1e0', '1e1', '1e2', '1e3'])
    ax.bar(V, acc)
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy %')
    plt.grid(True)
    fig.show()

    fig = plt.figure()  # 3
    ax = fig.add_subplot(1, 1, 1)

    # https://www.quora.com/What-is-the-purpose-for-using-slack-variable-in-SVM
    fig.suptitle("Linear SVM with validation accuracy of %2.2f" %
                 a_best, fontsize=14, fontweight='bold')
    clf = svm.LinearSVC(C=c_best, random_state=r_state)
    a = clf.fit(x_train, y_train).score(x_test, y_test)
    plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test,
               cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('C=%2.2E A=%.2f ' % (c_best, a))
    fig.show()

    fig = plt.figure()  # 4
    acc = np.empty(7)
    c = 1e-3
    c_best = 1e-3
    a_best = 0
    i = 1

    xx, yy = make_meshgrid(x[:, 0], x[:, 1])
    while c <= 1e3:
        clf = svm.SVC(kernel='rbf', C=c, random_state=r_state)
        acc[i - 1] = clf.fit(x_train, y_train).score(x_val, y_val)
        if acc[i - 1] > a_best:
            c_best = c
            a_best = acc[i - 1]
        ax = fig.add_subplot(3, 3, i)
        plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train,
                   cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        Z = clf.predict(x_val)
        ax.set_title('C=%2.2E A=%.2f ' % (c, acc[i - 1]))

        c = c * 10
        i = i + 1

    fig.suptitle("RBF Kernel - C/G tuning - C_best=%2.2E A_best=%2.2f" %
                 (c_best, a_best), fontsize=14, fontweight='bold')
    fig.show()

    fig = plt.figure()  # 5
    fig.suptitle("RBF Kernel - C/G tuning - C_best=%2.2E A_best=%2.2f" %
                 (c_best, a_best), fontsize=14, fontweight='bold')
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.array(['1e-3', '1e-2', '1e-1', '1e0', '1e1', '1e2', '1e3']), acc)
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy %')
    fig.show()

    fig = plt.figure()  # 6
    ax = fig.add_subplot(1, 1, 1)

    # https://www.quora.com/What-is-the-purpose-for-using-slack-variable-in-SVM
    fig.suptitle("RBF Kernel with validation accuracy of %2.2f" %
                 a_best, fontsize=14, fontweight='bold')

    clf = svm.SVC(kernel='rbf', C=c_best, random_state=r_state)
    a = clf.fit(x_train, y_train).score(x_test, y_test)
    plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test,
               cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('C=%2.2E A=%.2f ' % (c_best, a))

    c = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    Gamma = np.array(
        [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
         1e10])
    c_best = c.min()
    G_best = Gamma.min()
    a_best = 0
    res = np.zeros([c.shape[0], Gamma.shape[0]])
    for c, i in zip(c, range(0, c.shape[0])):
        for gamma, j in zip(Gamma, range(0, Gamma.shape[0])):
            clf = svm.SVC(kernel='rbf', gamma=gamma, C=c, random_state=r_state)
            res[i, j] = clf.fit(x_train, y_train).score(x_val, y_val)
            if res[i, j] > a_best:
                c_best = c
                G_best = gamma
                a_best = res[i, j]

    df = pd.DataFrame(res, index=c, columns=Gamma)

    fig, ax = plt.subplots()  # 7
    fig.suptitle("RBF Kernel - C/G tuning - C_best=%2.2E A_best=%2.2f" %
                 (c_best, a_best), fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(res.reshape(len(c), len(Gamma)),
               interpolation='nearest', cmap=plt.get_cmap("hot"))
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    plt.colorbar()
    ax.grid(True)
    plt.xticks(np.arange(len(Gamma)), Gamma, rotation=90)
    plt.yticks(np.arange(len(c)), c)

    ax.set_title('Validation accuracy')
    fig.show()

    clf = svm.SVC(kernel='rbf', gamma=G_best, C=c_best, random_state=r_state)

    acc = clf.fit(x_train, y_train).score(x_test, y_test)
    fig, ax = plt.subplots()
    fig.suptitle("RBF Kernel with validation accuracy of %2.2f" %
                 a_best, fontsize=14, fontweight='bold')
    plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test,
               cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('C=%2.2E Gamma=%2.2E  A=%.2f ' % (c_best, G_best, acc))
    fig.show()

    x_train = np.vstack([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])

    c_best = c.min()
    G_best = Gamma.min()
    a_best = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=r_state)
    scores = np.zeros([len(Gamma), len(c), kf.get_n_splits(x_train)])
    for i, gamma in enumerate(Gamma):
        for j, c in enumerate(c):
            for k, (train_index, test_index) in enumerate(kf.split(x_train)):
                # print("TRAIN:", train_index, "TEST:", test_index)
                clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
                scores[i, j, k] = clf.fit(x_train[train_index], y_train[train_index]).score(
                    x_train[test_index], y_train[test_index])
                if scores[i, j, k] > a_best:
                    c_best = c
                    G_best = gamma
                    a_best = scores[i, j, k]

    clf = svm.SVC(kernel='rbf', gamma=G_best, C=c_best, random_state=r_state)
    acc = clf.fit(x_train, y_train).score(x_test, y_test)
    fig, ax = plt.subplots()
    fig.suptitle("RBF Kernel K-Fold with validation accuracy of %2.2f" %
                 a_best, fontsize=14, fontweight='bold')
    plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test,
               cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('C=%2.2E Gamma=%2.2E  A=%.2f ' % (c_best, G_best, acc))
    fig.show()
    plt.show()

    print("end")


if __name__ == "__main__":
    main()
