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
from sklearn import svm, datasets
import matplotlib

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


def plot_data(ax, x, y, xx, yy, clf):
    label = ("setosa", "versicolor", "virginica")
    color = ["red", "blue", "lightgreen"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    cmap_plot = matplotlib.colors.ListedColormap(color)
    plot_contours(ax, clf, xx, yy, norm=norm, cmap=cmap_plot, alpha=0.4)
    for j in range(0, 3):
        ax.scatter(x[y == j, 0], x[y == j, 1], c=color[j], label=label[j], s=20, edgecolors='k')


def main():
    # Section 1
    (x, y) = datasets.load_iris(return_X_y=True)
    x = x[:, :2]  # we only take the first two features. Skip PCA
    # train:val:test 5:2:3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=r_state)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4, random_state=r_state)

    # Fig01a
    fig = plt.figure(figsize=(8, 12))
    acc = np.empty(7)
    c_i = 1e-3
    c_best = 1e-3
    a_best = 0
    i = 1
    xx, yy = make_meshgrid(x[:, 0], x[:, 1])
    while c_i <= 1e3:
        clf = svm.LinearSVC(C=c_i, random_state=r_state)
        acc[i - 1] = clf.fit(x_train, y_train).score(x_val, y_val) * 100
        if acc[i - 1] > a_best:
            c_best = c_i
            a_best = acc[i - 1]
        ax = fig.add_subplot(4, 2, i)
        plot_data(ax, x_train, y_train, xx, yy, clf)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.legend()
        ax.set_title('C=%2.2E A=%2.1f%% ' % (c_i, acc[i - 1]))
        c_i = c_i * 10
        i = i + 1

    fig.suptitle("Linear SVM - C tuning - C_best = %2.2E A_best = %2.1f%%" % (c_best, a_best), fontsize=14,
                 fontweight='bold')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig01a.png", transparent=False, dpi=300)
    fig.show()

    # fig01b
    fig = plt.figure(figsize=(7, 5))
    fig.suptitle("Linear SVM - C tuning - C_best = %2.2E A_best = %2.2f%%" % (c_best, a_best), fontsize=14,
                 fontweight='bold')
    ax = fig.add_subplot(1, 1, 1)
    V = np.array(['1e-3', '1e-2', '1e-1', '1e0', '1e1', '1e2', '1e3'])
    ax.bar(V, acc)
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy %')
    plt.grid(True)
    ax.set_yticks(acc)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig01b.png", transparent=False, dpi=300)
    fig.show()

    # fig01c
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle("Linear SVM with validation accuracy of %2.2f%%" % a_best, fontsize=14, fontweight='bold')
    clf = svm.LinearSVC(C=c_best, random_state=r_state)
    a = clf.fit(x_train, y_train).score(x_test, y_test) * 100
    plot_data(ax, x_test, y_test, xx, yy, clf)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('C=%2.2E A=%2.2f%% ' % (c_best, a))
    ax.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig01c.png", transparent=False, dpi=300)
    fig.show()

    # fig02a
    fig = plt.figure(figsize=(8, 12))
    acc = np.empty(7)
    c_i = 1e-3
    c_best = 1e-3
    a_best = 0
    i = 1
    xx, yy = make_meshgrid(x[:, 0], x[:, 1])
    while c_i <= 1e3:
        clf = svm.SVC(kernel='rbf', C=c_i, random_state=r_state)
        acc[i - 1] = clf.fit(x_train, y_train).score(x_val, y_val) * 100
        if acc[i - 1] > a_best:
            c_best = c_i
            a_best = acc[i - 1]
        ax = fig.add_subplot(4, 2, i)
        plot_data(ax, x_train, y_train, xx, yy, clf)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.legend()
        ax.set_title('C=%2.2E A=%.2f%%' % (c_i, acc[i - 1]))
        c_i = c_i * 10
        i = i + 1

    fig.suptitle("RBF Kernel - C/G tuning - C_best=%2.2E A_best=%2.2f%%" % (c_best, a_best), fontsize=14,
                 fontweight='bold')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig02a.png", transparent=False, dpi=300)
    fig.show()

    # fig02b
    fig = plt.figure()
    fig.suptitle("RBF Kernel - C/G tuning - C_best=%2.2E A_best=%2.2f%%" % (c_best, a_best), fontsize=14,
                 fontweight='bold')
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.array(['1e-3', '1e-2', '1e-1', '1e0', '1e1', '1e2', '1e3']), acc)
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy %')
    plt.grid(True)
    ax.set_yticks(acc)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig02b.png", transparent=False, dpi=300)
    fig.show()

    # fig02c
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig.suptitle("RBF Kernel with validation accuracy of %2.2f%%" % a_best, fontsize=14, fontweight='bold')

    clf = svm.SVC(kernel='rbf', C=c_best, random_state=r_state)
    a = clf.fit(x_train, y_train).score(x_test, y_test) * 100
    plot_data(ax, x_test, y_test, xx, yy, clf)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('C=%2.2E A=%2.2f%% ' % (c_best, a))
    ax.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig02c.png", transparent=False, dpi=300)
    fig.show()

    # Grid Search of C and Gamma
    c = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    gamma = np.array(
        [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    c_best = c.min()
    g_best = gamma.min()
    a_best = 0
    res = np.zeros([c.shape[0], gamma.shape[0]])
    for c_i, i in zip(c, range(0, c.shape[0])):
        for gamma_i, j in zip(gamma, range(0, gamma.shape[0])):
            clf = svm.SVC(kernel='rbf', gamma=gamma_i, C=c_i, random_state=r_state)
            res[i, j] = clf.fit(x_train, y_train).score(x_val, y_val) * 100
            if res[i, j] > a_best:
                c_best = c_i
                g_best = gamma_i
                a_best = res[i, j]

    # fig02d
    fig, ax = plt.subplots()
    fig.suptitle("RBF Kernel - C/G tuning - Validation Accuracy", fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(res.reshape(len(c), len(gamma)), interpolation='nearest', cmap=plt.get_cmap("hot"))
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    plt.colorbar()
    ax.grid(True)
    ax.ticklabel_format(axis='both', style='sci')
    plt.xticks(np.arange(len(gamma)), gamma, rotation=90)
    plt.yticks(np.arange(len(c)), c)
    ax.set_title('C_best=%2.2E G_best=%2.2E A_best=%2.2f%%' % (c_best, g_best, a_best))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig02d.png", transparent=False, dpi=300)
    fig.show()

    # fig02e
    clf = svm.SVC(kernel='rbf', gamma=g_best, C=c_best, random_state=r_state)
    acc = clf.fit(x_train, y_train).score(x_test, y_test) * 100
    fig, ax = plt.subplots()
    fig.suptitle("RBF Kernel with validation accuracy of %2.2f%%" % a_best, fontsize=14, fontweight='bold')
    plot_data(ax, x_test, y_test, xx, yy, clf)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('C=%2.2E Gamma=%2.2E  A=%2.2f%%' % (c_best, g_best, acc))
    ax.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig02e.png", transparent=False, dpi=300)
    fig.show()

    # Merging training and validation sets
    x_train = np.vstack([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])
    # 5-fold validation
    c_best = c.min()
    g_best = gamma.min()
    a_best = 0
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=r_state)
    res = np.zeros([c.shape[0], gamma.shape[0]])
    for i, c_i in enumerate(c):
        for j, gamma_i in enumerate(gamma):
            temp = np.zeros(kf.n_splits)
            for k_i, (train_index, test_index) in enumerate(kf.split(x_train)):
                clf = svm.SVC(kernel='rbf', gamma=gamma_i, C=c_i)
                clf.fit(x_train[train_index], y_train[train_index])
                temp[k_i] = clf.score(x_train[test_index], y_train[test_index]) * 100
            res[i, j] = np.average(temp)
            if res[i, j] > a_best:
                c_best = c_i
                g_best = gamma_i
                a_best = res[i, j]

    clf = svm.SVC(kernel='rbf', gamma=g_best, C=c_best, random_state=r_state)
    acc = clf.fit(x_train, y_train).score(x_test, y_test) * 100

    # fig03a
    fig, ax = plt.subplots()
    fig.suptitle("RBF Kernel - C/G tuning - 5-fold Crossvalidation", fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(res.reshape(len(c), len(gamma)), interpolation='nearest', cmap=plt.get_cmap("hot"))
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    plt.colorbar()
    ax.grid(True)
    ax.ticklabel_format(axis='both', style='sci')
    plt.xticks(np.arange(len(gamma)), gamma, rotation=90)
    plt.yticks(np.arange(len(c)), c)
    ax.set_title('C_best=%2.2E G_best=%2.2E A_best=%2.2f%%' % (c_best, g_best, a_best))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig03a.png", transparent=False, dpi=300)
    fig.show()

    # fig03b
    fig, ax = plt.subplots()
    fig.suptitle("RBF Kernel K-Fold with validation accuracy of %2.2f%%" %
                 a_best, fontsize=14, fontweight='bold')
    plot_data(ax, x_test, y_test, xx, yy, clf)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('C=%2.2E Gamma=%2.2E  A=%2.2f%% ' % (c_best, g_best, acc))
    ax.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0.03, 1, 0.95])
    # plt.savefig("../report/img/fig03b.png", transparent=False, dpi=300)
    fig.show()
    plt.show()

    print("end")


if __name__ == "__main__":
    main()
