from sklearn import datasets, svm
import numpy as np
from sklearn.model_selection import train_test_split, KFold

r_state = None


def knn_rbf_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=r_state)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4, random_state=r_state)
    x_train = np.vstack([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])
    c = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    gamma = np.array(
        [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    # 5-fold validation
    c_best = c.min()
    g_best = gamma.min()
    a_best = 0
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=r_state)
    scores = np.zeros([len(gamma), len(c)])
    for i, gamma_i in enumerate(gamma):
        for j, c_i in enumerate(c):
            temp = np.zeros(kf.n_splits)
            for k_i, (train_index, test_index) in enumerate(kf.split(x_train)):
                clf = svm.SVC(kernel='rbf', gamma=gamma_i, C=c_i)
                clf.fit(x_train[train_index], y_train[train_index])
                temp[k_i] = clf.score(x_train[test_index], y_train[test_index]) * 100
            acc_av = np.average(temp)
            if acc_av > a_best:
                c_best = c_i
                g_best = gamma_i
                a_best = acc_av

    clf = svm.SVC(kernel='rbf', gamma=g_best, C=c_best)
    f_score = clf.fit(x_train, y_train).score(x_test, y_test) * 100
    return a_best, f_score


def rbf_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=r_state)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4, random_state=r_state)
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

    clf = svm.SVC(kernel='rbf', gamma=g_best, C=c_best, random_state=r_state)
    f_score = clf.fit(x_train, y_train).score(x_test, y_test) * 100
    return a_best, f_score


def linear_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=r_state)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.4, random_state=r_state)
    acc = np.empty(7)
    c_i = 1e-3
    c_best = 1e-3
    a_best = 0
    i = 1
    while c_i <= 1e3:
        clf = svm.LinearSVC(C=c_i, random_state=r_state)
        acc[i - 1] = clf.fit(x_train, y_train).score(x_val, y_val) * 100
        if acc[i - 1] > a_best:
            c_best = c_i
            a_best = acc[i - 1]
        c_i = c_i * 10
        i = i + 1

    clf = svm.LinearSVC(C=c_best, random_state=r_state)
    f_score = clf.fit(x_train, y_train).score(x_test, y_test) * 100
    return a_best, f_score


def main():
    (x, y) = datasets.load_iris(return_X_y=True)
    x = x[:, :2]
    arr = np.empty(200)
    scores = np.empty(200)
    for index in range(0, 200):
        # (arr[index], scores[index]) = linear_test(x, y)
        (arr[index], scores[index]) = knn_rbf_test(x, y)
        # (arr[index], scores[index]) = rbf_test(x, y)
        print(index, "\t", arr[index], "1t", scores[index])
    print("Average : %f%%" % np.mean(arr))
    print("Variance: %f" % np.std(arr))
    print("Min: %f%%" % np.min(arr))
    print("Max: %f" % np.max(arr))
    print("Average: %f%%" % np.mean(scores))
    print("Variance: %f" % np.std(scores))
    print("Min: %f%%" % np.min(scores))
    print("Max: %f" % np.max(scores))


if __name__ == "__main__":
    main()
