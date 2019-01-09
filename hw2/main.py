import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, datasets
import pandas as pd
from IPython.display import display, HTML
r_state=1


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


# Point 1 and 2
(X, y) = datasets.load_iris(return_X_y=True)
X = X[:, :2]  # we only take the first two features. Skip PCA

# Point 3
# train:val:test 5:2:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=r_state)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=r_state)

fig = plt.figure()
acc = np.empty(7)
C = 1e-3
C_best = 1e-3
A_best = 0
i = 1
xx, yy = make_meshgrid(X[:,0], X[:,1])
while C <= 1e3:
    clf = svm.LinearSVC(C=C, random_state=r_state)
    acc[i-1] = clf.fit(X_train, y_train).score(X_val, y_val)
    if acc[i-1]>A_best:
        C_best = C
        A_best = acc[i-1]
    ax = fig.add_subplot(3,3,i)
    plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('C=%2.2E A=%.2f ' % (C,acc[i-1]))

    C = C*10
    i = i+1
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.bar(np.array(['1e-3','1e-2','1e-1','1e0','1e1','1e2','1e3']), acc)
ax.set_xlabel('C')
ax.set_ylabel('Accuracy %')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# https://www.quora.com/What-is-the-purpose-for-using-slack-variable-in-SVM
fig.suptitle("Linear SVM with validation accuracy of %2.2f"%A_best, fontsize=14, fontweight='bold')
clf = svm.LinearSVC(C=C_best,random_state=r_state)
a= clf.fit(X_train, y_train).score(X_test, y_test)
plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('C=%2.2E A=%.2f ' % (C_best,a))

 

fig = plt.figure()
acc = np.empty(7)
C = 1e-3
C_best = 1e-3
A_best = 0
i = 1

xx, yy = make_meshgrid(X[:, 0], X[:, 1])
while C <= 1e3:
    clf = svm.SVC(kernel='rbf', C=C,random_state=r_state)
    acc[i - 1] = clf.fit(X_train, y_train).score(X_val, y_val)
    if acc[i-1]>A_best:
        C_best = C
        A_best = acc[i-1]
    ax = fig.add_subplot(3, 3, i)
    plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    Z = clf.predict(X_val)
    ax.set_title('C=%2.2E A=%.2f ' % (C, acc[i - 1]))

    C = C * 10
    i = i + 1

fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(np.array(['1e-3', '1e-2', '1e-1', '1e0', '1e1', '1e2', '1e3']), acc)
ax.set_xlabel('C')
ax.set_ylabel('Accuracy %')
fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# https://www.quora.com/What-is-the-purpose-for-using-slack-variable-in-SVM
fig.suptitle("RBF Kernel with validation accuracy of %2.2f"%A_best, fontsize=14, fontweight='bold')

clf = svm.SVC(kernel='rbf', C=C_best,random_state=r_state)
a = clf.fit(X_train, y_train).score(X_test, y_test)
plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('C=%2.2E A=%.2f ' % (C_best, a))

 

C = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
Gamma = np.array(
    [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
     1e10])
C_best = C.min()
G_best = Gamma.min()
A_best = 0
res = np.zeros([C.shape[0], Gamma.shape[0]])
for c, i in zip(C, range(0, C.shape[0])):
    for gamma, j in zip(Gamma, range(0, Gamma.shape[0])):
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=c, random_state=r_state)
        res[i, j] = clf.fit(X_train, y_train).score(X_val, y_val)
        if res[i,j]>A_best:
            C_best = c
            G_best = gamma
            A_best = res[i,j]

df = pd.DataFrame(res, index=C, columns=Gamma)


fig, ax = plt.subplots()
fig.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(res.reshape(len(C), len(Gamma)), interpolation='nearest', cmap=plt.get_cmap("hot"))
ax.set_xlabel('gamma')
ax.set_ylabel('C')
plt.colorbar()
ax.grid(True)
plt.xticks(np.arange(len(Gamma)), Gamma, rotation=90)
plt.yticks(np.arange(len(C)), C)

ax.set_title('Validation accuracy')
fig.show()



clf = svm.SVC(kernel='rbf', gamma=G_best, C=C_best, random_state=r_state)
acc = clf.fit(X_train, y_train).score(X_test, y_test)
fig, ax = plt.subplots()
plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_title('C=%2.2E Gamma=%2.2E  A=%.2f ' % (C_best, G_best, acc))
fig.show()
 

X_train = np.vstack([X_train, X_val])
y_train = np.concatenate([y_train, y_val])

C_best = C.min()
G_best = Gamma.min()
A_best = 0
kf = KFold(n_splits=5, shuffle=True, random_state=r_state)
scores = np.zeros([len(Gamma), len(C), kf.get_n_splits(X_train)])
for i, gamma in enumerate(Gamma):
    for j, c in enumerate(C):
        for k, (train_index, test_index) in enumerate(kf.split(X_train)):
            # print("TRAIN:", train_index, "TEST:", test_index)
            clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
            scores[i,j,k] = clf.fit(X_train[train_index], y_train[train_index]).score(X_train[test_index], y_train[test_index])
            if scores[i,j,k]>A_best:
                C_best = c
                G_best = gamma
                A_best = scores[i,j,k]

clf = svm.SVC(kernel='rbf', gamma=G_best, C=C_best,random_state=r_state)
acc = clf.fit(X_train, y_train).score(X_test, y_test)
fig, ax = plt.subplots()
fig.suptitle("K-Fold with validation accuracy of %2.2f"%A_best, fontsize=14, fontweight='bold')
plot_contours(ax, clf, xx, yy, cmap=plt.get_cmap("coolwarm"), alpha=0.8)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.get_cmap("coolwarm"), s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_title('C=%2.2E Gamma=%2.2E  A=%.2f ' % (C_best, G_best, acc))
fig.show()
plt.show()



print("end")
