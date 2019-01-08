import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, datasets
import pandas as pd
from IPython.display import display, HTML


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
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. Skip PCA
y = iris.target

# Point 3
# train:val:test 5:2:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=1)
'''
fig = plt.figure()
acc = np.empty(7)
C = 1e-3
i = 1
xx, yy = make_meshgrid(X[:,0], X[:,1])
while C <= 1e3:
    clf = svm.LinearSVC(C=C)
    clf.fit(X_train, y_train)
    ax = fig.add_subplot(3,3,i)
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    Z = clf.predict(X_val)
    acc[i-1] = ((Z.shape[0] - (Z==y_val).sum())*100/Z.shape[0])
    ax.set_title('C=%2.2E A=%.2f %%' % (C,acc[i-1]))

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
C = 1e-2
clf = svm.LinearSVC(C=C)
clf.fit(X_train, y_train)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())
Z = clf.predict(X_test)
a= ((Z.shape[0] - (Z==y_test).sum())*100/Z.shape[0])
ax.set_title('C=%2.2E A=%.2f %%' % (C,a))

plt.show()
'''
fig = plt.figure()
acc = np.empty(7)
C = 1e-3
i = 1

xx, yy = make_meshgrid(X[:, 0], X[:, 1])
while C <= 1e3:
    clf = svm.SVC(kernel='rbf', C=C)
    clf.fit(X_train, y_train)
    ax = fig.add_subplot(3, 3, i)
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    Z = clf.predict(X_val)
    acc[i - 1] = ((Z.shape[0] - (Z == y_val).sum()) * 100 / Z.shape[0])
    ax.set_title('C=%2.2E A=%.2f %%' % (C, acc[i - 1]))

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
C = 1e-1
clf = svm.SVC(kernel='rbf', C=C)
clf.fit(X_train, y_train)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())
Z = clf.predict(X_test)
a = ((Z.shape[0] - (Z == y_test).sum()) * 100 / Z.shape[0])
ax.set_title('C=%2.2E A=%.2f %%' % (C, a))

C = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
Gamma = np.array(
    [1e-10,1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,1e10])
res = np.zeros([C.shape[0], Gamma.shape[0]])
for c, i in zip(C, range(0, C.shape[0])):
    for gamma, j in zip(Gamma, range(0, Gamma.shape[0])):
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
        clf.fit(X_train, y_train)
        Z = clf.predict(X_val)
        res[i, j] = ((Z.shape[0] - (Z == y_val).sum()) * 100 / Z.shape[0])
fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, frame_on=False)  # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

maxx = res.max()
df = pd.DataFrame(res, index=C, columns=Gamma)

display(df)
fig.show()
plt.show()
plt.figure()
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(res.reshape(len(C),len(Gamma)), interpolation='nearest', cmap=plt.get_cmap("hot"))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.grid(True)
plt.xticks(np.arange(len(Gamma)), Gamma, rotation=90)
plt.yticks(np.arange(len(C)), C)

plt.title('Validation accuracy')
plt.show()

c = 0.001
gamma = 1000

clf = svm.SVC(kernel='rbf', gamma=gamma, C=c)
clf.fit(X_train, y_train)
Z = clf.predict(X_test)
acc = ((Z.shape[0] - (Z == y_test).sum()) * 100 / Z.shape[0])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('C=%2.2E Gamma=%2.2E  A=%.2f %%' % (c, gamma, acc))
fig.show()

print("end")
