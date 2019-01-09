import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import PACS_loading as PACS



def show_scatterplot(axs, firstPC, secondPC, X, y, data, color, str_x, str_y):
    for i in range(0, 4):
        axs.scatter(X[y == i, firstPC], X[y == i, secondPC], c=color[i], label=data[i], s=10)
    axs.set_xlabel(str_x)
    axs.set_ylabel(str_y)
    axs.legend()
    axs.grid(True)


def pca_custom(X):
    """
    Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance
    and mean.
    """
    # get dimensions
    num_data, dim = X.shape
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    # PCA - compact trick used
    M = np.dot(X, X.T)  # covariance matrix, AA', not the A'A like usual
    e, EV = np.linalg.eigh(M)  # compute eigenvalues and eigenvectors
    tmp = np.dot(X.T, EV).T  # this is the compact trick
    V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
    S = np.sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
    for i in range(V.shape[1]):
        V[:, i] /= S  # What for?

    # return the projection matrix, the variance and the mean
    return V, S, mean_X


def reconstruction(X_t, st, nComp):
    if nComp > 0:
        X_b = np.dot(X_t[:, :nComp], pca.components_[:nComp, :])
    else:
        X_b = np.dot(X_t[:, nComp:], pca.components_[nComp:, :])

    orig = st.inverse_transform(X_b) / 255
    return orig.astype(np.float64)


# 1.1 DATA PREPARATION
if len(sys.argv) != 3 and len(sys.argv) != 4:
    print("Usage:")
    print("python program.py -d <PACS_homework folder>")
    print("- Loads Image from the specified Folder and save files in data.npy and label.npy")
    print("python program.py -l <data.npy file> <label.npy file>")
    print("- Loads Image from data.npy and label.npy files")
    exit(-1)
if sys.argv[1] == '-d' and len(sys.argv) == 3:
    X, y = PACS.gen_matrix(sys.argv[2])
elif sys.argv[1] == '-l' and len(sys.argv) == 4:
    X, y = PACS.load_matrix(sys.argv[2], sys.argv[3])
else:
    print("Usage:")
    print("- python program.py -d <PACS_homework folder>")
    print("-- Loads Image from the specified Folder and save files in data.npy and label.npy")
    print("- python program.py -l <data.npy file> <label.npy file>")
    print("-- Loads Image from data.npy and label.npy files")

    exit(-1)

legend_data = ["Dog", "Guitar", "House", "Person"]
legend_color = ["red", "blue", "green", "orange"]

# 1.2 PRINCIPAL COMPONENT VISUALIZATION

# Standardize the features
st = StandardScaler()
st.fit(X)
X_n = st.transform(X)
pca = PCA()
pca.fit(X_n)

# TODO: Remove
# V,S,mean = pca_custom(X_n)
# cov_mat = np.cov(X_n)
# eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Create the figure and set id of the image
fig = plt.figure()
image_id = 0

X_t = pca.transform(X_n)

# Plot Original Image
ax = fig.add_subplot(1, 5, 1)
orig = X / 255
# orig = orig.astype(np.float64)
ax.imshow(orig[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
ax.set_title('Original Image', fontsize=16)

# Plot 60PC Image
ax = fig.add_subplot(1, 5, 2)
X_60pc = reconstruction(X_t, st, 500)
ax.imshow(X_60pc[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
ax.set_title('60PC var: %d%%' % (np.sum(pca.explained_variance_ratio_[:500]) * 100), fontsize=16)

# Plot 6PC Image
ax = fig.add_subplot(1, 5, 3)
X_6pc = reconstruction(X_t, st, 6)
ax.imshow(X_6pc[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
ax.set_title('6PCvar: %d%%' % (np.sum(pca.explained_variance_ratio_[:6]) * 100), fontsize=16)

# Plot 2PC Image
ax = fig.add_subplot(1, 5, 4)
X_2pc = reconstruction(X_t, st, 2)
ax.imshow(X_2pc[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
ax.set_title('2PCvar: %d%%' % (np.sum(pca.explained_variance_ratio_[:2]) * 100), fontsize=16)

# Plot Last 6PC Image
ax = fig.add_subplot(1, 5, 5)
X_l6pc = reconstruction(X_t, st, -6)
ax.imshow(X_l6pc[image_id].reshape(227, 227, 3), vmin=0, vmax=1)
ax.set_title('Last 6PC var: %d%%' % (np.sum(pca.explained_variance_ratio_[-6:]) * 100), fontsize=16)
fig.show()

# Print Scatterplots

fig = plt.figure()

fig.suptitle("Projection on different PC", fontsize=16)
ax = fig.add_subplot(2, 2, 1)
show_scatterplot(ax, 0, 1, X_t, y, legend_data, legend_color, '1° PC', '2°PC')
ax = fig.add_subplot(2, 2, 2)
show_scatterplot(ax, 2, 3, X_t, y, legend_data, legend_color, '3° PC', '4°PC')
ax = fig.add_subplot(2, 2, 3)
show_scatterplot(ax, 9, 11, X_t, y, legend_data, legend_color, '10° PC', '11°PC')

ax = fig.add_subplot(2, 2, 4, projection='3d')
for i in range(0, 4):
    ax.scatter(X_t[y == i, 0], X_t[y == i, 1], X_t[y == i, 2], c=legend_color[i], label=legend_data[i], s=10)
ax.set_xlabel('1° PC')
ax.set_ylabel('2° PC')
ax.set_zlabel('3° PC')
ax.legend()
ax.grid(True)

fig.show()

# Plot Explained Variance
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

ind = np.arange(len(var_exp))  # the x locations for the groups
width = 0.5  # the width of the bars
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(ind - width / 2, var_exp, width, color='SkyBlue', label='Explained Variance')
ax.scatter(ind + width / 2, cum_var_exp, width, color='IndianRed', label='Cumulative Explained Variance')
ax.legend()
ax.grid(True)
fig.show()

x_train, x_test, y_train, y_test = train_test_split(X, y)
clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Original accuracy over (%d points): %d%%" % (
x_test.shape[0], ((y_test.shape[0] - (y_test != y_pred).sum()) * 100 / y_test.shape[0])))

x_train, x_test, y_train, y_test = train_test_split(X_t[:, 0:2], y)
clf = GaussianNB()
eclf = VotingClassifier(estimators=[('GaussianNB', clf)],
                        voting='soft')
clf.fit(x_train, y_train)
eclf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))

fig = plt.figure()
axarr = fig.add_subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
cmap_plot = matplotlib.colors.ListedColormap(legend_color)

axarr.contourf(xx, yy, Z, alpha=0.5, norm=norm, cmap=cmap_plot)
for i in range(0, 4):
    axarr.scatter(X_t[y == i, 0], X_t[y == i, 1], c=legend_color[i], label=legend_data[i], s=20)
axarr.set_title("1° and 2° PC accuracy over (%d points): %d%%" % (
x_test.shape[0], ((y_test.shape[0] - (y_test != y_pred).sum()) * 100 / y_test.shape[0])))
axarr.legend()
fig.show()

print("1° and 2° PC accuracy over (%d points): %d%%" % (
x_test.shape[0], ((y_test.shape[0] - (y_test != y_pred).sum()) * 100 / y_test.shape[0])))

x_train, x_test, y_train, y_test = train_test_split(X_t[:, 2:4], y)
clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
eclf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
x_min, x_max = X_t[:, 2].min() - 1, X_t[:, 2].max() + 1
y_min, y_max = X_t[:, 3].min() - 1, X_t[:, 3].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))

fig = plt.figure()
axarr = fig.add_subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr.contourf(xx, yy, Z, alpha=0.5, norm=norm, cmap=cmap_plot)
for i in range(0, 4):
    axarr.scatter(X_t[y == i, 2], X_t[y == i, 3], c=legend_color[i], label=legend_data[i], s=20)
axarr.set_title("3° and 4° PC accuracy over (%d points): %d%%" % (
x_test.shape[0], ((y_test.shape[0] - (y_test != y_pred).sum()) * 100 / y_test.shape[0])))
axarr.legend()
fig.show()

plt.show()

print("end")
