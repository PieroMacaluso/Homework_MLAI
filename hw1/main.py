from PIL import Image
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from standardize import Standardize
import PACS_loading as PACS


def reconstruction(X_t, image_id, st, nComp):
    if (nComp > 0):
        X_b = np.dot(X_t[:, :nComp], pca.components_[:nComp, :])
    else:
        X_b = np.dot(X_t[:, nComp:], pca.components_[nComp:, :])

    orig = st.destandardize(X_b)/255
    orig.astype(np.float32)
    return orig[image_id].reshape(227, 227, 3)


# Start of the program
if (len(sys.argv) != 3 and len(sys.argv) != 4):
    print("Usage:")
    print("python program.py -d <PACS_homework folder>")
    print("- Loads Image from the specified Folder and save files in data.npy and label.npy")
    print("python program.py -l <data.npy file> <label.npy file>")
    print("- Loads Image from data.npy and label.npy files")
    exit(-1)
if (sys.argv[1] == '-d' and len(sys.argv) == 3):
    X, y = PACS.gen_matrix(sys.argv[2])
elif (sys.argv[1] == '-l' and len(sys.argv) == 4):
    X, y = PACS.load_matrix(sys.argv[2], sys.argv[3])
else:
    print("Usage:")
    print("- python program.py -d <PACS_homework folder>")
    print("-- Loads Image from the specified Folder and save files in data.npy and label.npy")
    print("- python program.py -l <data.npy file> <label.npy file>")
    print("-- Loads Image from data.npy and label.npy files")

    exit(-1)

# Standardize the features
st = Standardize()
X_n = st.standardize(X)

pca = PCA()
pca.fit(X_n)

# Create the figure and set id of the image
plt.figure(0, figsize=(8,4))
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
plt.imshow(reconstruction(X_t, image_id, st, 60))
plt.title('60PC')

# Plot 6PC Image
plt.subplot(1, 5, 3)
plt.imshow(reconstruction(X_t, image_id, st, 6))
plt.title('6PC')

# Plot 2PC Image
plt.subplot(1, 5, 4)
plt.imshow(reconstruction(X_t, image_id, st, 2))
plt.title('2PC')

# Plot Last 6PC Image
plt.subplot(1, 5, 5)
plt.imshow(reconstruction(X_t, image_id, st, -6))
plt.title('Last 6PC')


plt.figure(1)
plt.scatter(X_t[:, 0], X_t[:, 1], c=y.ravel())
plt.show()
print("end")
