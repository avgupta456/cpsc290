from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
import math

from helper import utils
import constants

fold = 0
data_path = constants.processed_path+"/fold"+str(fold)
train, test, val = utils.load_data(data_path)
X_train, Y_train, times_train = train

num_people = constants.max_people
features = constants.features
features = features[0] + 2 * features[1] + features[2]


X_groups = np.reshape(X_train[0], (X_train[0].shape[0], (num_people-2)*features))
X_pairs = np.reshape(X_train[1], (X_train[1].shape[0], 2*features))
X = np.concatenate((X_groups, X_pairs), 1)

X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#pca = PCA(n_components=5)
pca = PCA(0.95)
X_pca = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
print(X_pca.shape)

combined = np.concatenate((X_pca, Y_train), 1)

fig, ax = plt.subplots()
colors = ['blue', 'orange']

for i in range(1000):
    color = colors[int(combined[i][-1])]
    ax.scatter(combined[i][0], combined[i][1], color=color)

plt.show()
