import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import utils

# https://github.com/mdeff/fma/issues/9
# look at comment to get branch with scripts that work with dataset if wanted

# Load metadata and features.
tracks = utils.load('fma_metadata/tracks.csv')
# genres = utils.load('fma_metadata/genres.csv')
features = utils.load('fma_metadata/features.csv')
# echonest = utils.load('fma_metadata/echonest.csv')

# np.testing.assert_array_equal(features.index, tracks.index)
# assert echonest.index.isin(tracks.index).all()

medium = tracks['set', 'subset'] <= 'medium'
print(medium.shape)

Y = tracks.loc[medium, ('track', 'genre_top')]
X = features.loc[medium]
print(X.shape)

print('{} features, {} classes'.format(X.shape[1], np.unique(Y).size))

X.to_csv('data/dataSetMedium.csv')
Y.to_csv('data/labelsMedium.csv')
