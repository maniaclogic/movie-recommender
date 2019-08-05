import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import NMF


MOVIES = pd.read_csv('movies.csv', header=0)
ratings = pd.read_csv('ratings.csv', header=0)
tags = pd.read_csv('tags.csv', header=0)
links = pd.read_csv('links.csv', header=0)

ratings = ratings[['userId', 'movieId', 'rating']]
tags = tags[['userId', 'movieId', 'tag']]
links = links[['movieId', 'imdbId']]


data_dfs = [movies, tags, links, ratings]
DF = pd.concat(data_dfs, join='outer', sort=True)

def df_to_matrix(df):
    pre_data = DF[['movieId', 'userId', 'rating']]
    data = pre_data.pivot_table(index='userId', columns='movieId', values='rating')
    data.fillna(3.0, inplace=True)
    return data, pre_data

R, PRE = df_to_matrix(DF)



"""     RATING  MODEL    """

nmf = NMF(n_components=100, init='random', solver='cd', max_iter=300)
"""     sparse matrix reconstruction_err_ = 161;
        fillna(3.0) reconstruction_err_ = 60        """

# nmf.fit(R)
# binary_nmf = pickle.dumps(nmf)
# open('nmf_model.bin', 'wb').write(binary_nmf)


binary_nmf = open('nmf_model.bin', 'rb').read()
nmf = pickle.loads(binary_nmf)

# Q = nmf.components_
# P = nmf.transform(a)
#nmf.reconstruction_err_

#reconstruction_matrix = np.dot(P, Q)
