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


"""        Blockbuster Cateogrization       """


# determine unique categories in database
        # genres = list(df['genres'].unique())
        # import re
        # re.sub("\|", " ", genres)
# write program that determines users favorite genres
        # determine movieIds user liked
        # determine genres of said movieIds
        # sum.rating per genres
        # highest numbers == favorite genre
# make new matrix with userId ; Genres; ratingsum for movies in genre
        # 2 highest sums == 2 favorite user genres
# write program that finds highest rated movies in genres
        # hierachial index movie genre -> movieId -> 5 highest rated in genres

# Use supervised or unsupervised model!! (Because data science)
# find blockbuster or highest rated movies in favorite 2 genres
