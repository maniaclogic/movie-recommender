import pandas as pd
import numpy as np
from models import R, nmf, DF

user_input_dict = # there will have to be a dictionary with
                    # movieId as key; rating as value


def recommend(dic):

    movie_Ids = sorted(list(R.columns))
    empty_list = [np.nan]*len(movie_Ids)
    ratings_dict = dict(zip(movie_Ids, empty_list))

    for key, value in dic.items():
        ratings_dict[key] = value


    reshape_user_dict = list(ratings_dict.values())
    profile = pd.DataFrame(reshape_user_dict, index = movie_Ids)
    profile = profile.transpose()
    profile_filled = profile.fillna(3.0) #?
    hidden_profile = nmf.transform(profile_filled)


    prediction = np.dot(hidden_profile, nmf.components_)
    coeff_profile = pd.DataFrame(prediction, columns = R.columns)

    movies_not_watched = coeff_profile.transpose()[0][np.isnan(profile_floats[0])]

    step = pd.DataFrame(movies_not_watched)
    list_movieIds = list(step[0].sort_values(ascending=False).index[:5])
    recommendation = [DF['title'].values[each] for each in list_movieIds]

    return recommendation
