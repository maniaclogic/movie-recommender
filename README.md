# movie-recommender

This is the Principle of a Movie Recommender with unsupervised Machine Learning, namely Non-Negative Matrix Factorization.
There is a second part to this Recommender made with cosine similarity.

If you wish to see the set up of the two check out Models.py.

The Recommender is assembled (including a bit of Data wrangling) in recommender.py.
The function returns two strings of movie titles. One for the recommendations computed with nnmf, one with 'other users also liked'.
