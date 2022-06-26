numerical_features = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev', 'userRatingCount',
                      'userAvgRating', 'userRatingStddev']

categorical_features = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5', 'movieGenre1',
                        'movieGenre2', 'movieGenre3']

embedding_features = ["userId", "movieId"]

cross_features = [("movieId", "userRatedMovie1")]