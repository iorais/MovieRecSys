import numpy as np
import pandas as pd
from collections import defaultdict

from data_utils import get_dataset_table

def get_dataframe(path: str):
    users = []
    movies = []
    ratings = []

    with open(path) as trainingset:
        for line in trainingset:
            userID, movieID, rating = np.array(line.split()).astype(int)

            users.append(userID)
            movies.append(movieID)
            ratings.append(rating)
    
    data = {
        'userID': users,
        'movieID': movies,
        'rating': ratings
    }

    df = pd.DataFrame(data)

    return df

def evaluate(results_file: str, groundtruth_file: str):
    res_df = get_dataframe(results_file)
    res_r = res_df['rating']
    
    gt_df = get_dataframe(groundtruth_file)
    gt_r = gt_df['rating']

    pred = np.array(res_r)
    truth = np.array(gt_r)

    print(pred.shape)
    print(truth.shape)

    n = len(pred)

    MAE = sum(abs(pred - truth)) / n

    return MAE

def eval_stats(results_file: str, groundtruth_file: str):
    res_T = get_dataset_table(results_file)
    gt_T = get_dataset_table(groundtruth_file)

    n = len(res_T)

    # [count, total error]
    missed_users = defaultdict(lambda : [0, 0, 0])
    missed_movies = defaultdict(lambda : [0, 0, 0])
 
    for i in range(n):
        user, movie, rating = res_T[i]
        groundtruth = gt_T[i][2]

        error = groundtruth - rating

        if error != 0:
            missed_users[user][0] += 1
            missed_users[user][1] += error
            missed_users[user][2] += abs(error)

            missed_movies[movie][0] += 1
            missed_movies[movie][1] += error
            missed_movies[movie][2] += abs(error)

    udata = {
        'userID': [user for user in missed_users.keys()],
        'count': [missed_users[user][0] for user in missed_users.keys()],
        'error': [missed_users[user][1] for user in missed_users.keys()],
        'mae': [missed_users[user][2] for user in missed_users.keys()]
    }

    mdata = {
        'movieID': [movie for movie in missed_movies.keys()],
        'count': [missed_movies[movie][0] for movie in missed_movies.keys()],
        'error': [missed_movies[movie][1] for movie in missed_movies.keys()],
        'mae': [missed_movies[movie][2] for movie in missed_movies.keys()]
    }
    
    user_df = pd.DataFrame(udata)
    movie_df = pd.DataFrame(mdata)

    return user_df, movie_df
