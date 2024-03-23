import numpy as np
import pandas as pd
from collections import defaultdict

def get_dataset_df(path: str) -> pd.DataFrame:
    '''
    Returns an adjacency matrix of the dataset of size n x m
    - n - number of users
    - m - number of movies\n
    Accepts data where the input file is formatted by\n 
    '<userID> <movieID> <rating>' for each row
    ## Parameters:
    1. path : str
    - The path to the dataset.
    ## Returns:\n
    out : np.ndarray
    '''
    T = get_dataset_table(path)

    users = []
    movies = []
    ratings = []

    line: np.ndarray
    for line in T:
        userID, movieID, rating = line.astype(int)

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

def get_dataset_matrix(path: str) -> np.ndarray:
    '''
    Returns an adjacency matrix of the dataset of size n x m
    - n - number of users
    - m - number of movies\n
    Accepts data where the input file is formatted by\n 
    '<userID> <movieID> <rating>' for each row
    ## Parameters:
    1. path : str
    - The path to the dataset.
    ## Returns:\n
    out : np.ndarray
    '''
    T = get_dataset_table(path)
    
    userIDmax = max(T[:,0])
    userIDmin = min(T[:,0])
    user_count = userIDmax - userIDmin + 1

    # initializes matrix of testset
    M = np.zeros((user_count, 1000))

    line: np.ndarray
    for line in T:
        userID, movieID, rating = line.astype(int)
        
        M[userID - userIDmin][movieID - 1] = rating
    
    return M

def get_dataset_table(path: str):
    '''
    Returns a table of the dataset as integer elements\n
    The table format is as follows:\n
    - dataset_table[i] - ith row
    - dataset_table[i][0] - userID of the ith row
    - dataset_table[i][1] - movieID of the ith row
    - dataset_table[i][2] - rating of the ith row\n
    Accepts data where the input file is formatted by\n 
    '<userID> <movieID> <rating>' for each row
    ## Parameters:
    1. path : str
    - The path to the dataset\n
    ## Returns:\n
    out : np.ndarray
    '''
    with open(path) as dataset:
        T = np.array([[int(elm) for elm in row.split()] for row in dataset])

    return T

def write_results_file(outputs: np.ndarray, path: str):
    '''
    Writes outputs into a txt file 'path' formatted by\n 
    '<userID> <movieID> <rating>' for each row
    ## Parameters:
    1. outputs: np.ndarray
    - A data table to be written in 'path' formatted by:\n
    dataset_table[i] - ith row\n
    dataset_table[i][0] - userID of the ith row\n
    dataset_table[i][1] - movieID of the ith row\n
    dataset_table[i][2] - rating of the ith row\n
    2. path : str
    - The path to the dataset\n
    ## Returns:\n
    None
    '''
    outputs.sort()

    with open(path, 'w') as output_file:
        output: str
        for output in outputs:
            u, m, r = output
            output_line = f'{u} {m} {r}'
            output_file.write(output_line + '\n')

def create_validationset(ratio=0.8):
    df = get_dataset_df('data/train/train.txt')
    
    total_user_count = 200
    training_user_count = total_user_count * ratio

    train_df = df.loc[df['userID'] <= training_user_count]
    val_df = df.loc[df['userID'] > training_user_count]

    # writes the training text file split from validation
    with open('data/validation/validation_train.txt', 'w') as file:
        for i, row in train_df.iterrows():
            u, m, r = row
            
            file.write(f'{u} {m} {r}\n')

    users = defaultdict(lambda : 0)
    # writes the validation text file to use for testing
    with open('data/validation/validation_test.txt', 'w') as file:
        for i, row in val_df.iterrows():
            u, m, r = row
            if users[u] >= 5:
                r = 0

            file.write(f'{u} {m} {r}\n')
            users[u] += 1

    users = defaultdict(lambda : 0)
    # writes the validation text file with answers
    with open('data/validation/validation_key.txt', 'w') as file:
        for i, row in val_df.iterrows():
            u, m, r = row
            if users[u] >= 5:
                file.write(f'{u} {m} {r}\n')

            users[u] += 1

def create_ensembleset(ratio=0.8, training_ratio=0.625):
    df = get_dataset_df('data/train/train.txt')

    total_user_count = 200
    training_user_count = total_user_count * ratio
    training1_user_count = training_user_count * training_ratio

    train_df = df.loc[df['userID'] <= training_user_count]

    # splits training data into two
    train1_df = train_df.loc[train_df['userID'] <= training1_user_count]
    train2_df = train_df.loc[train_df['userID'] > training1_user_count]

    val_df = df.loc[df['userID'] > training_user_count]

    # writes the training1 text file
    # this data will be used to train individual algorithms
    with open('data/ensemble/ensemble_train1.txt', 'w') as file:
        for i, row in train1_df.iterrows():
            u, m, r = row
            
            file.write(f'{u} {m} {r}\n')

    users = defaultdict(lambda : 0)
    # writes the training2 text file
    # this data will be the input for ensemble training
    with open('data/ensemble/ensemble_train2.txt', 'w') as file:
        for i, row in train2_df.iterrows():
            u, m, r = row
            if users[u] >= 5:
                r = 0

            file.write(f'{u} {m} {r}\n')
            users[u] += 1

    users = defaultdict(lambda : 0)
    # writes the training2 text file with answers
    # this data will be the labels for ensemble training
    with open('data/ensemble/ensemble_train2key.txt', 'w') as file:
        for i, row in train2_df.iterrows():
            u, m, r = row
            if users[u] >= 5:
                file.write(f'{u} {m} {r}\n')

            users[u] += 1

    users = defaultdict(lambda : 0)
    # writes the validation text file
    # this data will be used to tune the ensemble model
    with open('data/ensemble/ensemble_test.txt', 'w') as file:
        for i, row in val_df.iterrows():
            u, m, r = row
            if users[u] >= 5:
                r = 0

            file.write(f'{u} {m} {r}\n')
            users[u] += 1

    users = defaultdict(lambda : 0)
    # writes the validation text file with answers
    # this data will be used to tune the ensemble model
    with open('data/ensemble/ensemble_testkey.txt', 'w') as file:
        for i, row in val_df.iterrows():
            u, m, r = row
            if users[u] >= 5:
                file.write(f'{u} {m} {r}\n')

            users[u] += 1