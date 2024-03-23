from cf import collaborative_filter
import math
import numpy as np

import mae_score
from data_utils import get_dataset_matrix, get_dataset_table, write_results_file

# ITEM BASED COLLOBORATIVE FILTERING BASE CLASS

class ibcf(collaborative_filter):
    def __init__(self, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

        self.X: np.ndarray
        self.T: np.ndarray

        # matrix with rows of movies and columns of users from training set
        self.X = self.X.transpose()

        # matrix with rows of movies and columns of users from test set
        self.T = self.T.transpose()

        # matrix with rows of movies and columns of users from training and test sets
        self.XT = np.hstack((self.X, self.T))

    def set_validation(self):
        super().set_validation()

        # matrix with rows of movies and columns of users from training set
        self.X = self.X.transpose()

        # matrix with rows of movies and columns of users from test set
        self.T = self.T.transpose()

        # matrix with rows of movies and columns of users from training and test sets
        self.XT = np.hstack((self.X, self.T))

    def set_test(self, file_num):
        super().set_test(file_num)

        # matrix with rows of movies and columns of users from training set
        self.X = self.X.transpose()

        # matrix with rows of movies and columns of users from test set
        self.T = self.T.transpose()

        # matrix with rows of movies and columns of users from training and test sets
        self.XT = np.hstack((self.X, self.T))
    
    def set_paths(self, train, test, dst, key):
        super().set_paths(train, test, dst, key)
        
        # matrix with rows of movies and columns of users from training set
        self.X = self.X.transpose()

        # matrix with rows of movies and columns of users from test set
        self.T = self.T.transpose()

        # matrix with rows of movies and columns of users from training and test sets

    def _knn(self, idx, M, m, k=1):
        pass

    def predict(self, k=1, write=True):
        test_table = get_dataset_table(self.test_path)

        userIDmin = min(test_table[:,0])
        N = self.X.shape[1]

        outputs = []
        row: np.ndarray
        for row in test_table:
            userID, movieID, rating = row.astype(int)
            idx = userID - userIDmin + N

            if rating == 0:
            # writes the missing user rating for movie
            # if no prediction was made rating is set to 3
                rating = self._knn(idx, self.XT, movieID - 1, k) or 3

                output = (userID, movieID, rating)
                outputs.append(output)

        if write:
            write_results_file(outputs, self.dst_path)

        return np.array(outputs)

#
# COSINE SIMILARITY
#

class cosine_similarity(ibcf):
    def __init__(self, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

    def _cosine_sim(self, a: np.ndarray, u: np.ndarray):
        # mask of nonzero elements
        mask = (a * u) != 0

        # applies mask
        a = a[mask]
        u = u[mask]

        if a.shape[0] == 0:
            return 0

        # cosine similarity function
        cosine = lambda x, y : (np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))
        
        return cosine(a, u)

    def _cosine_pred(self, ratings: np.ndarray, scores: np.ndarray):
        weighted_avg = lambda x, w : np.dot(x, w) / sum(w) if sum(w) > 0 else 0

        prediction = weighted_avg(ratings, scores)

        threshhold = prediction - math.floor(prediction) >= 0.5

        return math.ceil(prediction) if threshhold else math.floor(prediction)

    def _knn(self, idx, M, m, k=1):
        # vector for movie of interest
        a = M[m]

        # mask of movies that have been watched by the user at idx
        mask = self.XT[:,idx].astype(int) != 0

        # movies that have been watched by the user at idx
        U = self.XT[mask]

        # similarity scores between movies in U and a
        scores = np.array([self._cosine_sim(a, u) for u in U])

        # descending index order of U based on similarity score
        order = scores.argsort()[::-1]

        # reodered U based on descending similarity scores
        U = U[order]
        scores = scores[order]

        # restrict k to the amount of users in U
        k = min(k, U.shape[0])

        # predicted rating of movie m
        prediction = self._cosine_pred(U[:k,idx], scores[:k])

        # returns prediction
        return prediction