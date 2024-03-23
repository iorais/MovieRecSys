from cf import collaborative_filter
import math
import numpy as np

from data_utils import get_dataset_matrix, get_dataset_table, write_results_file

# USER BASED COLLOBORATIVE FILTERING BASE CLASS

class ubcf(collaborative_filter):
    def __init__(self, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

    def _knn(self, idx, M, m, k=1):
        pass

    def predict(self, k=1, write=True):
        test_table = get_dataset_table(self.test_path)

        userIDmin = min(test_table[:,0])

        outputs = []
        row: np.ndarray
        for row in test_table:
            userID, movieID, rating = row.astype(int)
            idx = userID - userIDmin

            if rating == 0:
            # writes the missing user rating for movie
            # if no prediction was made rating is set to 3
                rating = self._knn(idx, self.T, movieID - 1, k) or 3

                output = (userID, movieID, rating)
                outputs.append(output)

        if write:
            write_results_file(outputs, self.dst_path)

        return np.array(outputs)

#
# COSINE SIMILARITY
#

class cosine_similarity(ubcf):
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
        # vector for user of interest
        a = M[idx]

        # mask of users who have watched movie m from training set
        mask = self.X[:,m].astype(int) != 0

        # users who have watched movie m from training set
        U = self.X[mask]
        idxs = np.arange(self.X.shape[0])[mask]

        # similarity scores between users in U and a
        scores = np.array([self._cosine_sim(a, u) for u in U])

        # descending index order of U based on similarity score
        order = scores.argsort()[::-1]

        # reodered U based on descending similarity scores
        U = U[order]
        scores = scores[order]
        idxs = idxs[order]

        # restrict k to the amount of users in U
        k = min(k, U.shape[0])

        # predicted rating of movie m
        prediction = self._cosine_pred(U[:k,m], scores[:k])

        # returns prediction
        return prediction

#
# PEARSON CORRELATION
#

class pearson_correlation(ubcf):
    def __init__(self, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

    def _pearson_calc(self, a: np.ndarray, u: np.ndarray, m):
        # average rating of user a
        mask = a != 0
        a_mean = a[mask].mean()

        # average rating of user u
        mask = u != 0
        u_mean = u[mask].mean()

        # normalizes vectors
        a_norm = a - a_mean
        u_norm = u - u_mean

        # noramlized rating of movie m by user u
        r_norm = u_norm[m]

        # mask of nonzero elements for both a and u
        mask = (a_norm * u_norm) != 0

        # applies mask
        a_norm = a_norm[mask]
        u_norm = u_norm[mask]

        n = a_norm.shape[0]

        if n == 0:
            # return zero if vector length is zero
            return [0, 0]

        # cosine similarity function for weight
        cosine = lambda x, y : (np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

        return [cosine(a_norm, u_norm), r_norm]
    
    def _pearson_pred(self, ratings: np.ndarray, weights: np.ndarray, a: np.ndarray):
        # average rating of user a
        mask = a != 0
        a_mean = a[mask].mean()

        prediction = a_mean

        if weights.shape[0] != 0:
            prediction += np.dot(ratings, weights) / np.linalg.norm(weights, ord=1)
        
        threshhold = prediction - math.floor(prediction) >= 0.5

        # sets prediction to closest integer
        prediction = math.ceil(prediction) if threshhold else math.floor(prediction)

        # lower bound of 1
        if prediction < 1:
            prediction = 1

        # upper bound of 5
        if prediction > 5:
            prediction = 5

        return prediction

    def _knn(self, idx, M, m, k=1):
        # vector for user of interest
        a = M[idx]

        # mask of users who have watched movie m from training set
        mask = self.X[:,m].astype(int) != 0

        # users who have watched movie m from training set
        U = self.X[mask]
        
        # weights between users u in U and a and u_mean
        calc = np.array([self._pearson_calc(a, u, m) for u in U])

        if calc.shape[0] != 0:
            weights = calc[:,0]
            ratings = calc[:,1]
        else:
            weights = np.array([1])
            ratings = np.array([0])

        # similarity scores between users u in U
        scores: np.ndarray
        scores = np.abs(weights)

        # descending index order of U based on similarity score
        order = scores.argsort()[::-1]

        # reodered ratings and weights based on descending similarity scores
        ratings = ratings[order]
        weights = weights[order]

        # restrict k to the amount of users in U
        k = min(k, len(ratings))

        # predicted rating of movie m
        prediction = self._pearson_pred(ratings[:k], weights[:k], a)

        # returns prediction
        return prediction
        
class pearson_correlation_IUF(pearson_correlation):
    def __init__(self, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

        self.get_inverse_user_freq()

    def get_inverse_user_freq(self):
        # vector of inverse user frequencies
        self.IUF = np.zeros((self.X.shape[1]))
        
        # total number of users
        N = self.X.shape[0]

        for i, movie in enumerate(self.X.transpose()):
            # mask to filter users who have not rated movie vecotr
            mask = movie != 0

            # number of users who have rated movie
            n = len(movie[mask])

            self.IUF[i] = math.log2(N/n) if n > 0 else 0

    def _pearson_calc(self, a: np.ndarray, u: np.ndarray, m):
        # average rating of user u
        mask = u != 0
        u_mean = u[mask].mean()

        # noramlized rating of movie m by user u
        r_norm = u[m] - u_mean

        # adjust ratings with IUF
        a = a * self.IUF
        u = u * self.IUF

        # average rating of user a
        mask = a != 0
        a_mean = a[mask].mean()

        # average rating of user u
        mask = u != 0
        u_mean = u[mask].mean()

        # normalizes vectors
        a_norm = a - a_mean
        u_norm = u - u_mean

        # mask of nonzero elements for both a and u
        mask = (a_norm * u_norm) != 0

        # applies mask
        a_norm = a_norm[mask]
        u_norm = u_norm[mask]

        n = a_norm.shape[0]

        if n == 0:
            # return zero if vector length is zero
            return [0, 0]

        # cosine similarity function for weight
        cosine = lambda x, y : (np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

        return [cosine(a_norm, u_norm), r_norm]
    
class pearson_correlation_casemod(pearson_correlation):
    def __init__(self, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

        self.p = 2.5

    def set_p(self, p):
        self.p = p

        if p < 1:
            raise ValueError('p must be greater than or equal to 1')

    def _pearson_calc(self, a: np.ndarray, u: np.ndarray, m):
        # average rating of user a
        mask = a != 0
        a_mean = a[mask].mean()

        # average rating of user u
        mask = u != 0
        u_mean = u[mask].mean()

        # normalizes vectors
        a_norm = a - a_mean
        u_norm = u - u_mean

        # noramlized rating of movie m by user u
        r_norm = u_norm[m]

        # mask of nonzero elements for both a and u
        mask = (a_norm * u_norm) != 0

        # applies mask
        a_norm = a_norm[mask]
        u_norm = u_norm[mask]

        n = a_norm.shape[0]

        if n == 0:
            # return zero if vector length is zero
            return [0, 0]

        # cosine similarity function for weight
        cosine = lambda x, y : (np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

        # original weight
        w = cosine(a_norm, u_norm)

        # calculate case modification term
        casemod = abs(w) ** (self.p - 1)

        return [w * casemod, r_norm]

    
#
#   ENSEMBLE
#
    
class ensemble(ubcf):
    def __init__(self, *model_classes, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

        self.models = [model_class(file_num, VALIDATION) for model_class in model_classes]
        self.weights = np.random.rand(len(self.models))

        if len(model_classes) == 0:
            raise ValueError('there are no models in the ensemble')
        
    def set_validation(self):
        self.VALIDATION = True

        # trains the model inputs for the ensemble
        self.train1_path = 'data/ensemble/ensemble_train1.txt'

        # trains the weights for the ensemble
        self.train2_path = 'data/ensemble/ensemble_train2.txt'
        self.train2key_path = 'data/ensemble/ensemble_train2key.txt'

        # tests the performance of the ensemble
        self.test_path = 'data/ensemble/ensemble_test.txt'
        self.dst_path = 'data/ensemble/ensemble_results.txt'
        self.key_path = 'data/ensemble/ensemble_testkey.txt'

        self.X1 = get_dataset_matrix(self.train1_path)
        self.X2 = get_dataset_matrix(self.train2_path)

        self.T = get_dataset_matrix(self.test_path)
    
    #
    # hyperparamter modification functions
    #

    def set_k(self, *k_vals):
        self.k_vals = k_vals

        if len(self.k_vals) != len(self.models):
            raise ValueError('self.k_vals is of different size than self.models')

    def save_weights(self, filename: str):
        np.save(self.weights, f'ensemble/weights/{filename}.npy')

    def load_weights(self, filename: str):
        self.weights = np.load(f'ensemble/weights/{filename}.npy')

        if len(self.weights) != len(self.models):
            raise ValueError('self.weights is of different size than self.models')

    def get_weights(self) -> np.ndarray:
        return self.weights

    def set_weights(self, *weights):
        self.weights = np.array(weights)

        if len(self.weights) != len(self.models):
            raise ValueError('self.weights is of different size than self.models')

    #    
    # prediction function
    #
        
    def predict(self, k=None, write=True):
        if not self.VALIDATION:
            raise ValueError('self.VALIDATION = False: Cannot train unless in VALIDATION mode')

        # inputs
        X = []

        # gets the outputs of the individual models
        model: collaborative_filter
        for model, k in zip(self.models, self.k_vals):
            model.set_paths(
                self.train1_path,
                self.test_path,
                None,
                None
            )

            outputs = model.predict(k, write=False)
            outputs = outputs[:,2]
            X.append(outputs)

        X = np.array(X)

        rating_outputs = np.dot(self.weights, X)
        rating_outputs = np.round(rating_outputs)
        rating_outputs = rating_outputs.astype(int)

        test_table = get_dataset_table(self.test_path)

        userIDmin = min(test_table[:,0])

        # gets the corresponding userID and movieID for predicted outputs
        outputs = []
        i = 0
        row: np.ndarray
        for row in test_table:
            userID, movieID, rating = row.astype(int)
            

            if rating == 0:
                rating = rating_outputs[i]
                output = (userID, movieID, rating)
                outputs.append(output)
                i += 1


        if write:
            write_results_file(outputs, self.dst_path)

        return np.array(outputs)

    #
    # training function
    #
       
    def train(self):
        if not self.VALIDATION:
            raise ValueError('self.VALIDATION = False: Cannot train unless in VALIDATION mode')

        # inputs
        X = []

        # labels
        Y = []

        groundtruth = get_dataset_table(self.train2key_path)
        groundtruth = groundtruth[:,2]
        Y.append(groundtruth)

        model: collaborative_filter
        for model, k in zip(self.models, self.k_vals):
            model.set_paths(
                self.train1_path,
                self.train2_path,
                None,
                self.train2key_path
            )

            outputs = model.predict(k, write=False)
            outputs = outputs[:,2]
            X.append(outputs)

        X = np.array(X)
        Y = np.array(Y)

        # least square fit closed form solution
        XTX = np.matmul(X, X.transpose())
        XTXinv = np.linalg.inv(XTX)

        XTY = np.matmul(X, Y.transpose())

        w = np.dot(XTXinv, XTY)

        self.weights = w.transpose()[0]