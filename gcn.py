import math
import pickle
import numpy as np

from cf import collaborative_filter
from data_utils import get_dataset_matrix, get_dataset_table, write_results_file


class link_prediction(collaborative_filter):
    def __init__(self, file_num=5, VALIDATION=False) -> None:
        super().__init__(file_num, VALIDATION)

        # link prediction rating matrix
        self.lp_mat = np.load('lp_mat.npy')

        # link prediction movieId to idx dictionary
        with open('lp_dict.pkl', 'rb') as f:
            self.movieID_to_idx = pickle.load(f)

    def _gcn_pred(self, uidx, midx):
        r = self.lp_mat[uidx][midx]
        u = self.lp_mat[uidx]

        if r < -0.6:
            return 1
        if r < -0.2:
            return 2
        if r < 0.2:
            return 3
        if r < 0.6:
            return 4
        
        return 5

    def predict(self, k=1, write=True):
        test_table = get_dataset_table(self.test_path)

        outputs = []
        row: np.ndarray
        for row in test_table:
            userID, movieID, rating = row.astype(int)

            if rating == 0:
            # writes the missing user rating for movie
            # if no prediction was made rating is set to 3
                uidx = userID - 1
                midx = self.movieID_to_idx[movieID]

                rating = self._gcn_pred(uidx, midx)

                output = (userID, movieID, rating)
                outputs.append(output)

        if write:
            write_results_file(outputs, self.dst_path)

        return np.array(outputs)