import mae_score
from data_utils import get_dataset_matrix, get_dataset_table

class collaborative_filter:
    def __init__(self, file_num=5, VALIDATION=False) -> None:    
        if VALIDATION:
            self.set_validation()
        else:
            self.set_test(file_num)

    def set_validation(self):
        self.VALIDATION = True

        self.train_path = 'data/validation/validation_train.txt'
        self.test_path = 'data/validation/validation_test.txt'
        self.dst_path = 'data/validation/validation_results.txt'
        self.key_path = 'data/validation/validation_key.txt'

        self.X = get_dataset_matrix(self.train_path)
        self.T = get_dataset_matrix(self.test_path)

    def set_test(self, file_num):
        self.VALIDATION = False

        self.train_path = 'data/train/train.txt'
        self.test_path = f'data/test/test{file_num}.txt'
        self.dst_path = f'data/results/results{file_num}.txt'
        self.key_path = ''

        self.X = get_dataset_matrix(self.train_path)
        self.T = get_dataset_matrix(self.test_path)

    def set_paths(self, train, test, dst, key):
        self.VALIDATION = False

        self.train_path = train
        self.test_path = test
        self.dst_path = dst
        self.key_path = key

        self.X = get_dataset_matrix(self.train_path)
        self.T = get_dataset_matrix(self.test_path)



    def evaluate(self, dst=None, key=None):
        dst = dst or self.dst_path
        key = key or self.key_path

        if self.VALIDATION:
            return mae_score.evaluate(dst, key)
        else:
            raise ValueError('self.VALIDATION = False: Cannot evaluate unless in VALIDATION mode')

    def predict(self, k=1, write=True):
        pass

    def _knn(self, idx, M, m, k=1):
        pass