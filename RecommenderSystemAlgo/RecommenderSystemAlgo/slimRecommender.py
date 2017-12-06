LR
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet

class Recommender(object):
    """Abstract Recommender"""

    def __init__(self):
        super(Recommender, self).__init__()
        self.dataset = None

    def _get_user_ratings(self, user_id):
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        return self.dataset[:, item_id]

    def fit(self, X):
        pass

    def recommend(self, user_id, n=None, exclude_seen=True):
        pass

    def _filter_seen(self, user_id, ranking):
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

class SLIM_Recommender(Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
    """

    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True):
        super(SLIM_Recommender, self).__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)

    def fit(self, X):
        self.dataset = X
        n_items = X.shape[1]
        X = X.tocsc()
        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False)

        #X = X.tocsr()
        # we'll store the W matrix into a sparse csr_matrix
        # let's initialize the vectors used by the sparse.csc_matrix constructor
        values, rows, cols = [], [], []
        print("end ElasticNet")
        # fit each item's factors sequentially (not in parallel)
        for j in range(n_items):
            # get the target column
            y = X[:, j].toarray()
            # set the j-th column of X to zero
            startptr = X.indptr[j]
            endptr = X.indptr[j + 1]
            bak = X.data[startptr: endptr].copy()
            X.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            self.model.fit(X, y)
            print("test")
            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nnz_idx = self.model.coef_ > 0.0
            values.extend(self.model.coef_[nnz_idx])
            rows.extend(np.arange(n_items)[nnz_idx])
            cols.extend(np.ones(nnz_idx.sum()) * j)

            # finally, replace the original values of the j-th column
            X.data[startptr:endptr] = bak
        print("end factor")
        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

from multiprocessing import Pool
from functools import partial


class MultiThreadSLIM(SLIM_Recommender):
    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True,
                 workers=4):
        super(MultiThreadSLIM, self).__init__(l1_penalty=l1_penalty,
                                              l2_penalty=l2_penalty,
                                              positive_only=positive_only)
        self.workers = workers

    def _partial_fit(self, j, X):
        model = ElasticNet(alpha=1.0,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=False,
                           copy_X=False)
        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, j].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[j]:X_j.indptr[j + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        nnz_idx = model.coef_ > 0.0
        values = model.coef_[nnz_idx]
        rows = np.arange(X.shape[1])[nnz_idx]
        cols = np.ones(nnz_idx.sum()) * j
        return values, rows, cols

    def fit(self, X):
        self.dataset = X
        X = X.tocsc()
        n_items = X.shape[1]
        # fit item's factors in parallel
        _pfit = partial(self._partial_fit, X=X)
        pool = Pool(processes=self.workers)
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            print("test")
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)
        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)