import logging
from sklearn.cluster import KMeans
import hdbscan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DocClusterer(object):
    def __init__(self, k, dbs_eps=3, dbs_min_samples=5):
        self.k = k
        self.dbs_eps = dbs_eps
        self.dbs_min_samples = dbs_min_samples

        self.cluster_obj = None
        self.X = None
        self.y = None

    def __call__(self, X, y=None, method='kmeans'):
        task_func = {
            'kmeans': self.kmeans,
            'lda': self.lda,
            'dbscan': self.dbscan
        }
        if method not in task_func.keys():
            raise NotImplementedError
        self.X = X
        self.y = y
        task_func[method]()
        return self.cluster_obj

    def kmeans(self):
        self.cluster_obj = KMeans(n_clusters=self.k)
        self.cluster_obj.fit(self.X)
    
    def lda(self):
        self.k -= 1
        if self.y is None:
            logging.warning('To run lda, you need labels input!')
            raise TypeError
        self.cluster_obj = LinearDiscriminantAnalysis(n_components=self.k)
        self.cluster_obj.fit(self.X, self.y)
        self.cluster_obj.labels_ = self.cluster_obj.predict(self.X)
    
    def dbscan(self):
        self.cluster_obj = hdbscan.HDBSCAN(
            cluster_selection_epsilon=self.dbs_eps, 
            min_cluster_size=self.dbs_min_samples,
            prediction_data=True
            )
        self.cluster_obj.fit(self.X)

        def dbscan_predict(X_test):
            test_labels, strengths = hdbscan.approximate_predict(self.cluster_obj, X_test)
            return test_labels
        self.cluster_obj.predict = dbscan_predict
