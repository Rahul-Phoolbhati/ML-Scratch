# implementing Knn from scrach
import math
from collections import Counter

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.x_train = X;
        self.y_train = Y;

    def _euclidean_distance(self, x1, x2):
        total = 0
        for a,b in zip(x1,x2):
            total += (a-b)**2
        return math.sqrt(total)

    def _ind_distances(self, x):
        distances = [(ind, self._euclidean_distance(x, xi)) for ind,xi in enumerate(self.x_train)]
        return distances
        # distances = [euclidean_distance(x, xi) for xi in x_train]
        # return list(enumerate(distances))

    def _predict(self, x):
        ind_distances = self._ind_distances(x)
        ind_distances.sort(key=lambda x: x[1])        # can use Sorted also but slightly lesser perf, bcs of creating copy 
        k_nearest = ind_distances[:self.k]
        labels = [self.y_train[ind] for ind,_ in k_nearest]
        prediction = Counter(labels).most_common(1)[0][0]
        return prediction 


    def predict(self, X):
        prediction = [self._predict(x) for x in X]
        return prediction


