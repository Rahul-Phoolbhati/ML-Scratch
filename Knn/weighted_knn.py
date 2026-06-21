# implementing Knn from scrach
import math
import heapq
from collections import Counter


def _uniform_vote(k_nearest, y_train):
    labels = [y_train[ind] for ind,_ in k_nearest]
    prediction = Counter(labels).most_common(1)[0][0]
    return prediction 

def _distance_vote(k_nearest, y_train):
    votes = {}

    for ind, dist in k_nearest:
        weight = 1/(dist+1e-10)  # 1e-10 ? to avoid devision by 0
        label = y_train[ind]
        votes[label] = votes.get(label, 0) + weight
    
    return max(votes, key=votes.get)

WEIGHT_FUNCS = {
    "uniform": _uniform_vote,
    "distance": _distance_vote
}

class KNN:

    def __init__(self, k=3, weights="distance"):
        self.k = k
        self.weight_fn = self._get_weight_fn(weights)

    def _get_weight_fn(self, weights):
        if callable(weights):
            return weights
        
        if isinstance(weights, str):
            if weights not in WEIGHT_FUNCS:
                raise ValueError(
                    f"Unknown weights : {weights}. "
                    f"Choose from {list(WEIGHT_FUNCS.keys())} or pass a callable."
                )
            return WEIGHT_FUNCS[weights]
        else:
            raise TypeError("weights must be 'uniform', 'distance' or a callable.")

    def fit(self, X, Y):
        if self.k > len(X):
            raise ValueError("Value of k can't be greater than the size of training set")
        
        if len(X) != len(Y):
            raise ValueError("X and y must have same number of samples")

        self.x_train = X
        self.y_train = Y

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
        # ind_distances.sort(key=lambda x: x[1])        # can use Sorted also but slightly lesser perf, bcs of creating copy 
        # k_nearest = ind_distances[:self.k]
        
        # heapq for k smallest elements
        k_nearest = heapq.nsmallest(
            self.k,
            ind_distances,
            key = lambda x:x[1]
        )

        return self.weight_fn(k_nearest, y_train)
        


    def predict(self, X):
        prediction = [self._predict(x) for x in X]
        return prediction


