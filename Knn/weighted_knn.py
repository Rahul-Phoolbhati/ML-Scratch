# implementing Knn from scrach
import math
import heapq
from collections import Counter

class KNN:

    def __init__(self, k=3, weights="distance"):
        self.k = k
        self.weights = weights

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

    def _uniform_vote(self, k_nearest):
        labels = [self.y_train[ind] for ind,_ in k_nearest]
        prediction = Counter(labels).most_common(1)[0][0]
        return prediction 

    def _distance_vote(self, k_nearest):
        votes = {}

        for ind, dist in k_nearest:
            weight = 1/(dist+1e-10)  # 1e-10 ? to avoid devision by 0
            label = self.y_train[ind]
            votes[label] = votes.get(label, 0) + weight
        
        return max(votes, key=votes.get)


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

        if self.weights == "uniform":
            return self._uniform_vote(k_nearest)

        elif self.weights == "distance":
            return self._distance_vote(k_nearest)

        else:
            raise ValueError(
                f"Unknown weights: {self.weights}"
            )
        


    def predict(self, X):
        prediction = [self._predict(x) for x in X]
        return prediction


