
from weighted_knn import KNN

# small Test 
X_train = [
    [1, 1],
    [2, 2],
    [8, 8],
    [9, 9]
]

y_train = [
    "A",
    "A",
    "B",
    "B"
]

knn = KNN(
    k=3,
    weights="uniform"
)

knn.fit(X_train, y_train)

X_test = [
    [3, 3],
    [8.5, 8.5]
]

print(knn.predict(X_test))


# Distance weight testing

X_train = [
    [0],
    [10],
    [11]
]

y_train = [
    "A",
    "B",
    "B"
]

knn = KNN(
    k=3,
    weights="distance"
)

knn.fit(X_train, y_train)

print(knn.predict([[1]]))

#  exact match test 

X_train = [
    [1, 1],
    [5, 5]
]

y_train = [
    "A",
    "B"
]

knn = KNN(
    k=2,
    weights="distance"
)

knn.fit(X_train, y_train)

print(knn.predict([[1, 1]]))



# Custom weight function test


X_train = [
    [1, 1],
    [2, 2],
    [3, 3],
    [8, 8],
    [9, 9]
]

y_train = [
    "A",
    "A",
    "A",
    "B",
    "B"
]

def custom_weight(k_nearest, y_train):
    """
    Stronger penalty for distance (squared inverse)
    """
    votes = {}

    for ind, dist in k_nearest:
        label = y_train[ind]
        weight = 1 / ((dist ** 2) + 1e-10)
        votes[label] = votes.get(label, 0) + weight

    return max(votes, key=votes.get)

knn3 = KNN(k=3, weights=custom_weight)
knn3.fit(X_train, y_train)

print("Custom    :", knn3.predict(X_test))