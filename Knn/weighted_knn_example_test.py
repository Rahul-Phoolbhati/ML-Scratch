
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