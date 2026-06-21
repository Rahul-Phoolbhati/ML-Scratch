
from knn import KNN

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

knn = KNN(k=3)

knn.fit(X_train, y_train)

X_test = [
    [3, 3],
    [8.5, 8.5]
]

print(knn.predict(X_test))