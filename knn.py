from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
x = pd.read_csv('datafix.csv')
y = pd.read_csv('label.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
n=20
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_train,y_train)
knnsplit = knn.score(x_test,y_test)
cv_scores = cross_val_score(knn, x, y, cv=2)

print(cv_scores)
print(knnsplit)
