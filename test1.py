import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
fish = pd.read_csv("fish.csv")
fish.head()
len(fish)
X = fish.drop(['Species'], axis = 'columns')
y = fish.Species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
len(X_train)
model = SVC(kernel = 'linear', C = 1)
model.fit(X_train, y_train)
svm_pred = model.predict(X_test)
svm_pred
accuracy = model.score(X_test, y_test)
accuracy
model.predict([[242,23,25,30,11,4]])
model.predict([[12,11,12,13,2,1]])



