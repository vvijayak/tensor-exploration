import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

"""
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica
"""

K_HYPER_PARAMETERS = [3, 5, 7, 9, 11]
PREDICT = 'class'
SHOW_PREDICTIONS = False
TRAIN_TEST_DATA_PATH = '../data/iris.data'
TEST_SET_RATIO=0.1


# read in data
data = pd.read_csv(TRAIN_TEST_DATA_PATH)

label_encoder = preprocessing.LabelEncoder()

# transform non-numerical fields
cls = label_encoder.fit_transform(list(data['class']))
print(data.head)

X = np.array(data.drop([PREDICT], 1))
y = list(cls)

# split train and test set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=TEST_SET_RATIO)

# train model and get model accuracy
for hyper_parameter in K_HYPER_PARAMETERS:
    model = KNeighborsClassifier(n_neighbors=hyper_parameter)
    model.fit(x_train, y_train)
    accuracy = "{:.2f}".format(model.score(x_test, y_test) * 100)
    print(f'k-hpyer-parameter: {hyper_parameter} accuracy: {accuracy}')

    if SHOW_PREDICTIONS:
        predictions = model.predict(x_test)
        names = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
        for i in range(len(predictions)):
            print(f'prediction: {names[predictions[i]]} label: {names[y_test[i]]} features: {x_test}')
