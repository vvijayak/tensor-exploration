import pandas as pd
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

"""
Class Values:

unacc, acc, good, vgood

Attributes:

buying: vhigh, high, med, low.
maint: vhigh, high, med, low.
doors: 2, 3, 4, 5more.
persons: 2, 4, more.
lug_boot: small, med, big.
safety: low, med, high.

"""


K_HYPER_PARAMETERS = [3, 5, 7, 9, 11]
PREDICT = 'class'
SHOW_PREDICTIONS = False
TRAIN_TEST_DATA_PATH = '../data/car.data'
TEST_SET_RATIO=0.1

# read in data
data = pd.read_csv(TRAIN_TEST_DATA_PATH)

label_encoder = preprocessing.LabelEncoder()

# convert non-numerical values to numerical ones, return list
cls = label_encoder.fit_transform(list(data['class']))
buying = label_encoder.fit_transform(list(data['buying']))
maintenance = label_encoder.fit_transform(list(data['maint']))
doors = label_encoder.fit_transform(list(data['door']))
persons = label_encoder.fit_transform(list(data['persons']))
lug_boot = label_encoder.fit_transform(list(data['lug_boot']))
safety = label_encoder.fit_transform(list(data['safety']))

X = list(zip(buying, maintenance, doors, persons, lug_boot, safety))
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
        names = ["unacc", "acc", "good", "vgood"]
        for i in range(len(predictions)):
            print(f'prediction: {names[predictions[i]]} label: {names[y_test[i]]} features: {x_test}')
