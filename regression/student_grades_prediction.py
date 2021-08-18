import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sklearn.model_selection
from sklearn import linear_model


TRAIN_DATA_FILE_PATH = '../data/student-por.csv'
TEST_DATA_FILE_PATH = '../data/student-mat.csv'
FEATURES = ['G1', 'G2', 'G3']
PREDICT = 'G3'
TEST_SET_RATIO=0.1
TRAIN_COUNT = 100


def get_data(data_file_path, features):
    data = pd.read_csv(data_file_path, sep=';')
    return data[features]


def plot():
    data = get_data(TRAIN_DATA_FILE_PATH,FEATURES)
    for _feature in FEATURES:
        plt.scatter(data[_feature], data[PREDICT])
        plt.xlabel(_feature)
        plt.ylabel('Final Grade')
        plt.show()


def predict_final_grades():
    data = get_data(TEST_DATA_FILE_PATH, FEATURES)
    print(data.head)
    x = np.array(data.drop([PREDICT], 1))
    y = np.array(data[PREDICT])

    models = sorted(os.listdir('models'), reverse=True)
    pickle_in = open(f'models/{models[0]}', 'rb')
    linear = pickle.load(pickle_in)
    predictions = linear.predict(x)

    for i in range(len(predictions)):
        print('{:.2f}'.format(predictions[i]), y[i])

    print('-------------------------')
    print('Coefficient: \n', linear.coef_)
    print('-------------------------')


def save_model(linear, accuracy, features):
    with open(f'models/{accuracy}-{features}', 'wb') as f:
        pickle.dump(linear, f)


def train(train_count):

    data = get_data(TRAIN_DATA_FILE_PATH, FEATURES)

    x = np.array(data.drop([PREDICT], 1))
    y = np.array(data[PREDICT])

    models = sorted(os.listdir('models'), reverse=True)
    current_best_accuracy = models[0].split('-')[0]

    for i in range(train_count):
        # split train and test set
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=TEST_SET_RATIO)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)

        accuracy = '{:.2f}'.format(linear.score(x_test,y_test) * 100)
        print(f'model trained accuracy {accuracy}')

        if accuracy > current_best_accuracy:
            print(f'Saving model.. with {accuracy}% accuracy')
            _features = ','.join(FEATURES).replace(',','-')
            save_model(linear, accuracy, _features)


def main():
    train(TRAIN_COUNT)
    plot()
    predict_final_grades()


if __name__ == '__main__':
    main()