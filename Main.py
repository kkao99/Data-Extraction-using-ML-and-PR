import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


DATA_PATH = "/Users/kevin/desktop/env/data/"
# FILE_NAME = "zomato_address_only.csv"
FILE_NAME = "zomato.csv"

'''
def add_Y_to_file(file_name):
    file = open(file_name)
    reader = csv.reader(file)
    row0 = next(reader)
    row0.append("Is_Address")

    csv_input = pd.read_csv(FILE_NAME)
    csv_input["Is Address"] =
    csv_input.to_csv('output.csv', index=False)
'''


def main():
    df_org = pd.read_csv(DATA_PATH + FILE_NAME)

    # print(df_org["Address"])

    target_data = df_org

    # separate the training set into 80% for training and 20% for testing
    msk = np.random.rand(len(target_data)) < 0.8
    train_data = target_data[msk]
    test_data = target_data[~msk]

    # print(train_data["Address"])

    #################### Naive Bayes Classifier Model ####################

    vect = TfidfVectorizer()

    X_train = vect.fit_transform(train_data["Address"])
    X_test = vect.fit_transform(test_data["Address"])

    #################### Naive Bayes - Model ####################

    text_clf_MNB = MultinomialNB()

    # create an array of ones, whose length equal to the size of the train_data
    # train_is_address = np.ones((len(train_data),), dtype=int)

    text_clf_MNB.fit(X_train, train_data["Is address"])

    # TODO: ERROR!
    predicted_NB = text_clf_MNB.predict(X_test)

    # create an array of ones, whose length equal to the size of the test_data
    # test_is_address = np.ones((len(test_data),), dtype=int)
    # NB_score = np.mean(predicted_NB == test_is_address)


if __name__ == "__main__":
    main()
