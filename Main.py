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
# FILE_NAME = "zomato.csv"
FILE_NAME = "address.csv"

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
    X_test = vect.transform(test_data["Address"])

    #################### Naive Bayes - Model ####################

    text_clf_MNB = MultinomialNB()

    # create an array of ones, whose length equal to the size of the train_data
    # train_is_address = np.ones((len(train_data),), dtype=int)

    text_clf_MNB.fit(X_train, train_data["Is address"])

    predicted_NB = text_clf_MNB.predict(X_test)

    # create an array of ones, whose length equal to the size of the test_data
    # test_is_address = np.ones((len(test_data),), dtype=int)
    NB_score = np.mean(predicted_NB == test_data["Is address"])

    #################### Support Vector Machine - Model ####################

    text_clf_SVM = SGDClassifier(loss='hinge', penalty='none', alpha=1e-3, random_state=42)

    text_clf_SVM.fit(X_train, train_data["Is address"])
    predicted_SVM = text_clf_SVM.predict(X_test)
    SVM_score = np.mean(predicted_SVM == test_data["Is address"])

    print(FILE_NAME)
    print('Multinomial Naive Bayes (MNB)')
    print('MNB_Score = ' + str(NB_score))

    print('Support Vector Machine (SVM)')
    print('SVM Score = ' + str(SVM_score))



if __name__ == "__main__":
    main()
