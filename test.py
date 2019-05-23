import os
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


DATA_PATH = "/Users/kevin/desktop/env/data/"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"


def parse_data(filename):
    df_org = pd.read_csv(filename)
    x = df_org["Address"]
    y = df_org["Is address"]
    return x, y


def NB_classifier_model(train_xs, test_xs):
    vect = TfidfVectorizer()
    X_train = vect.fit_transform(train_xs)
    X_test = vect.transform(test_xs)
    return X_train, X_test


# naive bayes classifier - feed the training data and predict the test data
def NB_model(X_train, X_test, train_ys, test_ys):
    text_clf_MNB = MultinomialNB()
    text_clf_MNB.fit(X_train, train_ys)
    predicted_NB = text_clf_MNB.predict(X_test)
    score = np.mean(predicted_NB == test_ys)
    return score


# support vector machine - with given learning rate
def SVM_model(X_train, X_test, train_ys, test_ys, args):
    text_clf_SVM = SGDClassifier(loss='hinge', penalty='none', alpha=args.lr, random_state=42)
    text_clf_SVM.fit(X_train, train_ys)
    predicted_SVM = text_clf_SVM.predict(X_test)
    score = np.mean(predicted_SVM == test_ys)
    return score


def main():

    ########## args parser ##########

    df_test_file = DATA_PATH + TRAIN_FILE_NAME

    parser = argparse.ArgumentParser(description='Linear Classifier for Identify Addresses')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--test', type=str, default=df_test_file, help='test file')
    args = parser.parse_args()

    # training dataset
    train_xs, train_ys = parse_data(DATA_PATH + args.test)

    # testing dataset
    test_xs, test_ys = parse_data(DATA_PATH + args.test)

    # naive bayes classifer
    X_train, X_test = NB_classifier_model(train_xs, test_xs)

    # score for Naive Bayes
    NB_score = NB_model(X_train, X_test, train_ys, test_ys)

    # score for SVM
    SVM_score = SVM_model(X_train, X_test, train_ys, test_ys, args)

    print('\n-----------------------------')
    print('Multinomial Naive Bayes (MNB)')
    print('MNB_Score = ' + str(NB_score))

    print('Support Vector Machine (SVM)')
    print('SVM Score = ' + str(SVM_score))
    print('-----------------------------\n')


if __name__ == '__main__':
    main()
