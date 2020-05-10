# EECS 445 - Winter 2018
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string
import nltk

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt
from random import uniform
from nltk.corpus import stopwords

from helper import *


def text_process(text):
    strip_punc = [c for c in text if c not in string.punctuation]
    strip_punc = ''.join(strip_punc)
    strip_punc = strip_punc.lower()
    stopwords = nltk.corpus.stopwords.words('english')
    newstopwords = ['4', '2', '7', '3', '5', '1', '8', '0', '9', 'f', 'n', 'g', 'u', 'w', 'b', 'p', 'r', '6', 'k', 'x',
                    'cs', 'kp', 'kn', 'fa', 'ua', 'fo', 'st', 'jt', 'rr', 'pr', 'ey', 'gt', 'ff',
                    'lk', 'yo', 'um', 'jj', 'jh', 'ya', 'cr', 'th', 'lh', 'http']
    stopwords.extend(newstopwords)
    return [word for word in strip_punc.split() if word.lower() not in stopwords]


def compute_score(y_pred, y_true, metric):
    """
    :param y_pred: Predicted y values from svm model
    :param y_true: True y values from test data
    :param metric: Metric to be evaluated and returned
    :return: Returns the metric specified, computed from y_pred and y_true
    """

    # Challenge
    n = len(y_true)
    conf_mat = metrics.confusion_matrix(y_true, y_pred, [1, 0, -1])
    tp = 0
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat[i])):
            if i == j:
                tp += conf_mat[i][j]
    return tp / n

    # if metric == 'auroc':
    #     return metrics.roc_auc_score(y_true, y_pred)
    #
    # tp, fn, fp, tn = metrics.confusion_matrix(y_true, y_pred, [1, -1]).ravel()
    #
    # if metric == 'accuracy':
    #     return metrics.accuracy_score(y_true, y_pred)
    # if metric == 'precision':
    #     return metrics.precision_score(y_true, y_pred)
    # if metric == 'sensitivity':
    #     return tp/(tp+fn)
    # if metric == 'specificity':
    #     return tn/(tn+fp)
    # if metric == 'f1-score':
    #     return metrics.f1_score(y_true, y_pred)


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.

    if penalty == 'l1':
        clf = LinearSVC(penalty=penalty, C=c, dual=False, class_weight='balanced')
    else:
        clf = LinearSVC(C=c, kernel='linear', class_weight='balanced')

    return clf


def extract_dictionary(df):
    """
    Reads a panda dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """

    word_dict = {}

    # TODO: Implement this function

    # Challenge
    X = df['text']

    for rev in range(len(X)):
        arr = text_process(X[rev])
        for w in arr:
            if w not in word_dict:
                word_dict[w] = 0
            word_dict[w] += 1

    word_dict = {k: v for k, v in word_dict.items() if v != 1}

    # for i in range(len(list(df["text"]))):
    #     for punc in string.punctuation:
    #         df.iat[i, 1] = df.iat[i, 1].replace(punc, " ")
    #     df.iat[i, 1] = df.iat[i, 1].lower()
    #     for word in df.iat[i, 1].split():
    #         if word not in word_dict:
    #             word_dict[word] = 0
    #         word_dict[word] += 1

    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (number of reviews, number of words).
    Input:
        df: dataframe that has the ratings and labels
        word_list: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (number of reviews, number of words)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words+1))
    # TODO: Implement this function

    # dict = list(word_dict.keys())
    # col = list(df["text"])
    #
    # for i in range(number_of_reviews):
    #     review = col[i].split()
    #     for j in range(number_of_words):
    #         if dict[j] in review:
    #             feature_matrix[i][j] = 1

    # Challenge
    dict = list(word_dict.keys())
    X = df['text']
    rt = df['retweet_count']

    for i in range(number_of_reviews):
        review = text_process(X[i])
        for j in range(number_of_words):
            if dict[j] in review:
                feature_matrix[i][j] = 1
        feature_matrix[i][number_of_words] = rt[i]

    return feature_matrix


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful

    #Put the performance of the model on each fold in the scores array
    scores = []

    skf = StratifiedKFold(n_splits=k, shuffle=False)
    skf.get_n_splits(X, y)

    for train_index, test_index in skf.split(X, y):
        clf.fit(X[train_index], y[train_index])
        if metric == "auroc":
            pred = clf.decision_function(X[test_index])
        else:
            pred = clf.predict(X[test_index])
        actual = y[test_index]
        scores.append(compute_score(pred, actual, metric))

    #And return the average performance across all fold splits.
    return np.array(scores).mean()


def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM

    print(metric)

    C_final = C_range[0]
    score = 0
    for c in C_range:
        clf = SVC(C=c, kernel="linear")
        print(c)
        print(cv_performance(clf, X, y, k, metric))
        print()
        if cv_performance(clf, X, y, k, metric) > score:
            score = cv_performance(clf, X, y, k, metric)
            C_final = c

    print(C_final)
    print(score)
    return C_final


def plot_weight(X,y,penalty,metric,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)

    for c in C_range:
        clf = select_classifier(penalty, c)
        clf.fit(X, y)
        norm0.append(np.count_nonzero(clf.coef_))


    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            parameter_values: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter value(s) for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...

    #
    # Grid Search
    #
    # print("-------------Part 3.2.b--------------")
    # print("-------------Grid Search-------------")
    # final_c = 0
    # final_r = 0
    # final_score = 0
    # for c_ind in range(len(param_range)):
    #     for r_ind in range(len(param_range)):
    #         c = param_range[c_ind][0]
    #         r = param_range[r_ind][1]
    #         clf = SVC(kernel="poly", degree=2, C=c, coef0=r, class_weight="balanced")
    #         score = cv_performance(clf, X, y, k, metric)
    #         print("Score: ")
    #         print(score)
    #         print(c)
    #         print(r)
    #         print()
    #         if score > final_score:
    #             final_score = score
    #             final_c = c
    #             final_r = r
    #
    # print(final_score)
    # print(final_c)
    # print(final_r)

    #
    # Random Search
    #
    print("-------------Part 3.2.b--------------")
    print("-----------Random Search-------------")
    final_c = 0
    final_r = 0
    final_score = 0

    for i in range(25):
        c = 10 ** uniform(-3, 3)
        r = 10 ** uniform(-3, 3)
        clf = SVC(kernel="poly", degree=2, C=c, coef0=r, class_weight="balanced")
        score = cv_performance(clf, X, y, k, metric)
        print("Score: ")
        print(score)
        print(c)
        print(r)
        print()
        if score > final_score:
            final_score = score
            final_c = c
            final_r = r

    print(final_score)
    print(final_c)
    print(final_r)


def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.


def part_2(x_train):
    print("-------------Part 2--------------")
    print(x_train.shape[1])
    tot = 0
    for i in range(x_train.shape[0]):
        tot += sum(x_train[i])
    print(tot/x_train.shape[0])


def part_3_c(X_train, Y_train, X_test, Y_test):
    print("-------------Part 3_c--------------")
    c_range = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    m_s = ['accuracy', 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
    for m in m_s:
        select_param_linear(X_train, Y_train, k=5, metric=m, C_range=c_range)


def part_3_d(X_train, Y_train, X_test, Y_test, metric="accuracy"):
    print("-------------Part 3_d--------------")
    clf = SVC(C=0.1, kernel="linear", class_weight="balanced")
    clf.fit(X_train, Y_train)
    if metric == "auroc":
        pred = clf.decision_function(X_test)
    else:
        pred = clf.predict(X_test)

    print(metric)
    print(compute_score(pred, Y_test, metric))


def part_3_f(X_train, Y_train, dict):
    print("-------------Part 3_f--------------")
    clf = SVC(C=0.1, kernel="linear")
    clf.fit(X_train, Y_train)
    a = np.array(clf.coef_[0])
    top_ind = np.argpartition(a, -4)[-4:]
    bot_ind = np.argpartition(a, 4)[:4]

    print(top_ind)
    print(a[top_ind])
    print(bot_ind)
    print(a[bot_ind])
    for i in top_ind:
        print(list(dict.keys())[i])
    for ix in bot_ind:
        print(list(dict.keys())[ix])


def part_3_4_a(X_train, Y_train, k=5, metric='accuracy'):
    print("-------------Part 3_4_a--------------")
    final_c = 0
    final_score = 0
    c_range = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    for c in c_range:
        clf = LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced')
        score = cv_performance(clf, X_train, Y_train, k, metric)
        print(c)
        print(score)
        print()
        if score > final_score:
            final_score = score
            final_c = c

    print(final_score)
    print(final_c)


def part_3_4_b(X_train, Y_train):
    print("-------------Part 3_4_b--------------")
    c_range = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    plot_weight(X_train, Y_train, penalty='l1', metric='auroc', C_range=c_range)


def part_4_1_b(X_train, Y_train, X_test, Y_test, mets):
    print("-------------Part 4_1_b--------------")
    for m in mets:
        #clf = SVC(C=0.01, kernel='linear', class_weight={-1: 10, 1: 1})
        clf = SVC(kernel='linear', C=0.01, class_weight={-1: 10, 1: 1})
        clf.fit(X_train, Y_train)
        if m == 'auroc':
            pred = clf.decision_function(X_test)
        else:
            pred = clf.predict(X_test)
        print(m)
        print(compute_score(pred, Y_test, m))


def part_4_2(X_train, Y_train, X_test, Y_test, mets):
    print("-------------Part 4_2--------------")
    for m in mets:
        clf = SVC(C=0.01, kernel='linear', class_weight={-1: 1, 1: 1})
        clf.fit(X_train, Y_train)
        if m == 'auroc':
            pred = clf.decision_function(X_test)
        else:
            pred = clf.predict(X_test)
        print(m)
        print(compute_score(pred, Y_test, m))


def part_4_3(X_train, Y_train, X_test, Y_test, mets):
    print("-------------Part 4_3--------------")
    w_p = [6, 7, 8, 9, 10]
    w_n = [1, 2, 3, 4, 5]

    final_pweight = 0
    final_nweight = 0
    final_score = 0
    for p in w_p:
        for n in w_n:
            clf = SVC(C=0.01, kernel='linear', class_weight={-1: n, 1: p})
            score = cv_performance(clf, X_train, Y_train, 5, 'f1-score')
            print(score)
            print(p)
            print(n)
            print()
            if score > final_score:
                final_score = score
                final_pweight = p
                final_nweight = n

    print(final_score)
    print(final_pweight)
    print(final_nweight)


def part_4_4(X_train, Y_train, X_test, Y_test):
    print("-------------Part 4_4--------------")
    clf1 = SVC(C=0.01, kernel='linear', class_weight='balanced')
    clf2 = SVC(C=0.01, kernel='linear', class_weight={-1: 5, 1: 9})
    clf1.fit(X_train, Y_train)
    clf2.fit(X_train, Y_train)
    pred1 = clf1.decision_function(X_test)
    pred2 = clf2.decision_function(X_test)
    fpr1, tpr1, threshold1 = metrics.roc_curve(Y_test, pred1)
    fpr2, tpr2, threshold2 = metrics.roc_curve(Y_test, pred2)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr1, tpr1, 'b', label='AUC(W_p = 1, W_n = 1) = %0.2f' % roc_auc1)
    plt.plot(fpr2, tpr2, 'r', label='AUC(W_p = 9, W_n = 5) = %0.2f' % roc_auc2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('4_4_AUROC.png')
    plt.close()


def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    # IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    # TODO: Questions 2, 3, 4

    # part_2(X_train)
    #
    # # Part 3.1.c
    # part_3_c(X_train, Y_train, X_test, Y_test)
    #
    # # Part 3.1.d
    # m_s = ['accuracy', 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
    # for m in m_s:
    #     part_3_d(X_train, Y_train, X_test, Y_test, m)
    #
    # # Part 3.1.e
    # c_range = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    # plot_weight(X_train, Y_train, "l2", "accuracy", c_range)
    #
    # # Part 3.1.f
    # part_3_f(X_train, Y_train, dictionary_binary)
    #
    # # Part 3.2.b
    # pr = np.array([[10 ** -3, 10 ** -3],
    #                 [10 ** -2, 10 ** -2],
    #                 [10 ** -1, 10 ** -1],
    #                 [10 ** 0, 10 ** 0],
    #                 [10 ** 1, 10 ** 1],
    #                 [10 ** 2, 10 ** 2],
    #                 [10 ** 3, 10 ** 3]])
    # select_param_quadratic(X_train, Y_train, 5, metric="auroc", param_range=pr)
    #
    # # Part 3.4.a
    # part_3_4_a(X_train, Y_train, 5, metric='auroc')
    #
    # # Part 3.4.b
    # part_3_4_b(X_train, Y_train)
    #
    # # Part 4.1.b
    # part_4_1_b(X_train, Y_train, X_test, Y_test, m_s)
    #
    # # Part 4.2
    # part_4_2(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, m_s)
    #
    # # Part 4.3
    # part_4_3(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, m_s)
    # clf = SVC(C=0.01, kernel='linear', class_weight={-1: 5, 1: 9})
    # clf.fit(IMB_features, IMB_labels)
    # for m in m_s:
    #     if m == 'auroc':
    #         pred = clf.decision_function(IMB_test_features)
    #     else:
    #         pred = clf.predict(IMB_test_features)
    #     print(m)
    #     print(compute_score(pred, IMB_test_labels, m))
    #
    # # Part 4.4
    # part_4_4(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)
    #



    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    print('====Challenge====')
    clf = LinearSVC(C=0.1, loss='hinge')
    print(cv_performance(clf, multiclass_features, multiclass_labels, 5, 'accuracy'))
    clf = LinearSVC(C=0.1, loss='hinge')
    clf.fit(multiclass_features, multiclass_labels)
    pred = clf.predict(heldout_features)
    generate_challenge_labels(pred, 'cadelau')


if __name__ == '__main__':
    main()
