'''
File: nb.py
Project: Downloads
File Created: March2021
Author: Yuzi Hu (yhu495@gatech.edu)
'''

import numpy as np
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_SEED = 5


class NaiveBayes(object):
    def __init__(self):
        pass

    def likelihood_ratio(self, X_negative, X_neutral, X_positive):  # [5pts]
        '''
        Args:
            X_negative: N_negative x D where N_negative is the number of negative news that we have,
                while D is the number of features (we use the word count as the feature)
            X_neutral: N_neutral x D where N_neutral is the number of neutral news that we have,
                while D is the number of features (we use the word count as the feature)
            X_positive: N_positive x D where N_positive is the number of positive news that we have,
                while D is the number of features (we use the word count as the feature)
        Return:
            likelihood_ratio: 3 x D matrix of the likelihood ratio of different words for different class of news
        '''

        #neg_sum = (np.sum((np.sum(X_negative, axis = 0)), 1, axis = 0))/(X_negative.shape[1])
        #neut_sum = (np.sum((np.sum(X_neutral, axis = 0)), 1, axis = 0))/(X_neutral.shape[1])
        #pos_sum = (np.sum((np.sum(X_positive, axis = 0)), 1, axis = 0))/(X_positive.shape[1])
        neg_sum = (np.sum(X_negative, axis = 0) + 1)/(np.sum(X_negative) + X_negative.shape[1])
        neut_sum = (np.sum(X_neutral, axis = 0) + 1)/(np.sum(X_neutral) + X_neutral.shape[1])
        pos_sum = (np.sum(X_positive, axis = 0) + 1)/(np.sum(X_positive) + X_positive.shape[1])

        likelihood_ratio = np.vstack((neg_sum, neut_sum, pos_sum))

        return likelihood_ratio

        raise NotImplementedError

    def priors_prob(self, X_negative, X_neutral, X_positive):  # [5pts]
        '''
        Args:
            X_negative: N_negative x D where N_negative is the number of negative news that we have,
                while D is the number of features (we use the word count as the feature)
            X_neutral: N_neutral x D where N_neutral is the number of neutral news that we have,
                while D is the number of features (we use the word count as the feature)
            X_positive: N_positive x D where N_positive is the number of positive news that we have,
                while D is the number of features (we use the word count as the feature)
        Return:
            priors_prob: 1 x 3 matrix where each entry denotes the prior probability for each class
        '''

        tot = np.sum(X_negative) + np.sum(X_neutral) + np.sum(X_positive)

        neg = np.sum(X_negative)/tot
        neut = np.sum(X_neutral)/tot
        pos = np.sum(X_positive)/tot

        priors_prob = np.array([neg, neut, pos])

        return priors_prob

        raise NotImplementedError

    # [5pts]
    def analyze_sentiment(self, likelihood_ratio, priors_prob, X_test):
        '''
        Args:
            likelihood_ratio: 3 x D matrix of the likelihood ratio of different words for different class of news
            priors_prob: 1 x 3 matrix where each entry denotes the prior probability for each class
            X_test: N_test x D bag of words representation of the N_test number of news that we need to analyze its sentiment
        Return:
             1 x N_test list, each entry is a class label indicating the sentiment of the news (negative: 0, neutral: 1, positive: 2)

        '''
        labels = []

        for row in X_test:
            #test_arr = np.vstack([(X_test[i]), (X_test[i]), (X_test[i])])
            test_arr = np.vstack([row, row, row])
            exp_arr = np.power(likelihood_ratio, test_arr)
            prod_arr = np.prod(exp_arr, axis = 1)
            check_arr = priors_prob.T * prod_arr

            labels.append(np.argmax(check_arr))

        return labels


        raise NotImplementedError






