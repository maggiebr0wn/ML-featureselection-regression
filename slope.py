import numpy as np
from pca import PCA
from regression import Regression

class Slope(object):

    def __init__(self):
        pass

    def pca_slope(self, X, y):
        """
        Calculates the slope of the first principal component given by PCA

        Args: 
            x: (N,) vector of feature x
            y: (N,) vector of feature y
        Return:
            slope: Scalar slope of the first principal component
        """

        comb = np.array((X,y))
        pca = PCA()
        pca.fit(comb.T)
        V = pca.get_V()

        slope = V.T[1][0]/V.T[0][0]

        return slope

        raise NotImplementedError
   
    def lr_slope(self, X, y):
        """
        Calculates the slope of the best fit as given by Linear Regression
        
        For this function don't use any regularization

        Args: 
            X: N*1 array corresponding to a dataset
            y: N*1 array of labels y
        Return:
            slope: slope of the best fit
        """

        lr = Regression()
        w = lr.linear_fit_closed(X, y)[0]

        return w

        raise NotImplementedError

    def addNoise(self, c, x_noise = False, seed = 1):
        """
        Creates a dataset with noise and calculates the slope of the dataset
        using the pca_slope and lr_slope functions implemented in this class.

        Args: 
            c: Scalar, a given noise level to be used on Y and/or X
            x_noise: Boolean. When set to False, X should not have noise added
                     When set to True, X should have noise
            seed: Random seed
        Return:
            pca_slope_value: slope value of dataset created using pca_slope
            lr_slope_value: slope value of dataset created using lr_slope

        """
        np.random.seed(seed) #### DO NOT CHANGE THIS ####
        ############# START YOUR CODE BELOW #############

        X = np.zeros(1000)
        for i in range(1000):
            X[i] = (i+1)/1000
        
        if x_noise == False:

            y = (2*X) + np.random.normal(loc=[0], scale = c, size=X.shape)
             
            pca_slope_value = self.pca_slope(X, y) 
            lr_slope_value = self.lr_slope(X[:, np.newaxis], y[:, np.newaxis])

        elif x_noise == True:

            X = X + np.random.normal(loc=[0], scale = c, size=X.shape)

            y = (2*X) + np.random.normal(loc=[0], scale = c, size=X.shape)

            pca_slope_value = self.pca_slope(X, y)
            lr_slope_value = self.lr_slope(X[:, np.newaxis], y[:, np.newaxis])
        
        return pca_slope_value, lr_slope_value

        raise NotImplementedError












