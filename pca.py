import numpy as np
from matplotlib import pyplot as plt

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X): # 5 points
        """
        Decompose dataset into principal components.
        You may use your SVD function from the previous part in your implementation or numpy.linalg.svd function.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """

        # center the data

        M = X.mean(axis = 0)

        Xcentered = np.subtract(X, np.transpose(M[:, np.newaxis]))

        self.U, self.S, self.V = np.linalg.svd(Xcentered, full_matrices=False)

        return None
        raise NotImplementedError

    def transform(self, data, K=2): # 2 pts
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """

        new_mat = data @ np.transpose(self.V)

        X_new = new_mat[:, :K]

        return X_new

        raise NotImplementedError


    def transform_rv(self, data, retained_variance=0.99): # 3 pts
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """

        U = self.U
        S = self.S
        V = self.V

        new_mat = data @ np.transpose(self.V)

        numsum = 0
        densum = 0

        # total var
        densum = 0
        for j in range(len(S)):
            densum = densum + (S[j]**2)        

        # recovered var   
        num_cols = new_mat.shape[1]
        for i in range(num_cols): # for each column, compute cumulative variance
            recovered_var = 0
            if recovered_var < retained_variance:
                numsum = numsum + (S[i]**2)    
                recovered_var = numsum/densum
                cols_n = i
        
        X_new = new_mat[:, :i]

        return X_new

        raise NotImplementedError


    def get_V(self):
        """ Getter function for value of V """
        
        return self.V
    
    def visualize(self, X, y): 
        """
        Use your PCA implementation to reduce the dataset to only 2 features.

        Create a scatter plot of the reduced data set and differentiate points that
        have different true labels using color.

        Args:
            xtrain: NxD numpy array, where N is number of instances and D is the dimensionality 
            of each instance
            ytrain: numpy array (N,), the true labels
            
        Return: None
        """

        self.fit(X)

        X_new = self.transform(X, 2)

        print(X_new.shape)

        X_coords = X_new[:, 0]
        y_coords = X_new[:, 1]


        # extract class 0 and class 1 separately 

        class0_x = []
        class0_y = []

        class1_x = []
        class1_y = []

        for i in range(len(X_coords)):
            if y[i] == 0:
                class0_x.append(X_coords[i])
                class0_y.append(y_coords[i])
            elif y[i] == 1:
                class1_x.append(X_coords[i])
                class1_y.append(y_coords[i])

        pl0 = plt.scatter(class0_x, class0_y, color='red')
        pl1 = plt.scatter(class1_x, class1_y, color='blue')

        plt.legend([pl0, pl1], ["0", "1"])

        return None

        raise NotImplementedError
        
        
        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend()
        plt.show()






