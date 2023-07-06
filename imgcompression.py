import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N for black and white images / N * N * 3 for color images
            S: min(N, D) * 1 for black and white images / min(N, D) * 3 for color images
            V: D * D for black and white images / D * D * 3 for color images
        """

        if len(X.shape) == 2: # b&w

            #print("B&W", X.shape)
            U, S, V = np.linalg.svd(X, full_matrices=True)

        elif len(X.shape) == 3: # color

            #print("Color", X.shape)

            red_arr = X[:, :, 0]
            green_arr = X[:, :, 1]
            blue_arr = X[:, :, 2]

            Ured, Sred, Vred = np.linalg.svd(red_arr, full_matrices=True)
            Ugr, Sgr, Vgr = np.linalg.svd(green_arr, full_matrices=True)
            Ublu, Sblu, Vblu = np.linalg.svd(blue_arr, full_matrices=True)

            U = np.stack((Ured, Ugr, Ublu), axis=2)
            S = np.stack((Sred, Sgr, Sblu), axis=1)
            V = np.stack((Vred, Vgr, Vblu), axis=2)


        #print(U.shape, S.shape, V.shape)

        return U, S, V

        raise NotImplementedError


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """

        if len(U.shape) == 2: #b&w

            # make S have correct dimensions
            n = U.shape[0] # no. samps
            d = V.shape[0] # no. features

            s = np.zeros((n, d))
            for i in range(n):
                s[i, i] = S[i]

            Xrebuild = U @ s[:, :k] @ V[:k, :]

            #print("S", s.shape)

            #print("re-B&W", Xrebuild.shape)
        
        elif len(U.shape) == 3: #color

            #print("S", S.shape)
            #print("U", U.shape)
            #print("V", V.shape)

            Sred = S[:, 0]
            Sgr = S[:, 1]
            Sblu = S[:, 2]

            Ured = U[:, :, 0]
            Ugr = U[:, :, 1]
            Ublu = U[:, :, 2]

            Vred = V[:, :, 0]
            Vgr = V[:, :, 1]
            Vblu = V[:, :, 2]

            n = Ured.shape[0] # no. samps
            d = Vred.shape[0] # no. features

            # rebuild red
            sred = np.zeros((n, d))
            for i in range(n):
                sred[i, i] = Sred[i]

            Xrebuild_red = Ured @ sred[:, :k] @ Vred[:k, :]

            # rebuild green
            sgr = np.zeros((n, d))
            for i in range(n):
                sgr[i, i] = Sgr[i]

            Xrebuild_gr= Ugr @ sgr[:, :k] @ Vgr[:k, :]

            #rebuild blue
            sblu = np.zeros((n, d))
            for i in range(n):
                sblu[i, i] = Sblu[i]

            Xrebuild_blu = Ublu @ sblu[:, :k] @ Vblu[:k, :]

            # stack em r,g,b
            Xrebuild = np.stack((Xrebuild_red, Xrebuild_gr, Xrebuild_blu), axis=2)

        return(Xrebuild)

        raise NotImplementedError
        

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in compressed)/(num stored values in original)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """

        if len(X.shape) == 2: #b&w
            #print(X.shape)
            uncomp_pixs = X.shape[0] * X.shape[1]
            comp_pixs = k * (1 + X.shape[0] + X.shape[1])

            compression_ratio = comp_pixs/uncomp_pixs

        elif len(X.shape) == 3: #color
            #print(X.shape)
            uncomp_pixs = X.shape[0] * X.shape[1]
            comp_pixs = k * (1 + X.shape[0] + X.shape[1])

            compression_ratio = comp_pixs/uncomp_pixs

        return(compression_ratio)

        raise NotImplementedError


    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """

        if len(S.shape) == 1: #b&w

            numsum = 0
            for i in range(k):
                numsum = numsum + (S[i]**2)

            densum = 0
            for j in range(len(S)):
                densum = densum + (S[j]**2)           

            recovered_var = numsum/densum

        elif len(S.shape) == 2: #color

            numsum = 0
            for i in range(k):
                numsum = numsum + (S[i]**2)

            densum = 0
            for j in range(len(S)):
                densum = densum + (S[j]**2)           

            recovered_var = numsum/densum

        return recovered_var

        raise NotImplementedError

        
