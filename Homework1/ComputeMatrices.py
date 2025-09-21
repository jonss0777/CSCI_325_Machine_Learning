# you should fill in the functions in this file,
# do NOT change the name, input and output of these functions

import numpy as np
import time
import matplotlib.pyplot as plt


# Notes

# What is the pairwise eucledian distance?
# Calculating the pairwise eucledian distance allows us to create a 
# distance matrix between vectors, similar to the distance matrix of a weighted graph. 


# first function to fill, compute distance matrix using loops
def compute_distance_naive(X):
    N = X.shape[0]      # num of rows
    D = X.shape[1]   # num of cols
    
    M = np.zeros([N,N])
    for i in range(N):
        for j in range(N): 
            xi = X[i,:]
            xj = X[j,:]
         
            #square_length_xi = sum([x**2 for x in xi])
            #square_length_xj = sum([x**2 for x in xj])
            #dot_product = sum([xi[k] * xj[k] for k in range(D)])
            dist = np.sqrt(np.sum((X[i, :] - X[j,:] )**2) )
            #  dist = math.sqrt(max(square_length_xi -2*dot_product + square_length_xj, 0))

            M[i,j] = dist
            
    return M



# second function to fill, compute distance matrix without loops
def compute_distance_smart(X):
    N = X.shape[0]  # num of rows
    D = X.shape[1]  # num of cols
    
    # use X to create M
     
    norm_squared = np.sum(X**2,axis=1)
    dot_product = np.dot(X, X.T)
    M = np.sqrt(np.clip(norm_squared[:,np.newaxis] + norm_squared[np.newaxis,:] -2*dot_product, 0, None))
    
    return M


'''
What is a correlation matrix?

https://www.questionpro.com/blog/correlation-matrix/

A covarian matrix can help identify hidden relationships in a dataset.

It displays correlation coefficients that meassyre the strength and direction of relationships between variables.

Ranges (-1, +1) -> 1 means a perfect negative correlation, 1 means a perfect positive correlation and 0 means there is zero correlation
between the variables.

It can be useful to identify variables that are correlated which are useful for a ML model. 

'''

# third function to fill, compute correlation matrix using loops
def compute_correlation_naive(X):
    N = X.shape[0]  # num of rows
    D = X.shape[1]  # num of cols

    # Calculate the median along the columns
    mean = np.mean(X, axis=0)

    # Calculate the standard deviation along the columns 
    std = np.std( X, axis=0)
   
    M = np.zeros([D, D])
 
    for i in range(D):
        for j in range(D):
            xi = X[:, i]
            xj = X[:, j]
            covariance = np.sum((xi - mean[i])*(xj - mean[j]))/(N-1)
            corr = covariance/ (std[i]*std[j])
            M[i, j] = corr

    return M

# fourth function to fill, compute correlation matrix without loops
def compute_correlation_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    # use X to create M
    
    mean = np.mean(X, axis=0)
    std =np.std(X, axis=0)
    R = np.outer(std,std)
    X_centered = X-mean 
    S = (X_centered.T @ X_centered) / (N-1)
    M = S/R

    return M

def main():
    print('starting comparing distance computation .....')
    np.random.seed(100)
    params = range(10,141,10)   # different param setting
    nparams = len(params)       # number of different parameters

    perf_dist_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_dist_cool = np.zeros([10,nparams])
    perf_corr_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_corr_cool = np.zeros([10,nparams])

    counter = 0

    for ncols in params:
        nrows = ncols * 10

        print("matrix dimensions: ", nrows, ncols)

        for i in range(10):
            X = np.random.rand(nrows, ncols)   # random matrix

            # compute distance matrices
            st = time.time()
            dist_loop = compute_distance_naive(X)
            et = time.time()
            perf_dist_loop[i,counter] = et - st              # time difference

            st = time.time()
            dist_cool = compute_distance_smart(X)
            et = time.time()
            perf_dist_cool[i,counter] = et - st

            assert np.allclose(dist_loop, dist_cool, atol=1e-06) # check if the two computed matrices are identical all the time

            # compute correlation matrices
            st = time.time()
            corr_loop = compute_correlation_naive(X)
            et = time.time()
            perf_corr_loop[i,counter] = et - st              # time difference

            st = time.time()
            corr_cool = compute_correlation_smart(X)
            et = time.time()
            perf_corr_cool[i,counter] = et - st

            assert np.allclose(corr_loop, corr_cool, atol=1e-06) # check if the two computed matrices are identical all the time

        counter = counter + 1

    mean_dist_loop = np.mean(perf_dist_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_dist_cool = np.mean(perf_dist_cool, axis = 0)
    std_dist_loop = np.std(perf_dist_loop, axis = 0)      # standard deviation
    std_dist_cool = np.std(perf_dist_cool, axis = 0)

    plt.figure(1)
    plt.errorbar(params, mean_dist_loop[0:nparams], yerr=std_dist_loop[0:nparams], color='red',label = 'Loop Solution for Distance Comp')
    plt.errorbar(params, mean_dist_cool[0:nparams], yerr=std_dist_cool[0:nparams], color='blue', label = 'Matrix Solution for Distance Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Distance Computation Methods')
    plt.legend()
    plt.savefig('CompareDistanceCompFig.pdf')
    # plt.show()    # uncomment this if you want to see it right way
    print("result is written to CompareDistanceCompFig.pdf")

    mean_corr_loop = np.mean(perf_corr_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_corr_cool = np.mean(perf_corr_cool, axis = 0)
    std_corr_loop = np.std(perf_corr_loop, axis = 0)      # standard deviation
    std_corr_cool = np.std(perf_corr_cool, axis = 0)

    plt.figure(2)
    plt.errorbar(params, mean_corr_loop[0:nparams], yerr=std_corr_loop[0:nparams], color='red',label = 'Loop Solution for Correlation Comp')
    plt.errorbar(params, mean_corr_cool[0:nparams], yerr=std_corr_cool[0:nparams], color='blue', label = 'Matrix Solution for Correlation Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Correlation Computation Methods')
    plt.legend()
    plt.savefig('CompareCorrelationCompFig.pdf')
    # plt.show()    # uncomment this if you want to see it right way
    print("result is written to CompareCorrelationCompFig.pdf")

if __name__ == "__main__": main()
