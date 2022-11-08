import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    centroids = X[np.random.choice(X.shape[0], size=k)]
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    distances = [((np.abs(X - c) ** p).sum(axis=1)) ** (1 / p) for c in centroids]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return np.array(distances)


def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    classes = np.zeros_like(X.shape[0])  # each pixel will be assigned its closest centroid

    for iteration in range(max_iter):
        # closest centroid for each pixel
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        new_centroids = np.array([X[classes == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
        

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None 
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    centroids=(X[np.random.choice(X.shape[0])]) # first uniform random centroid

    c = 1
    while len(centroids) < k:
        distances = lp_distance(X, centroids, p)#.flatten()
        min_dist = np.min(distances, axis=0)
        weights = min_dist ** 2
        probs = weights / np.sum(weights)
        new_centroid_index = np.random.choice(a=X.shape[0], p=probs)
        centroids = np.vstack((centroids, X[new_centroid_index]))

        c +=1
    
    classes = np.zeros_like(X.shape[0])  # each pixel will be assigned its closest centroid

    for iteration in range(max_iter):
        # closest centroid for each pixel
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        new_centroids = np.array([X[classes == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
         
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
