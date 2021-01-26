import numpy as np

def calc_dist_matrix_lc1(x, y, weighted=False):
    dist = lambda x,y : np.sum(np.square(x - y))
    n, m = len(x), len(y)
    w = 1
    if weighted:
        w = 2
    
    dist_matrix = np.zeros((n, m))
    for i in range(1, n):
        for j in range(1, m):
            cost = dist(x[i], y[j])
            dist_matrix[i, j] = np.min([cost + dist_matrix[i-1, j], 
                                        cost + dist_matrix[i, j-1], 
                                        w*cost + dist_matrix[i-1, j-1]])
    return dist_matrix


def calc_dist_matrix_lc2(x, y, weighted=False):
    dist = lambda x,y : np.sum(np.square(x - y))
    n, m = len(x), len(y)
    w = 1
    if weighted:
        w = 0.5
        
    dist_matrix = np.zeros((n, m))
    for i in range(1, n):
        for j in range(1, m):
            cost = dist(x[i], y[j])
            dist_matrix[i, j] = np.min([w*(cost + dist(x[i-1], y[j])) + dist_matrix[i-2,j-1], 
                                        w*(cost + dist(x[i], y[j-1])) + dist_matrix[i-1,j-2], 
                                        cost + dist_matrix[i-1, j-1]])
    return dist_matrix



def get_optimal_path(D):
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        ind = np.argmin((D[i-1, j-1], D[i - 1, j], D[i, j- 1]))
        if ind == 0:
            i -= 1
            j -= 1
        elif ind == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def DTW(feat1, feat2, lc_type='1', weighted=False):
    calc_dist_matrix = {
        '1': calc_dist_matrix_lc1,
        '2' : calc_dist_matrix_lc2
    }
    D = calc_dist_matrix[lc_type](x=feat1, y=feat2, weighted=weighted)
    path = get_optimal_path(D)
    return D, path