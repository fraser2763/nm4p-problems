import numpy as np

def pollsf(x, y, sigma, M):
    '''Performs polynomial least squares fit

    Inputs:
    x    -   Independent variable, points where y is measured \n
    y   -    Dependent variable, measured data \n
    sigma -  Estimated Error in Data \n
    M     -  Number of parameters curve is fit to \n

    Outputs: 
    a_fit    -   Curve fit coefficients \n
    sigma_a  -   Error bars in a values \n
    Y_x     -    Curve fit function \n
    chi_sqr -     Chi squared statistic \n
    '''
    # Form vector b and design matrix A
    N = len(x)
    b = np.empty(N)
    A = np.empty((N,M))
    for i in range(N):
        b[i] = y[i]/sigma[i]
        for j in range(M):
            A[i,j] = x[i]**j / sigma[i]

    # Correlation Matrix:
    C = np.linalg.inv(A.T @ A)

    # a coefficients:
    a_fit = C @ A.T @ b

    # Est error bars:
    sigma_a = np.empty(M)
    for j in range(M):
        sigma_a[j] = np.sqrt(C[j,j])

    # Evaluate Y(x_i)
    Y_x = np.zeros(N)
    chi_sqr = 0.
    for i in range(N):
        for j in range(M):
            Y_x[i] += a_fit[j] * x[i]**j
        chi_sqr += (Y_x[i] - y[i])**2 / sigma[i]**2
    
    return [a_fit, sigma_a, Y_x, chi_sqr]

    

