import numpy as np

def linreg(x, y, sigma):
    '''...'''
    # Evaluate various sigma sums
    S = 0.; sx = 0.; sy = 0.; sx_sqr = 0.; sxy = 0.
    for i in range(len(x)):
        S += 1/sigma[i]**2
        sx += x[i]/sigma[i]**2
        sy += y[i]/sigma[i]**2
        sx_sqr += x[i]**2/sigma[i]**2
        sxy += (x[i]*y[i])/sigma[i]**2
    
    # Evaluate a coefficients
    a_fit = np.empty(2)
    a_fit[0] = (sy * sx_sqr - sx * sxy) / (S * sx_sqr - sx**2)        # Curve y-intercept
    a_fit[1] = (S * sxy - sy * sx) / (S * sx_sqr - sx**2)             # Curve slope 

    # Error bars for each coeffcient
    sig_a = np.empty(2)
    sig_a[0] = np.sqrt(sx_sqr / (S * sx_sqr - sx**2))
    sig_a[1]= np.sqrt(S / (S * sx_sqr - sx**2))

    # Evaluate all Y_i and the chi-squared value
    Y_x = np.empty(len(x))
    chi_sqr = 0.
    for i in range(len(x)):
        Y_x[i] = a_fit[0] + a_fit[1]*x[i]
        chi_sqr += (Y_x[i] - y[i])**2 / sigma[i]**2
    
    return [a_fit, sig_a, Y_x, chi_sqr]



def linreg_no_err(x, y):
    '''...'''
    # Evaluate various sigma sums
    S = 0.; sx = 0.; sy = 0.; sx_sqr = 0.; sxy = 0.
    sigma = 1 
    for i in range(len(x)):
        S += 1/sigma**2
        sx += x[i]/sigma**2
        sy += y[i]/sigma**2
        sx_sqr += x[i]**2/sigma**2
        sxy += (x[i]*y[i])/sigma**2
    
    # Evaluate a coefficients
    a_fit = np.empty(2)
    a_fit[0] = (sy * sx_sqr - sx * sxy) / (S * sx_sqr - sx**2)        # Curve y-intercept
    a_fit[1] = (S * sxy - sy * sx) / (S * sx_sqr - sx**2)             # Curve slope 


    # Evaluate all Y_i and the chi-squared value
    Y_x = np.empty(len(x))
    chi_sqr = 0.
    for i in range(len(x)):
        Y_x[i] = a_fit[0] + a_fit[1]*x[i]
        chi_sqr += (Y_x[i] - y[i])**2 / sigma**2
    
    return [a_fit, Y_x, chi_sqr]