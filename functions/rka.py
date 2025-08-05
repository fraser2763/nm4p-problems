from nm4p.rk4 import rk4
import numpy as np

def rka(x, t, tau, err, derivsRK, param):
    '''Adaptive Runge-Kutta algorithm
    
        Inputs:
            x = current value of dependent variable 
            t = independent variable (typically time)
            tau = Step size (ie. timestep)
            err = desired fractional truncation error (as estimated by routine)
            derivsRK = right hand side of the ODE; name of function
                       which returns dx/dt, derivsRk(x,t,param)
            param = Extra parameters to be passed to derivsRK

        Outputs:
            xSmall = New value of the dependent variable (as calculated by the two half steps)
            t = New value opf the independent variable (eg time)
            tau = Suggested time step for next call to rka'''

    # set initial vars  
    tSave, xSave = t, x          # save initial values 
    safe1, safe2 = 0.9, 4.0      # safety factors
    eps = 1.e-15

    # Loop over max number of attempts to satisfy error bound
    maxTry = 100
    for i in range(maxTry):

        # Take the two small time steps
        half_tau = 0.5 * tau
        xTemp = rk4(xSave, tSave, half_tau, derivsRK, param)
        t = tSave + half_tau
        xSmall = rk4(xTemp, t, half_tau, derivsRK, param)

        # Take single big step
        t = tSave + tau
        xBig = rk4(xSave, t, tau, derivsRK, param)

        # Compute estimated truncation error
        scale = err * (abs(xSmall) + abs(xBig) /2)
        xDiff = xSmall - xBig 
        errorRatio = np.max(np.absolute(xDiff) / (scale + eps))

        # estimate new value with safety factors
        tau_old = tau
        tau = safe1 * tau_old * errorRatio**(-.2)
        tau = max(tau, tau_old/safe2)
        tau = min(tau, safe2*tau_old)

        # if error is acceptable, return computed values
        if errorRatio < 1:
            return xSmall, t, tau
        
    # Error message if error bound never satisfied
    print('ERROR: Adaptive RK routine failed')
    return xSmall, t, tau




    
