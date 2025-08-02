def rk4(x, t, tau, derivsRK, param):
    '''4-th order Runge-Kutta Integrator

    Input args:
     x = current value of dependent variable 
     t = independent variable (typically time)
     tau = Step size (ie. timestep)
     derivsRK = right hand side of the ODE; name of function
     which returns dx/dt, derivsRk(x,t,param)

    Output args:
     xout = new value of x after step tau
    '''

    half_tau = 0.5 * tau
    F1 = derivsRK(x, t, param)
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp, t_half, param)
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp, t_half, param)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp, t_full, param)
    xout = x + 1/6 * tau * (F1 + 2*F2 + 2*F3 + F4)
    return xout
    
    
