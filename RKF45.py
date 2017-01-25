# -*- coding: utf-8 -*-
# RKF45.py
"""
Solve ODE's numerically via adaptive RKF45.

    x'(t) = f(t, x(t))

Manages step size based on error estimate.

@version: 10.12.2016
@author: Luke_Wortsmann
"""
import numpy as np
from numba import jit, autojit
from scipy.interpolate import interp1d


@jit
def RKF45_step(f, t, x, h):
    """
    A evaluation step of RKF45, performs 6 function evaluations. Step accuracy
    is O(h^4) and error accuracy is O(h^5).

    Parameters
    ----------
    f : function
        ODE functional, x'(t) = f(t, x(t))
    t : number
        ODE current step time
    x : number or np.ndarray
        ODE current state
    h : number
        step size

    Returns
    -------
    tn : float
         Next ODE time, t + h
    xn : float or np.ndarray
         Next ODE state, x(t + h)
    e  : float
         Error estimate of next ODE state
    """
    k1 = f(t, x)
    y2 = x + (h / 4.) * k1
    k2 = f(t + (1./4.) * h, y2)
    y3 = x + h * ((3. / 32.) * k1 + (9. / 32.) * k2)
    k3 = f(t + (3./8.) * h, y3)
    y4 = x + h * ((1932. / 2197.) * k1 - (7200. / 2197.) * k2 + (7296. / 2197.) * k3)
    k4 = f(t + (12./13.) * h, y4)
    y5 = x + h * ((439. / 216.) * k1 + (-8.) * k2 + (3680. / 513.) * k3 - (845. / 4104.) * k4)
    k5 = f(t + h, y5)
    y6 = x + h * (-(8. / 27.) * k1 + (2.) * k2 - (3544. / 2565.) * k3
                   - (1859. / 4104.) * k4 - (11. / 40.) * k5)
    k6 = f(t + (1./2.) * h, y6)
    b5 = np.array([16. / 135., 0., 6656. / 12825., 28561. / 56430., -9. / 50., 2. / 55.])
    b4 = np.array([25. / 216., 0., 1408. / 2565., 2197. / 4104., -1. / 5., 0.])
    K =  np.array([k1, k2, k3, k4, k5, k6])
    # Update state:
    xn = x + h * (b4.dot(K))
    # Error estimate:
    e = np.linalg.norm(abs( (h * (b5 - b4).dot(K)) / (abs(xn) + 1e-15)))
    return t + h, xn, e


def RKF45(f, x0, tf, ti=0, fkwargs=dict(), tol=1e-5, max_steps=1e6, interpolation='linear'):
    """
    Implimentation of the Runge–Kutta–Fehlberg method or RKF45 for solving ODE's
    numerically.

    Parameters
    ----------
    f  : function
         ODE functional, x'(t) = f(t, x(t), **fkwargs)
    x0 : number or np.ndarray
         Initial ODE state, x(ti) = x0
    tf : number
         Final time, ODE time to solve until

    Keyword Arguments
    -----------------
    ti  : number, 0
          Inital time of the ODE, corresponding to state x0
    fkwargs: dict, None
          Keyword arguments for the ODE functional
    tol : number, 1e-5
          Error tolerance
    max_steps : number, 1e6
          Maximum number of steps to take
    interpolation : str or int, 'linear'
          Interpolation kind, see scipy.interpolate.interp1d

    Returns
    -------
    x : function
        Interpolated solution to the ODE, call x(t) for ti <= t <= tf for
        numerical solution to the ODE.
    """
    # Minimum stepsize:
    min_h = (tf - ti) / max_steps
    # Initialize varibles:
    T, E = np.zeros((2, 2 * int(max_steps)))
    X = np.zeros((2 * int(max_steps),) + np.array(x0).shape)
    T[0], X[0], i, h = ti, x0, 0, min_h
    # Add kwargs to function:
    f_eval = lambda t, x: f(t, x, **fkwargs)
    while True:
        # Get update for stepsize:
        t_update, y_update, error = RKF45_step(f_eval, T[i], X[i], h)
        if (error < tol) or (h < min_h):
            # Keep if error acceptable or smaller than minimum stepsize
            i += 1
            h *= 2
            T[i] = t_update
            X[i] = y_update
            E[i] = error
            if t_update >= tf:
                # Done if past tf
                break
        else:
            # Otherwise decrease stepsize
            h *= 0.5
    # Return interpolation function from calculated points:
    return interp1d(T[:i + 1], X[:i + 1], kind=interpolation, axis=0, assume_sorted=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f = lambda t, x: np.exp(1j * t * np.cos(x)**2) - x
    g = RKF45(f, 0, 1)
    t = np.linspace(0, 1, 1000)
    plt.plot(t, g(t))
    plt.show()
