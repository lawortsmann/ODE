# -*- coding: utf-8 -*-
# RKF45.py
"""
Solve ODE's numerically via adaptive RKF45.

    x'(t) = f(t, x(t))

Manages step size based on error estimate.

@version: 04.24.2016
@author: Luke_Wortsmann
"""
import numpy as np
from scipy.interpolate import UnivariateSpline


class RKF45Error(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


B0 = np.array([1. / 4., 0., 0., 0., 0., 0.])
B1 = np.array([3. / 32., 9. / 32., 0., 0., 0., 0.])
B2 = np.array([1932. / 2197., -7200. / 2197., 7296. / 2197., 0., 0., 0.])
B3 = np.array([439. / 216., -8., 3680. / 513., - 845. / 4104., 0., 0.])
B4 = np.array([-8. / 27., 2., -3544. / 2565., -1859. / 4104., -11. / 40., 0.])
B5 = np.array([25. / 216., 0., 1408. / 2565., 2197. / 4104., -1. / 5., 0.])
B6 = np.array([16. / 135., 0., 6656. / 12825., 28561. / 56430., -9. / 50., 2. / 55.])


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
    K = np.zeros((6,) + x.shape, dtype=x.dtype)
    K[0] = f(t, x)

    y2 = x + h * B0.dot(K)
    K[1] = f(t + (1./4.) * h, y2)

    y3 = x + h * B1.dot(K)
    K[2] = f(t + (3./8.) * h, y3)

    y4 = x + h * B2.dot(K)
    K[3] = f(t + (12./13.) * h, y4)

    y5 = x + h * B3.dot(K)
    K[4] = f(t + h, y5)

    y6 = x + h * B4.dot(K)
    K[5] = f(t + (1./2.) * h, y6)

    # Update state:
    xn = x + h * B5.dot(K)

    # Error estimate:
    xmag = np.abs(xn)
    xerr = np.abs( h * (B6 - B5).dot(K) )

    xerr[xerr == 0.0] = 1.
    xmag[xmag == 0.0] = xerr[xmag == 0.0]
    e = np.max( xerr / xmag )

    return t + h, xn, e


def RKF45(f, x0, stop, ti=0, fargs=(), fkwargs=dict(), h0=1e-6, tol=1e-6, maxSteps=1e6, warn=True):
    """
    Implimentation of the Runge–Kutta–Fehlberg method or RKF45 for solving ODE's
    numerically.

    Parameters
    ----------
    f    : function
           ODE functional, x'(t) = f(t, x(t))
    x0   : number or np.ndarray
           Initial ODE state, x(ti) = x0
    stop : function
           Integrate until stopf(t, x(t)) is True

    Keyword Arguments
    -----------------
    ti       : number, 0
               Inital time of the ODE, corresponding to state x0
    fargs    : tuple, None
               Additional arguments for the ODE functional
    fkwargs  : dict, None
               Keyword arguments for the ODE functional
    h0       : number, 1e-6
               Error tolerance
    tol      : number, 1e-6
               Error tolerance
    maxSteps : number, 1e6
               Maximum number of steps to take
    warn     : bool, True
               Raise RKF45Error if failed to stop

    Returns
    -------
    tf : number
         Stopping time
    x  : dict of scipy.interpolate.InterpolatedUnivariateSpline
         Interpolated solution to the ODE, call x[n](t) for ti <= t <= tf for
         numerical solution to the nth varible of the ODE.
    """
    # Initialize varibles:
    x0, ti = np.array(x0), float(ti)
    X, T, E = [x0], [ti], []
    f_eval = lambda t, x: f(t, x, *fargs, **fkwargs)
    h, maxSteps = float(h0), int(maxSteps)
    warn, stopping = bool(warn), False

    try:
        stop(ti, x0)
    except TypeError:
        tf = float(stop)
        stop = lambda t, x, *fargs, **fkwargs: t > tf

    for i in range(maxSteps):
        t_n, x_n, e_n = RKF45_step(f_eval, T[-1], X[-1], h)

        if stop(t_n, x_n, *fargs, **fkwargs):
            stopping = True

        elif e_n < tol:
            T.append( t_n )
            X.append( x_n )
            E.append( e_n )

        if stopping:
            if (h / t_n) < tol:
                warn = False
                break
            else:
                h *= 0.5

        elif e_n < tol:
            h *= 2

        else:
            h *= 0.5

        if h / t_n == 0.0:
            # stiff solution
            warn = False
            break

    T, X, E = np.array(T), np.array(X), np.array(E)

    if warn:
        raise RKF45Error( 'Did not stop after %s steps'%maxSteps )

    nV = X.shape[1]
    Xspl = dict()
    for j in range(nV):
        reQ = np.all(np.imag(X[:, j]) == 0.)
        if reQ:
            Xspl[j] = UnivariateSpline(T, np.real(X[:, j]), k=3, ext=3, s=0)
        else:
            Xspl[str(j) + ' RE'] = UnivariateSpline(T, np.real(X[:, j]), k=3, ext=3, s=0)
            Xspl[str(j) + ' IM'] = UnivariateSpline(T, np.imag(X[:, j]), k=3, ext=3, s=0)

    return T[-1], Xspl
