# -*- coding: utf-8 -*-
# Roots.py
"""
Find roots of a polynomial equation.

@version: 10.17.2016
@author: Luke_Wortsmann
"""
import numpy as np
from numba import jit


@jit(["complex128[:](complex128[:])"])
def roots(coef):
    """
    Find all roots of the polynomial:
        Sum_i (coef[i] * x**i)

    Parameters
    ----------
    coef: np.ndarray
          List of coefficients of the polynomial

    Returns
    -------
    roots: np.ndarray
           List of the roots of the polynomial
    """
    A = np.diag(np.ones(len(coef) - 2, dtype=np.complex128), -1)
    A[0] = - (coef[:-1][::-1] / np.complex128(coef[-1]))
    return np.linalg.eigvals(A)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from itertools import product
    from multiprocessing import Pool

    def coef_generator(n=20):
        for c in product([-1, 1], repeat=n):
            yield np.concatenate([[1], c])

    # Plot roots of a polynomial with coefficients = {-1 or +1}
    p = Pool()
    _roots  = np.array(p.map(roots, coef_generator())).flatten()
    H, _, _ = np.histogram2d(_roots.real, _roots.imag, bins=(4000, 3000))
    # This will throw a warning but its fine
    plt.imsave('roots.png', np.log(H.T))
