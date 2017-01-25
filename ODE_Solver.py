# -*- coding: utf-8 -*-
# ODE_Solver.py
"""
Analytically solve linear ODE's of the form:

    x'(t) = A.x(t) + f(t)

Or a linear nonhomogeneous constant coefficients ODE. For some n x n matrix A,
some f(t), and an intial point x(t0) = x0. Solution in general form:

    x(t) = \exp(A t).( c + \int_{t0}^{t} \exp(-A k).f(k) dk )

Where c is such that x(t0) = x0 and exp is the matrix exponential. Efficiently
    recompute x0 and f with set_x0 and set_f methods. 50x faster than
    scipy.linalg.expm in nondeficient case, 5x faster in deficient case.

To do: nonconstant A.

@version: 12.10.2015
@author: Luke_Wortsmann
"""
import numpy as np
import scipy.linalg as sci_lin

class linear_ODE:
    def __init__(self, A, x0 = None, f = None, t0 = 0., check = True):
        self.t0 = t0
        self.tol = 1e-15
        self.use_expm = False
        self.n = len(A)
        self.A = A
        self.error = False
        self.eigenvals,self.eigenvects = np.linalg.eig(self.A)
        self.deficient = (abs(np.linalg.det(self.eigenvects)) < self.tol)
        if self.deficient == False:
            self.v_LUP = sci_lin.lu_factor(self.eigenvects, check_finite = False)
        else:
            self._set_matrix_exp()
        if type(x0) == type(None):
            x0 = np.zeros(self.n)
        if type(f) == type(None):
            f = np.zeros(self.n)
        self.set_x0(x0, t0, f)
        if check:
            errtol = self.test()
            if errtol > 1e-5:
                print 'Error greater than tolerance'
                print 'Will use scipy.linalg.expm'
                self.error = True
                self.use_expm = True

    def test(self, bounds = 2., points = 25):
        err = 0.
        for t in np.linspace(self.t0 - bounds, self.t0 + bounds, points):
            s1 = sci_lin.expm(self.A*t).dot(self.x0)
            if self.deficient:
                s2 = self._eval_matrix_exp(t).dot(self.x0)
            else:
                s2 = self.matrix_exp_eig(self.x0,t)
            err += np.linalg.norm(2*abs(s1 - s2)/abs(s1 + s2))
        return err/points

    def set_x0(self, x0, t0 = 0., f = None):
        self.t0 = t0
        self.x0 = x0
        self.prev_int = (t0, 0.)
        if type(f) == type(None):
            f = self.f
        self.set_f(f)
        self.get_c()
        if self.homogeneous and (self.deficient == False):
            coeff = sci_lin.lu_solve(self.v_LUP, self.c, check_finite = False)
            self.homogeneous_C = self.eigenvects * coeff

    def set_f(self, f):
        self.function_f = (type(f) == type(lambda x: x))
        if self.function_f == False:
            self.homogeneous = (f.dot(f) == 0.)
            self.f = f
            if abs(np.linalg.det(self.A)) < 1e-15:
                self.function_f = True
                self.f = lambda t: f
            else:
                self.A_LUP = sci_lin.lu_factor(self.A, check_finite = False)
                self.constant = (self.A).dot(self.x0)
        else:
            self.homogeneous = False
            self.f = f

    def get_c(self):
        if self.t0 == 0.:
            self.c = self.x0

        elif self.deficient == False:
            eig_exp = np.exp(self.eigenvals * self.t0)
            kappa = (self.eigenvects)*eig_exp
            coeff = np.linalg.solve(kappa,self.x0)
            self.c = self.eigenvects.dot(coeff)

        elif self.deficient:
            self.c = np.linalg.solve(self._eval_matrix_exp(self.t0), self.x0)

    def _eval(self, t):
        if self.homogeneous:
            if self.deficient == False:
                m_power = np.exp(self.eigenvals * t)
                return (self.homogeneous_C).dot(m_power)
            if self.deficient:
                return self._eval_matrix_exp(t).dot(self.c)
        else:
            if self.function_f:
                net = self.c + self.integrate(t)
                if self.deficient == False:
                    return self.matrix_exp_eig(net, t)
                if self.deficient:
                    return self._eval_matrix_exp(t).dot(net)
            else:
                net = self.f + self.constant
                if self.deficient == False:
                    K = self.matrix_exp_eig(net, t - self.t0) - self.f
                    return sci_lin.lu_solve(self.A_LUP, K, check_finite = False)
                if self.deficient:
                    K = self._eval_matrix_exp(t - self.t0).dot(net)  - self.f
                    return sci_lin.lu_solve(self.A_LUP, K, check_finite = False)

    def matrix_exp_eig(self, v, t):
        if self.use_expm == False:
            eig_exp = np.exp(self.eigenvals * t)
            coeff = sci_lin.lu_solve(self.v_LUP, v, check_finite = False)
            return ((self.eigenvects).dot(eig_exp*coeff))
        else:
            return sci_lin.expm(self.A*t)

    def _eval_matrix_exp(self, t):
        if self.use_expm == False:
            out = self.exp_A_set[0].copy()
            for i in xrange(1,len(self.exp_A_set)):
                M = self.exp_A_set[i]*((t-self.t0)**float(i))
                out += M
            return out
        else:
            return sci_lin.expm(self.A*t)

    def _set_matrix_exp(self, max_deg = 1000, tol = 1e-10):
        exp_A0 = sci_lin.expm(self.A*self.t0)
        im = 0j*self.A.copy()
        n_a = 1.*self.A.copy() + im
        self.exp_A_set = [exp_A0+im, n_a]
        for i in xrange(2, max_deg + 1):
            n_a = n_a.dot(self.A)
            fact = np.math.factorial(i)
            M = (exp_A0.dot(n_a))/fact
            self.exp_A_set.append(M)
            if max(sum(abs(M))) < tol:
                return
        self.error = True
        print 'Deficient Matrix Error'
        print 'Will use scipy.linalg.expm'
        self.exp_A_set = []
        self.use_expm = True


    def integrate(self, t):
        pt,pi = self.prev_int
        if pt == t:
            return pi
        def f_eval(t):
            if self.deficient == False:
                return self.matrix_exp_eig(self.f(t), -t)
            if self.deficient:
                return self._eval_matrix_exp(-t).dot(self.f(t))
        ni = self._quad(f_eval, pt, t) + pi
        self.prev_int = (t, ni)
        return ni

    def _quad(self, f, lb, ub, deg = 4):
        def h_n(n,a,b):
            return (2.**-n)*(b-a)
        def Romberg(f, n, m, a, b):
            if n == 0 and m == 0:
                return h_n(1,a,b)*(f(a)+f(b))
            elif m == 0:
                h = h_n(n,a,b)
                t = 0.
                for k in xrange(1, int(2**(n-1) + 1)):
                    t += f(a + (2*k-1)*h)
                return 0.5*Romberg(f,n-1,0,a,b) + h*t
            else:
                Rmm = Romberg(f,n,m-1,a,b)
                return Rmm + (1./((4.**m) - 1.))*(Rmm - Romberg(f,n-1,m-1,a,b))
        return Romberg(f,deg,deg,lb,ub)

    def __call__(self, t):
        if (type(t) == list) or (type(t) == np.ndarray):
            if self.homogeneous == False:
                self.prev_int = (self.t0, 0.)
            t.sort()
            return np.array([self._eval(t_i) for t_i in t])
        else:
            if self.homogeneous == False:
                self.prev_int = (self.t0, 0.)
            return self._eval(t)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    f = lambda t: np.array([0, 0, 0, -9.8])
    x0 = np.array([0, 5, 10, 5])
    X = linear_ODE(A, x0, f)
    T = np.linspace(0, 2, 1000)
    sln = X(T)
    plt.plot(sln[:,0], sln[:,2])
    plt.show()
