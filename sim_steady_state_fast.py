"""
Slightly more advanced code for standard incomplete markets
model, speeding up sim_steady_state.py.

Built up in "Lecture 1a, Speeding Up the Steady State.ipynb".

See SPEEDUP comments for specific improvements.

Note that some functions have not been changed from their
previous source code in sim_steady_state.py, but since they
are calling functions that have been sped up here, we need
to define them again. These are marked with # NO CHANGE.
"""

import numpy as np
import numba

# little need to speed up these functions
from sim_steady_state import (discretize_assets, rouwenhorst_Pi, forward_iteration,
                              expectation_iteration, expectation_functions)

"""Part 0: example calibration from notebook"""

# NO CHANGE
def example_calibration():
    y, _, Pi = discretize_income(0.975, 0.7, 7) 
    return dict(a_grid = discretize_assets(0, 10_000, 500),
                y=y, Pi=Pi,
                r = 0.01/4, beta=1-0.08/4, eis=1)


"""Part 1: discretization tools"""

# NO CHANGE
def discretize_income(rho, sigma, n_e):
    # choose inner-switching probability p to match persistence rho
    p = (1+rho)/2
    
    # start with states from 0 to n_e-1, scale by alpha to match standard deviation sigma
    e = np.arange(n_e)
    alpha = 2*sigma/np.sqrt(n_e-1)
    e = alpha*e
    
    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_e, p)
    pi = stationary_markov(Pi)
    
    # e is log income, get income y and scale so that mean is 1
    y = np.exp(e)
    y /= np.vdot(pi, y)
    
    return y, pi, Pi


"""Support for part 1: equality testing and Markov chain convergence"""

@numba.njit
def equal_tolerance(x1, x2, tol):
    # "ravel" flattens both x1 and x2, without making copies, so we can compare the
    # with a single for loop even if they have multiple dimensions
    x1 = x1.ravel()
    x2 = x2.ravel()

    # iterate over elements and stop immediately if any diff by more than tol
    for i in range(len(x1)):
        if np.abs(x1[i] - x2[i]) >= tol:
            return False
    return True


@numba.njit
def stationary_markov(Pi, tol=1E-14):
    # start with uniform distribution over all states
    n = Pi.shape[0]
    pi = np.full(n, 1/n)
    
    # update distribution using Pi until successive iterations differ by less than tol
    for it in range(10_000):
        pi_new = Pi.T @ pi
        # SPEEDUP: only test for convergence every 10 iterations
        # SPEEDUP: use equal_tolerance rather than inefficient NumPy test
        if it % 10 == 0 and equal_tolerance(pi, pi_new, tol):
            return pi_new
        pi = pi_new


"""Part 2: Backward iteration for policy"""

def backward_iteration(Va, Pi, a_grid, y, r, beta, eis):
    # step 1: discounting and expectations
    Wa = beta * Pi @ Va
    
    # step 2: solving for asset policy using the first-order condition
    c_endog = Wa**(-eis)
    coh = y[:, np.newaxis] + (1+r)*a_grid
    # SPEEDUP: interpolation exploits monotonicity and avoids pure-Python for loop
    a = interpolate_monotonic_loop(coh, c_endog + a_grid, a_grid)
        
    # step 3: enforcing the borrowing constraint and backing out consumption
    # SPEEDUP: enforce constraint in place, without creating new arrays, stop when unnecessary
    setmin(a, a_grid[0])
    c = coh - a
    
    # step 4: using the envelope condition to recover the derivative of the value function
    Va = (1+r) * c**(-1/eis)
    
    return Va, a, c


def policy_ss(Pi, a_grid, y, r, beta, eis, tol=1E-9):
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = y[:, np.newaxis] + (1+r)*a_grid
    c = 0.05 * coh
    Va = (1+r) * c**(-1/eis)
    
    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    for it in range(10_000):
        Va, a, c = backward_iteration(Va, Pi, a_grid, y, r, beta, eis)
        
        # after iteration 0, can compare new policy function to old one
        # SPEEDUP: only test for convergence every 10 iterations
        # SPEEDUP: use equal_tolerance rather than inefficient NumPy test
        if it % 10 == 1 and equal_tolerance(a, a_old, tol):
            return Va, a, c
        
        a_old = a


"""Support for part 2: equality testing and Markov chain convergence"""

@numba.njit
def interpolate_monotonic(x, xp, yp):
    """Linearly interpolate the data points (xp, yp) and evaluate at x, with both x and xp monotonic"""
    nx, nxp = x.shape[0], xp.shape[0]
    y = np.empty(nx)
    
    # at any given moment, we are looking between data points with indices xp_i and xp_i+1
    # we'll keep track of xp_i and also the values of xp at xp_i and xp_i+1
    # (note: if x is outside range of xp, we'll use closest data points and extrapolate)
    xp_i = 0
    xp_lo = xp[xp_i]
    xp_hi = xp[xp_i + 1]
    
    # iterate through all points in x
    for xi_cur in range(nx):
        x_cur = x[xi_cur]
        while xp_i < nxp - 2:
            # if current x (x_cur) is below upper data point (xp_hi), we're good
            if x_cur < xp_hi:
                break
                
            # otherwise, we need to look at the next pair of data points until x_cur is less than xp_hi
            # (so between xp_lo and xp_hi)
            xp_i += 1
            xp_lo = xp_hi
            xp_hi = xp[xp_i + 1]

        # find the pi such that x_cur = pi*x_lo + (1-pi)*x_hi
        pi = (xp_hi - x_cur) / (xp_hi - xp_lo)
        
        # use this pi to interpolate the y
        y[xi_cur] = pi * yp[xp_i] + (1 - pi) * yp[xp_i + 1]
    return y


@numba.njit
def interpolate_monotonic_loop(x, xp, yp):
    ne = x.shape[0]
    y = np.empty_like(x)
    for e in range(ne):
        y[e, :] = interpolate_monotonic(x[e, :], xp[e, :], yp)
    return y


@numba.njit
def setmin(x, xmin):
    """Set 2-dimensional array x, where each row is ascending, equal to max(x, xmin)."""
    ni, nj = x.shape
    for i in range(ni):
        for j in range(nj):
            if x[i, j] < xmin:
                # if below minimum, enforce minimum in place
                x[i, j] = xmin
            else:
                # SPEEDUP: otherwise, do nothing, and skip to next row (thanks to monotonicity)
                break


"""Part 3: forward iteration for distribution"""

@numba.njit
def interpolate_lottery(x, xp):
    """Given a grid of xp, for each entry x_cur in (increasing) x, find the i and pi
    such that x_cur = pi*xp[i] + (1-pi)*xp[i+1], where xp[i] and xp[i+1] bracket x_cur"""
    nx, nxp = x.shape[0], xp.shape[0]
    i = np.empty(nx, dtype=np.int64)
    pi = np.empty(nx)
    
    # at any given moment, we are looking between data points with indices xp_i and xp_i+1
    # we'll keep track of xp_i and also the values of xp at xp_i and xp_i+1
    # (note: if x is outside range of xp, we'll use closest data points and extrapolate)
    xp_i = 0
    xp_lo = xp[xp_i]
    xp_hi = xp[xp_i + 1]
    
    # iterate through all points in x
    for xi_cur in range(nx):
        x_cur = x[xi_cur]
        while xp_i < nxp - 2:
            # if current x (x_cur) is below upper data point (xp_hi), we're good
            if x_cur < xp_hi:
                break
                
            # otherwise, we need to look at the next pair obf data points until x_cur is less than xp_hi
            # (so between xp_lo and xp_hi)
            xp_i += 1
            xp_lo = xp_hi
            xp_hi = xp[xp_i + 1]

        # find the pi such that x_cur = pi*x_lo + (1-pi)*x_hi
        i[xi_cur] = xp_i
        pi[xi_cur] = (xp_hi - x_cur) / (xp_hi - xp_lo)
    return i, pi


@numba.njit
def interpolate_lottery_loop(x, xp):
    i = np.empty_like(x, dtype=np.int64)
    pi = np.empty_like(x)
    for e in range(x.shape[0]):
        i[e, :], pi[e, :] = interpolate_lottery(x[e, :], xp)
    return i, pi


def distribution_ss(Pi, a, a_grid, tol=1E-10):
    a_i, a_pi = interpolate_lottery_loop(a, a_grid)
    
    # as initial D, use stationary distribution for s, plus uniform over a
    pi = stationary_markov(Pi)
    D = pi[:, np.newaxis] * np.ones_like(a_grid) / len(a_grid)
    
    # now iterate until convergence to acceptable threshold
    for it in range(10_000):
        D_new = forward_iteration(D, Pi, a_i, a_pi)
        # SPEEDUP: only test for convergence every 10 iterations
        # SPEEDUP: use equal_tolerance rather than inefficient NumPy test
        if it % 10 == 0 and equal_tolerance(D_new, D, tol):
            return D_new
        D = D_new


"""Part 4: solving for steady state, including aggregates"""

# ALMOST NO CHANGE (calling interpolate_lottery_loop instead of get_lottery)
def steady_state(Pi, a_grid, y, r, beta, eis):
    Va, a, c = policy_ss(Pi, a_grid, y, r, beta, eis)
    D = distribution_ss(Pi, a, a_grid)
    a_i, a_pi = interpolate_lottery_loop(a, a_grid)
    
    return dict(D=D, Va=Va, 
                a=a, c=c, a_i=a_i, a_pi=a_pi,
                A=np.vdot(a, D), C=np.vdot(c, D),
                Pi=Pi, a_grid=a_grid, y=y, r=r, beta=beta, eis=eis)