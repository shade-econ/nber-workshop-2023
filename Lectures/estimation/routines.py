"""Core utilities for simulation, second moments, and estimation"""

import numpy as np
import sequence_jacobian as sj
import numba

"""Simulation"""

def simulate(impulses, outputs, T_sim):
    """
    impulses: list of ImpulseDicts, each an impulse to independent unit normal shock
    outputs: list of outputs we want in simulation
    T_sim: length of simulation

    simulation: dict mapping each output to length-T_sim simulated series
    """

    simulation = {}
    epsilons = [np.random.randn(T_sim + impulses[0].T - 1) for _ in impulses]
    for o in outputs:
        simulation[o] = sum(simul_shock(imp[o], eps) 
                            for imp, eps in zip(impulses, epsilons))
        
    return simulation

@numba.njit(parallel=True)
def simul_shock(dX, epsilons):
    """Take in any impulse response dX to epsilon shock, plus path of epsilons, and simulate"""    
    # if I have T_eps epsilons, can simulate length T_eps - T + 1 dXtildes
    # by indexing as eps_(-T-1), ... , eps_(T_eps-T+1) and implementing formula
    T = len(dX)
    T_eps = len(epsilons)
    dXtilde = np.empty(T_eps - T + 1) 
    
    dX_flipped = dX[::-1].copy() # flip because dX_s multiplies eps_(t-s)
    for t in numba.prange(T_eps - T + 1):
        dXtilde[t] = np.vdot(dX_flipped, epsilons[t:t + T]) # sum as single dot product

    return dXtilde


"""Log-likelihood of priors"""

def log_priors(thetas, priors_list):
    """Given a vector 'thetas', where entry i is drawn from the prior
    distribution specified in entry i of priors_list, calculate sum of
    log prior likelihoods of each theta. Distributions over theta should be specified 
    in the same way that arguments are given to the 'log_prior' function: first the
    name of the family, and then two parameters"""
    return sum(log_prior(theta, *prior) for theta, prior in zip(thetas, priors_list))


def log_prior(theta, dist, arg1, arg2):
    """Calculate log prior probability of 'theta', if prior is from family
    'dist' with parameters 'arg1' and 'arg2' (depends on prior)"""
    if dist == 'Normal':
        mu = arg1
        sigma = arg2
        return - 0.5 * ((theta - mu)/sigma)**2
    elif dist == 'Uniform':
        lb = arg1
        ub = arg2
        return - np.log(ub-lb)
    elif dist == 'Invgamma':
        s = arg1
        v = arg2
        return (-v-1) * np.log(theta) - v*s**2/(2*theta**2)
    elif dist == 'Gamma':
        theta = arg2**2 / arg1
        k = arg1 / theta
        return (k-1) * np.log(theta) - theta/theta
    elif dist == 'Beta':
        alpha = (arg1*(1 - arg1) - arg2**2) / (arg2**2 / arg1)
        beta = alpha / arg1 - alpha
        return (alpha-1) * np.log(theta) + (beta-1) * np.log(1-theta)
    else:
        raise ValueError('Distribution provided is not implemented in log_prior!')


def construct_stacked_A(As, To, To_out=None, sigma_e=None, sigma_o=None, reshape=True, long=False):
    Tm, O, E = As.shape

    # how long should the IRFs be that we stack in A_full?
    if To_out is None:
        To_out = To
    if long:
        To_out = To + Tm  # store even the last shock's IRF in full!

    # allocate memory for A_full
    A_full = np.zeros((To_out, O, To, E))

    for o in range(O):
        for itshock in range(To):
            # if To > To_out, allow the first To - To_out shocks to happen before the To_out time periods
            if To <= To_out:
                iA_full = itshock
                iAs = 0

                shock_length = min(Tm, To_out - iA_full)
            else:
                # this would be the correct start time of the shock
                iA_full = itshock - (To - To_out)

                # since it can be negative, only start IRFs at later date
                iAs = - min(iA_full, 0)

                # correct iA_full by that date
                iA_full += - min(iA_full, 0)

                shock_length = min(Tm, To_out - iA_full)

            for e in range(E):
                A_full[iA_full:iA_full + shock_length, o, itshock, e] = As[iAs:iAs + shock_length, o, e]
                if sigma_e is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] *= sigma_e[e]
                if sigma_o is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] /= sigma_o[o]
    if reshape:
        A_full = A_full.reshape((To_out * O, To * E))
    return A_full
