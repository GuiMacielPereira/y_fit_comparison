from iminuit_fit_helpers import oddPointsRes, save_result_of_global_fit
from scipy import signal, optimize
from mantid.simpleapi import *
from iminuit import Minuit, cost
from iminuit.util import describe
import numpy as np
import time

ws_resolution = Load("D_HMT_resolution.nxs")
ws_to_fit = Load("D_HMT_forward.nxs")

# Need to remove any Masked spectra on both workspaces
# In this case spectra idx 38 has very anomalous data, need to remove for good fit
ws_to_fit = RemoveSpectra(ws_to_fit, WorkspaceIndices=[45, 38])
ws_resolution = RemoveSpectra(ws_resolution, WorkspaceIndices=[45, 38])

# Can try other combination of starting parameters
initial_pars = {"y0": 0, "A": 1, "x0": 0, "sigma": 5}

# iminuit Fit
dataY = ws_to_fit.extractY()
dataE = ws_to_fit.extractE()
dataX = ws_to_fit.extractX()
dataRes = ws_resolution.extractY()

def model(x, A, x0, sigma):
    return  A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


def convolved_model(xrange, y0, A, x0, sigma, resDense, xDelta):
    """Evaluate the fitted profile before the least-squares residual is formed."""
    return y0 + signal.convolve(model(xrange, A, x0, sigma), resDense, mode="same") * xDelta

defaultPars = {}
totCost = 0
fitGrids = []
for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
    xDelta, resDense = oddPointsRes(x, res)
    fitGrids.append((x, xDelta, resDense))

    def conv_model(xrange, y0, *pars):
        """Performs numerical convolution"""
        return y0 + signal.convolve(model(xrange, *pars), resDense, mode="same") * xDelta
        # return y0 + model(xrange, *pars)

    limits = {"x": None, f"y0{i}": None, f"A{i}": None, f"x0{i}": None, "sigma": None}
    conv_model._parameters = limits

    costFun = cost.LeastSquares(x, y, yerr, conv_model)

    totCost += costFun

    defaultPars[f"y0{i}"] = initial_pars['y0']
    defaultPars[f"A{i}"] = initial_pars['A']
    defaultPars[f"x0{i}"] = initial_pars['x0']
    defaultPars["sigma"] = initial_pars['sigma']

print('Initial iminuit parameters:\n', defaultPars)
m = Minuit(totCost, **defaultPars)

t0 = time.time()

# m.simplex()
# m.migrad()

totSig = describe(totCost)  # This signature has 'x' already removed
print(f"\nDescribe: {totSig}")
sharedPars = ["sigma"]
shared_idxs = [totSig.index(shPar) for shPar in sharedPars]
nCostFunctions = len(totCost)  # Number of individual cost functions

def constr(*pars):
    """
    Constraint for positivity of the full convolved model.
    Input: All parameters defined in global cost function.
    Builds one concatenated constraint vector from each spectrum.
    """

    shared_pars = [pars[i] for i in shared_idxs]  # sigma1, c4, c6 in original GC
    unshared_pars_total = np.delete(pars, shared_idxs, None)
    unshared_pars_splitted = np.split(unshared_pars_total, nCostFunctions)  # Splits unshared parameters per individual cost fun

    joined_constr = np.zeros(sum(x.size for x, _, _ in fitGrids))
    offset = 0
    for i, (unshared_pars, (x, xDelta, resDense)) in enumerate(zip(unshared_pars_splitted, fitGrids)):
        # Each chunk is [y0i, Ai, x0i]; evaluate the same profile used by the fit.
        joined_constr[offset : offset + x.size] = convolved_model(
            x,
            *unshared_pars,
            *shared_pars,
            resDense,
            xDelta,
        )
        offset += x.size

    return joined_constr

# m.simplex()
# m.migrad()
m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))
# m.scipy()
# Explicitly calculate errors
m.hesse()

print(f"\nTime of iminuit: {time.time() - t0:.2f} seconds")
print(f"Value of Chi2/ndof: {m.fval / m.ndof:.2f}")
print(f"Migrad Minimum valid: {m.valid}")
print(f"Number of function calls: {m.nfcn}")
print("\nResults of iminuit Fit:\n")
for p, v, e in zip(m.parameters, m.values, m.errors):
    print(f"{p:>7s} = {v:>8.4f} \u00B1 {e:<8.4f}")

save_result_of_global_fit(dataX, dataY, dataE, m, totCost, ws_to_fit.name(), "gauss", m.fval / m.ndof)
