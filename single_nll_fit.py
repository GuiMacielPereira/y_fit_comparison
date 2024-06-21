from iminuit_fit_helpers import extractFirstSpectra, selectNonZeros, createFitResultsWorkspace, createCorrelationTableWorkspace, createFitParametersTableWorkspace, plotAutoMinos, oddPointsRes
from scipy import optimize
from scipy import  signal
from scipy import stats
from matplotlib import pyplot as plt
from mantid.simpleapi import *
from iminuit import Minuit, cost
import numpy as np
import jacobi

# User inputs
nonlinear_contraint = False

ws_resolution = Load("./benzoic_250k_resolution_sum.nxs")
ws_to_fit = Load("./benzoic_250k_mass0_avg.nxs")

# # Fit with iminuit
# resX, resY, resE = extractFirstSpectra(ws_resolution)
# xDelta, resDense = oddPointsRes(resX, resY)
#
# def convolvedModel(x, y0, *pars):
#     return y0 + signal.convolve(model(x, *pars), resDense, mode="same") * xDelta

# def model(x, x0, sig):
#     return stats.norm.pdf(x, x0, sig)
#
# defaultPars = {"x0": 0, "sig": 6}
# limits = {"x": None, "x0": None, "sig": (0, np.inf)}
# model._parameters = limits

def model(x, x0, sigma1, c4, c6):
    return  np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
            * (1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 - 48*((x-x0)/np.sqrt(2)/sigma1)**2+12)
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 - 480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))

# Fit with iminuit
resX, resY, resE = extractFirstSpectra(ws_resolution)
xDelta, resDense = oddPointsRes(resX, resY)
def convolvedModel(x, *pars):
    return signal.convolve(model(x, *pars), resDense, mode="same") * xDelta

defaultPars = {"x0": 0, "sigma1": 6, "c4": 0, "c6": 0}
limits = {"x": None, "x0": None, "sigma1": (0, None), "c4": None, "c6": None}
convolvedModel._parameters = limits
model._parameters = limits

dataX, dataY, dataE = extractFirstSpectra(ws_to_fit)
dataXNZ, dataYNZ, dataENZ = selectNonZeros(dataX, dataY, dataE)

dataYNZ[dataYNZ<0] = 0

sample = np.array([])
for x, y in zip(dataXNZ, dataYNZ):
    sample = np.append(sample, np.repeat(x, int(max(y * 1e5, 0))))

fig, ax = plt.subplots()
ax.plot(dataX, dataY, "k.", label="data")

for costFun, label in zip((cost.UnbinnedNLL(sample, model), cost.LeastSquares(dataXNZ, dataYNZ, dataENZ, model)), ("NLL", "LS")):
    print(cost)

    m = Minuit(costFun, **defaultPars)

    m.simplex()

    if not nonlinear_contraint:
        m.migrad()
    else:
        def constrFunc(*pars):   # Constrain physical model before convolution
            return model(dataXNZ, *pars[1:])   # First parameter is intercept, not part of model()

        m.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))


# Explicit calculation of Hessian after the fit
    m.hesse()
    m.minos()

    print(m.params)
    print(m.errors)
    print(m.covariance)


# Weighted Chi2
    chi2 = m.fval / m.ndof

# Best fit and confidence band
# Calculated for the whole range of dataX, including where zero
    dataYFit, dataYCov = jacobi.propagate(lambda pars: convolvedModel(dataX, *pars), m.values, m.covariance)
    dataYSigma = np.sqrt(np.diag(dataYCov)) * chi2        # Weight the confidence band

    
    ax.plot(dataX, dataYFit, label=label)

plt.legend()
plt.show()
# Residuals = dataY - dataYFit
#
# # Create workspace to store best fit curve and errors on the fit
# wsMinFit = createFitResultsWorkspace(ws_to_fit, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals)
#
# # Calculate correlation matrix
# corrMatrix = m.covariance.correlation() * 100
#
# # Create correlation tableWorkspace
# createCorrelationTableWorkspace(ws_to_fit, m.parameters, corrMatrix)
#
# # Extract info from fit before running any MINOS
# parameters = list(m.parameters).copy()
# values = list(m.values).copy()
# errors = list(m.errors).copy()
# minosAutoErr = list(np.zeros((len(parameters), 2)))
# minosManErr = list(np.zeros((len(parameters), 2)))
#
# # Run Minos
# m.minos()
# me = m.merrors
#
# # Build minos errors lists in suitable format
# print("\nWriting Minos errors")
# for i, p in enumerate(parameters):
#     minosAutoErr[i] = [me[p].lower, me[p].upper]
#
# # Create workspace with final fitting parameters and their errors
# createFitParametersTableWorkspace(ws_to_fit, parameters, values, errors, minosAutoErr, minosManErr, chi2)
# plotAutoMinos(m, ws_to_fit.name())
