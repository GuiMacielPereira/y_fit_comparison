from iminuit_fit_helpers import oddPointsRes
from scipy import signal
from mantid.simpleapi import *
from iminuit import Minuit, cost
import numpy as np
import time

ws_resolution = Load("./D_HMT_resolution.nxs")
ws_to_fit = Load("./D_HMT_forward.nxs")

# Need to remove any Masked spectra on both workspaces
# In this case spectra idx 38 has very anomalous data, need to remove for good fit
ws_to_fit = RemoveSpectra(ws_to_fit, WorkspaceIndices=[45, 38])
ws_resolution = RemoveSpectra(ws_resolution, WorkspaceIndices=[45, 38])

# Can try other combination of starting parameters
initial_pars = {"y0": 0, "A": 1, "x0": 0, "sigma": 5}

# Mantid shared fit
gauss_str = "y0 + A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5"
gauss_str += f", y0={initial_pars['y0']},A={initial_pars['A']},x0={initial_pars['x0']},sigma={initial_pars['sigma']}"

function = "composite=MultiDomainFunction,NumDeriv=true;"
for i in range(ws_to_fit.getNumberHistograms()):
    function += \
        f"(composite=Convolution, FixResolution=true,NumDeriv=true, $domains={i};" \
        f"name=Resolution, Workspace=ws_resolution, WorkspaceIndex={i};" \
        f"name=UserFunction,Formula={gauss_str});"

ties = "ties=(f1.f1.sigma=f0.f1.sigma"
for i in range(2, ws_to_fit.getNumberHistograms()):
    ties += f", f{i}.f1.sigma=f0.f1.sigma"
ties += ")"

function += ties

fit_kwargs = {
    "InputWorkspace": ws_to_fit,
    "WorkspaceIndex": 0
}
for i in range(1, ws_to_fit.getNumberHistograms()):
    fit_kwargs[f"InputWorkspace_{i}"] = ws_to_fit
    fit_kwargs[f"WorkspaceIndex_{i}"] = i

# Fit
t0 = time.time()
fit_output = Fit(
    Function=function,
    **fit_kwargs,
    MaxIterations=10000,
    Minimizer="Levenberg-Marquardt",
    CostFunction="Least squares",
    EvaluationType="CentrePoint",
    Output="Output_Fit"
)
print(f"\nTime of Mantid Fit: {time.time() - t0:.2f} seconds")

# iminuit Fit
dataY = ws_to_fit.extractY()
dataE = ws_to_fit.extractE()
dataX = ws_to_fit.extractX()
dataRes = ws_resolution.extractY()

def model(x, A, x0, sigma):
    return  A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

defaultPars = {}
totCost = 0
for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
    xDelta, resDense = oddPointsRes(x, res)

    def convolvedModel(xrange, y0, *pars):
        """Performs numerical convolution"""
        return y0 + signal.convolve(model(xrange, *pars), resDense, mode="same") * xDelta

    limits = {"x": None, f"y0{i}": None, f"A{i}": None, f"x0{i}": None, "sigma": None}
    convolvedModel._parameters = limits

    costFun = cost.LeastSquares(x, y, yerr, convolvedModel)

    totCost += costFun

    defaultPars[f"y0{i}"] = initial_pars['y0']
    defaultPars[f"A{i}"] = initial_pars['A']
    defaultPars[f"x0{i}"] = initial_pars['x0']
    defaultPars["sigma"] = initial_pars['sigma']

print('Initial iminuit parameters:\n', defaultPars)
m = Minuit(totCost, **defaultPars)

t0 = time.time()

m.simplex()
m.migrad()
# Explicitly calculate errors
m.hesse()

print(f"\nTime of iminuit: {time.time() - t0:.2f} seconds")
print(f"Value of Chi2/ndof: {m.fval / m.ndof:.2f}")
print(f"Migrad Minimum valid: {m.valid}")
print(f"Number of function calls: {m.nfcn}")
print("\nResults of iminuit Fit:\n")
for p, v, e in zip(m.parameters, m.values, m.errors):
    print(f"{p:>7s} = {v:>8.4f} \u00B1 {e:<8.4f}")

