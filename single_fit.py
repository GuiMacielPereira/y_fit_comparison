from iminuit_fit_helpers import extractFirstSpectra, selectNonZeros, createFitResultsWorkspace, createCorrelationTableWorkspace, createFitParametersTableWorkspace, plotAutoMinos, oddPointsRes
from scipy import optimize
from scipy import  signal
from mantid.simpleapi import *
from iminuit import Minuit, cost
import numpy as np
import jacobi

# User inputs
nonlinear_contraint = False
initial_parameters = {"y0": 0, "A": 1, "x0": 0, "sigma1": 6, "c4": 0, "c6": 0}

ws_resolution = Load("./BaH2_500C_resolution_1_group.nxs")
ws_to_fit = Load("./BaH2_500C_1_group.nxs")

def model(x, A, x0, sigma1, c4, c6):
    return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
            * (1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 - 48*((x-x0)/np.sqrt(2)/sigma1)**2+12)
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 - 480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))

mantid_formula = "y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2)) \
                *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12) \
                +c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120))"

# Fit with iminuit
resX, resY, resE = extractFirstSpectra(ws_resolution)
xDelta, resDense = oddPointsRes(resX, resY)

def convolvedModel(x, y0, *pars):
    return y0 + signal.convolve(model(x, *pars), resDense, mode="same") * xDelta

defaultPars = {**initial_parameters}
limits = {"x": None, "y0": None, "A": None, "x0": None, "sigma1": None, "c4": None, "c6": None}
convolvedModel._parameters = limits

dataX, dataY, dataE = extractFirstSpectra(ws_to_fit)
dataXNZ, dataYNZ, dataENZ = selectNonZeros(dataX, dataY, dataE)

costFun = cost.LeastSquares(dataXNZ, dataYNZ, dataENZ, convolvedModel)

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

# Weighted Chi2
chi2 = m.fval / m.ndof

# Best fit and confidence band
# Calculated for the whole range of dataX, including where zero
dataYFit, dataYCov = jacobi.propagate(lambda pars: convolvedModel(dataX, *pars), m.values, m.covariance)
dataYSigma = np.sqrt(np.diag(dataYCov)) * chi2        # Weight the confidence band
Residuals = dataY - dataYFit

# Create workspace to store best fit curve and errors on the fit
wsMinFit = createFitResultsWorkspace(ws_to_fit, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals)

# Calculate correlation matrix
corrMatrix = m.covariance.correlation() * 100

# Create correlation tableWorkspace
createCorrelationTableWorkspace(ws_to_fit, m.parameters, corrMatrix)

# Extract info from fit before running any MINOS
parameters = list(m.parameters).copy()
values = list(m.values).copy()
errors = list(m.errors).copy()
minosAutoErr = list(np.zeros((len(parameters), 2)))
minosManErr = list(np.zeros((len(parameters), 2)))

# Run Minos
m.minos()
me = m.merrors

# Build minos errors lists in suitable format
print("\nWriting Minos errors")
for i, p in enumerate(parameters):
    minosAutoErr[i] = [me[p].lower, me[p].upper]

# Create workspace with final fitting parameters and their errors
createFitParametersTableWorkspace(ws_to_fit, parameters, values, errors, minosAutoErr, minosManErr, chi2)
plotAutoMinos(m, ws_to_fit.name())

# Mantid Fit
minimizer = 'Levenberg-Marquardt'

function = f"""
composite=Convolution,FixResolution=true,NumDeriv=true;
name=Resolution,Workspace={ws_resolution.name()},WorkspaceIndex=0;
name=UserFunction,Formula={mantid_formula},
y0={initial_parameters["y0"]},
A={initial_parameters["A"]},
x0={initial_parameters["x0"]},
sigma1={initial_parameters["sigma1"]},
c4={initial_parameters["c4"]},
c6={initial_parameters["c4"]},
ties=()
"""

Fit(
    Function=function,
    InputWorkspace=ws_to_fit,
    Output=ws_to_fit.name() + "_Mantid_Fit",
    Minimizer=minimizer
    )
