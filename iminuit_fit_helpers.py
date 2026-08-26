import matplotlib.pyplot as plt
import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from scipy import  signal
from iminuit import Minuit, cost
from iminuit.util import make_func_code, describe
import jacobi


def extractFirstSpectra(ws):
    dataY = ws.extractY()[0]
    dataX = ws.extractX()[0]
    dataE = ws.extractE()[0]
    return dataX, dataY, dataE


def selectNonZeros(dataX, dataY, dataE):
    """
    Selects non zero points.
    Uses zeros in dataY becasue dataE can be all zeros in one of the bootstrap types.
    """
    zeroMask = dataY==0

    dataXNZ = dataX[~zeroMask]
    dataYNZ = dataY[~zeroMask]
    dataENZ = dataE[~zeroMask]
    return dataXNZ, dataYNZ, dataENZ


def createFitResultsWorkspace(wsYSpaceSym, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals):
    """Creates workspace similar to the ones created by Mantid Fit."""

    wsMinFit = CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)),
                               DataY=np.concatenate((dataY, dataYFit, Residuals)),
                               DataE=np.concatenate((dataE, dataYSigma, np.zeros(len(dataE)))),
                               NSpec=3,
                               OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit")
    return wsMinFit


def createCorrelationTableWorkspace(wsYSpaceSym, parameters, corrMatrix):
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_NormalizedCovarianceMatrix")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str',name="Name")
    for p in parameters:
        tableWS.addColumn(type='float',name=p)
    for p, arr in zip(parameters, corrMatrix):
        tableWS.addRow([p] + list(arr))


def createFitParametersTableWorkspace(wsYSpaceSym, parameters, values, errors, minosAutoErr, minosManualErr, chi2):
    # Create Parameters workspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_Parameters")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str', name="Name")
    tableWS.addColumn(type='float', name="Value")
    tableWS.addColumn(type='float', name="Error")
    tableWS.addColumn(type='float', name="Auto Minos Error-")
    tableWS.addColumn(type='float', name="Auto Minos Error+")
    tableWS.addColumn(type='float', name="Manual Minos Error-")
    tableWS.addColumn(type='float', name="Manual Minos Error+")

    for p, v, e, mae, mme in zip(parameters, values, errors, minosAutoErr, minosManualErr):
        tableWS.addRow([p, v, e, mae[0], mae[1], mme[0], mme[1]])

    tableWS.addRow(["Cost function", chi2, 0, 0, 0, 0, 0])
    return


def plotAutoMinos(minuitObj, wsName):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})
    fig.canvas.setWindowTitle(wsName+"_Plot_Automatic_MINOS")

    for p, ax in zip(minuitObj.parameters, axs.flat):
        loc, fvals, status = minuitObj.mnprofile(p, bound=2)

        minfval = minuitObj.fval
        minp = minuitObj.values[p]
        hessp = minuitObj.errors[p]
        lerr = minuitObj.merrors[p].lower
        uerr = minuitObj.merrors[p].upper
        plotProfile(ax, p, loc, fvals, lerr, uerr, minfval, minp, hessp)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    fig.show()


def plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr):
    """
    Plots likelihood profilef for the Migrad fvals.
    varSpace : x axis
    fValsMigrad : y axis
    """

    ax.set_title(var+f" = {varVal:.3f} {lerr:.3f} {uerr:+.3f}")

    ax.plot(varSpace, fValsMigrad, label="fVals Migrad")

    ax.axvspan(lerr+varVal, uerr+varVal, alpha=0.2, color="red", label="Minos error")
    ax.axvspan(varVal-varErr, varVal+varErr, alpha=0.2, color="green", label="Hessian Std error")

    ax.axvline(varVal, 0.03, 0.97, color="k", ls="--")
    ax.axhline(fValsMin+1, 0.03, 0.97, color="k")
    ax.axhline(fValsMin, 0.03, 0.97, color="k")


def oddPointsRes(x, res):
    """
    Make a odd grid that ensures a resolution with a single peak at the center.
    """

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if res.size % 2 == 0:
        dens = res.size+1  # If even change to odd
    else:
        dens = res.size    # If odd, keep being odd

    xDense = np.linspace(np.min(x), np.max(x), dens)    # Make gridd with odd number of points - peak at center
    xDelta = xDense[1] - xDense[0]

    resDense = np.interp(xDense, x, res)

    return xDelta, resDense

def save_result_of_global_fit(data_x, data_y, data_e, m, total_cost_fun, ws_name, fit_model, chi2):
    global_sum_name = ws_name + "_global_fit_sum"
    individual_names = []

    # Create table with only chi2
    chi2_table_name = ws_name + "_global_fit_" + fit_model + "_chi2"
    chi2_table = CreateEmptyTableWorkspace(OutputWorkspace=chi2_table_name)
    chi2_table.setTitle("Global Fit Chi2")
    chi2_table.addColumn(type="float", name="Chi2")
    chi2_table.addRow([chi2])

    # Create table of parameters
    pars_table_name = ws_name + "_global_fit_" + fit_model + "_Parameters"
    pars_table = CreateEmptyTableWorkspace(OutputWorkspace=pars_table_name)
    pars_table.setTitle("Global Fit Parameters")
    pars_table.addColumn(type="int", name="Group")
    signature = describe(total_cost_fun[0])
    for parameter in signature:
        parameter = parameter[:-1] if parameter.endswith("0") else parameter
        pars_table.addColumn(type="str", name=parameter)

    for i, (x, y, yerr, cost_fun) in enumerate(zip(data_x, data_y, data_e, total_cost_fun)):
        signature = describe(cost_fun)
        values = m.values[signature]

        yfit = cost_fun.model(x, *values)
        res = y - yfit

        CreateWorkspace(
            DataX=np.concatenate([x, x, x]),
            DataY=np.concatenate([y, yfit, res]),
            DataE=np.concatenate([yerr, np.zeros_like(yerr), np.zeros_like(yerr)]),
            Nspec=3,
            OutputWorkspace=ws_name + f"_global_fit_{i}",
            Distribution=True,
        )
        individual_names.append(ws_name + f"_global_fit_{i}")

        if i == 1:
            Plus(ws_name + "_global_fit_0", ws_name + "_global_fit_1", OutputWorkspace=global_sum_name)
        elif i > 1:
            Plus(global_sum_name, ws_name + f"_global_fit_{i}", OutputWorkspace=global_sum_name)

        # Build strings for par values
        errors = m.errors[signature]
        pars_table.addRow([i] + [f" {v:.3f} +/- {e:.3f} " for v, e in zip(values, errors)])

    individual_names.append(global_sum_name)
    individual_names.append(pars_table_name)
    individual_names.append(chi2_table_name)

    return GroupWorkspaces(individual_names, OutputWorkspace=ws_name + "_global_fit_group")
