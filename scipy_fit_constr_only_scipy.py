from scipy import optimize
from mantid.simpleapi import Load, RemoveSpectra
import numpy as np
import time

from scipy_global_fit_builder import GlobalConvolvedLeastSquares, save_global_fit_result_scipy


ws_resolution = Load("D_HMT_resolution.nxs")
ws_to_fit = Load("D_HMT_forward.nxs")

# Need to remove any Masked spectra on both workspaces
# In this case spectra idx 38 has very anomalous data, need to remove for good fit
ws_to_fit = RemoveSpectra(ws_to_fit, WorkspaceIndices=[45, 38])
ws_resolution = RemoveSpectra(ws_resolution, WorkspaceIndices=[45, 38])

# Can try other combination of starting parameters
initial_pars = {"y0": 0.0, "A": 1.0, "x0": 0.0, "sigma": 5.0}

# Choose SciPy constrained optimizer: "SLSQP" or "trust-constr"
optimizer_method = "SLSQP"

data_y = ws_to_fit.extractY()
data_e = ws_to_fit.extractE()
data_x = ws_to_fit.extractX()
data_res = ws_resolution.extractY()

builder = GlobalConvolvedLeastSquares(
    data_x=data_x,
    data_y=data_y,
    data_e=data_e,
    data_res=data_res,
    shared_param_names=("sigma",),
)

theta0 = builder.initial_vector(initial_pars)
param_names = builder.parameter_names

constraint = optimize.NonlinearConstraint(builder.positivity_constraint, 0.0, np.inf)

if optimizer_method == "trust-constr":
    minimizer_options = {
        "maxiter": 2000,
        "xtol": 1e-10,
        "gtol": 1e-8,
        "barrier_tol": 1e-10,
        "verbose": 3,
    }
elif optimizer_method == "SLSQP":
    minimizer_options = {"maxiter": 2000, "ftol": 1e-10, "disp": True}
else:
    raise ValueError("optimizer_method must be 'SLSQP' or 'trust-constr'")

t0 = time.time()
result = optimize.minimize(
    fun=builder.chi2,
    x0=theta0,
    method=optimizer_method,
    bounds=builder.default_bounds(),
    constraints=[constraint],
    options=minimizer_options,
)
elapsed = time.time() - t0

chi2 = builder.chi2(result.x)
chi2_ndof = chi2 / builder.ndof if builder.ndof > 0 else np.nan

print(f"\nTime of scipy optimize.minimize ({optimizer_method}): {elapsed:.2f} seconds")
print(f"Value of Chi2/ndof: {chi2_ndof:.2f}")
print(f"Converged: {result.success}")
print(f"Status code: {result.status}")
print(f"Message: {result.message}")
print(f"Number of function calls: {result.nfev}")
print("\nResults of scipy Fit:\n")
for p, v in zip(param_names, result.x):
    print(f"{p:>7s} = {v:>10.6f}")

save_global_fit_result_scipy(builder, result.x, ws_to_fit.name(), "gauss_scipy")