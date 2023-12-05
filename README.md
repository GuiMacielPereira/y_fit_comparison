# Mantid Fit & iminuit
Simple scripts for comparing fitting with Mantid and iminuit.
`single_fit.py` fits a single profile and shows the capability of Minos in determining asymmetric errors on the parameters.
`shared_fit.py` fits multiple profiles with shared parameters and is meant to be used to compare output workspaces of both Fits.

## Set-up
Download and install latest version of Mantid (at the time of writing, this was 6.8)
Run the following command to make iminuit and jacobi accessible by Mantid:
`path/to/mantid/python -m pip install iminuit, jacobi`
