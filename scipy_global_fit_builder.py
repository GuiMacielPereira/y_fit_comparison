from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal


def gaussian_model(x: np.ndarray, A: float, x0: float, sigma: float) -> np.ndarray:
    return A / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))


def odd_points_res(x: np.ndarray, res: np.ndarray) -> Tuple[float, np.ndarray]:
    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    dens = res.size + 1 if res.size % 2 == 0 else res.size
    x_dense = np.linspace(np.min(x), np.max(x), dens)
    x_delta = x_dense[1] - x_dense[0]
    res_dense = np.interp(x_dense, x, res)
    return x_delta, res_dense


@dataclass
class GlobalConvolvedLeastSquares:
    data_x: np.ndarray
    data_y: np.ndarray
    data_e: np.ndarray
    data_res: np.ndarray
    shared_param_names: Tuple[str, ...] = ("sigma",)

    def __post_init__(self) -> None:
        self.n_spectra = int(self.data_x.shape[0])
        self.local_param_names = ("y0", "A", "x0")

        self.fit_grids: List[Tuple[np.ndarray, float, np.ndarray]] = []
        self.n_points = 0
        for x, res in zip(self.data_x, self.data_res):
            x_delta, res_dense = odd_points_res(x, res)
            self.fit_grids.append((x, x_delta, res_dense))
            self.n_points += x.size

        self.parameter_names = self._build_parameter_names()
        self.ndof = self.n_points - len(self.parameter_names)

    def _build_parameter_names(self) -> List[str]:
        names: List[str] = []
        for i in range(self.n_spectra):
            names.extend([f"y0{i}", f"A{i}", f"x0{i}"])
        names.extend(self.shared_param_names)
        return names

    def initial_vector(self, initial_pars: Dict[str, float]) -> np.ndarray:
        theta0: List[float] = []
        for _ in range(self.n_spectra):
            theta0.extend(
                [
                    float(initial_pars["y0"]),
                    float(initial_pars["A"]),
                    float(initial_pars["x0"]),
                ]
            )
        for name in self.shared_param_names:
            theta0.append(float(initial_pars[name]))
        return np.asarray(theta0, dtype=float)

    def unpack(self, theta: Sequence[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        theta_arr = np.asarray(theta, dtype=float)
        n_local = 3 * self.n_spectra
        local = theta_arr[:n_local].reshape(self.n_spectra, 3)
        shared_values = theta_arr[n_local:]
        shared = dict(zip(self.shared_param_names, shared_values))
        return local, shared

    def _convolved_model_one(
        self,
        x: np.ndarray,
        x_delta: float,
        res_dense: np.ndarray,
        y0: float,
        A: float,
        x0: float,
        sigma: float,
    ) -> np.ndarray:
        return y0 + signal.convolve(gaussian_model(x, A, x0, sigma), res_dense, mode="same") * x_delta

    def evaluate_profiles(self, theta: Sequence[float]) -> List[np.ndarray]:
        local, shared = self.unpack(theta)
        sigma = float(shared["sigma"])

        profiles: List[np.ndarray] = []
        for (x, x_delta, res_dense), pars in zip(self.fit_grids, local):
            y0, A, x0 = pars
            profiles.append(self._convolved_model_one(x, x_delta, res_dense, y0, A, x0, sigma))
        return profiles

    def residual_vector(self, theta: Sequence[float]) -> np.ndarray:
        profiles = self.evaluate_profiles(theta)
        chunks: List[np.ndarray] = []
        for y_obs, y_err, y_fit in zip(self.data_y, self.data_e, profiles):
            safe_err = np.where(y_err > 0.0, y_err, 1.0)
            chunks.append((y_obs - y_fit) / safe_err)
        return np.concatenate(chunks)

    def chi2(self, theta: Sequence[float]) -> float:
        r = self.residual_vector(theta)
        return float(np.dot(r, r))

    def positivity_constraint(self, theta: Sequence[float]) -> np.ndarray:
        return np.concatenate(self.evaluate_profiles(theta))

    def default_bounds(self) -> List[Tuple[float | None, float | None]]:
        bounds: List[Tuple[float | None, float | None]] = [(None, None)] * (3 * self.n_spectra)
        for name in self.shared_param_names:
            if name == "sigma":
                bounds.append((1e-12, None))
            else:
                bounds.append((None, None))
        return bounds


def save_global_fit_result_scipy(
    builder: GlobalConvolvedLeastSquares,
    theta_best: Sequence[float],
    ws_name: str,
    model_tag: str,
) -> None:
    from mantid.simpleapi import (  # Imported lazily to keep this module reusable outside Mantid runtime.
        CreateEmptyTableWorkspace,
        CreateWorkspace,
        GroupWorkspaces,
        Plus,
    )

    chi2 = builder.chi2(theta_best)
    chi2_ndof = chi2 / builder.ndof if builder.ndof > 0 else np.nan
    profiles = builder.evaluate_profiles(theta_best)
    local, shared = builder.unpack(theta_best)

    global_sum_name = ws_name + "_global_fit_sum"
    individual_names: List[str] = []

    chi2_table_name = ws_name + "_global_fit_" + model_tag + "_chi2"
    chi2_table = CreateEmptyTableWorkspace(OutputWorkspace=chi2_table_name)
    chi2_table.setTitle("Global Fit Chi2")
    chi2_table.addColumn(type="float", name="Chi2/ndof")
    chi2_table.addRow([chi2_ndof])

    pars_table_name = ws_name + "_global_fit_" + model_tag + "_Parameters"
    pars_table = CreateEmptyTableWorkspace(OutputWorkspace=pars_table_name)
    pars_table.setTitle("Global Fit Parameters")
    pars_table.addColumn(type="int", name="Group")
    pars_table.addColumn(type="str", name="y0")
    pars_table.addColumn(type="str", name="A")
    pars_table.addColumn(type="str", name="x0")
    for par_name in builder.shared_param_names:
        pars_table.addColumn(type="str", name=par_name)

    for i, (x, y, yerr, yfit) in enumerate(zip(builder.data_x, builder.data_y, builder.data_e, profiles)):
        res = y - yfit
        ws_i = ws_name + f"_global_fit_{i}"
        CreateWorkspace(
            DataX=np.concatenate([x, x, x]),
            DataY=np.concatenate([y, yfit, res]),
            DataE=np.concatenate([yerr, np.zeros_like(yerr), np.zeros_like(yerr)]),
            Nspec=3,
            OutputWorkspace=ws_i,
            Distribution=True,
        )
        individual_names.append(ws_i)

        if i == 1:
            Plus(ws_name + "_global_fit_0", ws_name + "_global_fit_1", OutputWorkspace=global_sum_name)
        elif i > 1:
            Plus(global_sum_name, ws_name + f"_global_fit_{i}", OutputWorkspace=global_sum_name)

        row = [i, f"{local[i, 0]:.6g}", f"{local[i, 1]:.6g}", f"{local[i, 2]:.6g}"]
        row.extend(f"{float(shared[name]):.6g}" for name in builder.shared_param_names)
        pars_table.addRow(row)

    if builder.n_spectra == 1:
        global_sum_name = ws_name + "_global_fit_0"

    individual_names.append(global_sum_name)
    individual_names.append(pars_table_name)
    individual_names.append(chi2_table_name)
    GroupWorkspaces(individual_names, OutputWorkspace=ws_name + "_global_fit_group")