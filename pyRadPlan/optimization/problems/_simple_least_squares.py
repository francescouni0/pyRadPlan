from typing import Union
import numpy as np
from numpy.typing import ArrayLike

from ...plan import Plan
from ._optiprob import NonLinearPlanningProblem
from ..solvers import NonLinearOptimizer


class SimpleLeastSquaresFluenceOptimization(NonLinearPlanningProblem):
    """Simple least squares fluence-based planning problem."""

    name = "VHEE Least-Squares Fluence Problem"
    short_name = "vhee_simple_ls"
    possible_radiation_modes = ["VHEE"]

    penalties: ArrayLike
    result: dict[str]

    def __init__(self, pln: Union[Plan, dict] = None):
        self.penalties = np.asarray([1000, 1])
        self.target_prescription = 2.0
        self._target_voxels = None
        self._patient_voxels = None

        self._grad_cache_intermediate = None
        self._grad_cache = None
        self._result_cache = None
        self._w_cache = None
        self.result = {}

        super().__init__(pln)

    def _initialize(self):
        # Clean up objectives list to avoid invalid placeholders coming from legacy datasets
        for voi in self._cst.vois:
            if isinstance(voi.objectives, list):
                cleaned = []
                for obj in voi.objectives:
                    skip = False
                    if obj is None:
                        skip = True
                    elif isinstance(obj, (list, tuple)) and len(obj) == 0:
                        skip = True
                    elif isinstance(obj, dict) and len(obj) == 0:
                        skip = True
                    elif hasattr(obj, "size") and getattr(obj, "size") == 0:
                        skip = True

                    if not skip:
                        cleaned.append(obj)
                voi.objectives = cleaned

        super()._initialize()
        self._target_voxels = self._cst.target_union_voxels(order="numpy")
        self._patient_voxels = self._cst.patient_voxels(order="numpy")
        if self._target_voxels.size == 0:
            raise ValueError("Target structure not available for VHEE optimisation.")
        if self._patient_voxels.size == 0:
            raise ValueError("Patient mask not available for VHEE optimisation.")

        self.penalties = np.asarray(self.penalties, dtype=np.float64)
        if self.penalties.ndim != 1 or self.penalties.size != 2:
            raise ValueError("penalties must be a length-2 vector [target_weight, patient_weight].")

        # Check if the solver is adequate to solve this problem
        # TODO: check that it can do constraints
        if not isinstance(self.solver, NonLinearOptimizer):
            raise ValueError("Solver must be an instance of SolverBase")

        self.solver.objective = self._objective_function
        self.solver.gradient = self._objective_gradient
        self.solver.bounds = (0.0, np.inf)
        self.solver.max_iter = 500
        self.solver.abs_obj_tol = 1e-4
        if hasattr(self.solver, "options"):
            self.solver.options.update(
                {
                    "print_level": 0,
                    "print_user_options": "no",
                    "print_options_documentation": "no",
                    "print_timing_statistics": "no",
                }
            )

    def _objective_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the objective functions."""

        if not np.array_equal(x, self._w_cache):
            self._result_cache = self._dij.get_result_arrays_from_intensity(x)
            self._w_cache = x.copy()

        dose = self._result_cache["physical_dose"]
        target_fun = (
            np.sum((dose[self._target_voxels] - self.target_prescription) ** 2)
            / self._target_voxels.size
        )
        patient_fun = np.sum((dose[self._patient_voxels]) ** 2) / self._patient_voxels.size
        return np.array([target_fun, patient_fun])

    def _objective_function(self, x: np.ndarray) -> np.float64:
        return np.dot(np.asarray(self.penalties), self._objective_functions(x))

    def _objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective jacobian."""

        if not np.array_equal(x, self._w_cache):
            self._result_cache = self._dij.get_result_arrays_from_intensity(x)
            self._w_cache = x.copy()

        dose = self._result_cache["physical_dose"]

        if self._grad_cache_intermediate is None:
            self._grad_cache_intermediate = np.zeros((dose.size, 2))
        else:
            self._grad_cache_intermediate.fill(0.0)

        # We do no sanity checks here, i.e., the grids need to be the same

        self._grad_cache_intermediate[self._target_voxels, 0] = (
            2.0 / self._target_voxels.size * (dose[self._target_voxels] - self.target_prescription)
        )
        self._grad_cache_intermediate[self._patient_voxels, 1] = (
            2.0 / self._patient_voxels.size * (dose[self._patient_voxels])
        )

        self._grad_cache = self._dij.physical_dose.flat[0].T @ self._grad_cache_intermediate

        return self._grad_cache

    def _objective_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self._objective_jacobian(x) * self.penalties, axis=1)

    def _objective_hessian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective hessian."""
        return {}

    def _constraint_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint functions."""
        return None

    def _constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint jacobian."""
        return None

    def _constraint_jacobian_structure(self) -> np.ndarray:
        """Define the constraint jacobian structure."""
        return None

    def _variable_bounds(self, x: np.ndarray) -> np.ndarray:
        """Define the variable bounds."""
        return {}

    def _solve(self) -> tuple[np.ndarray, dict]:
        """Solve the problem."""

        x0 = np.zeros((self._dij.total_num_of_bixels,), dtype=np.float64)
        return self.solver.solve(x0)
