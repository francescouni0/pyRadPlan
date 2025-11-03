from __future__ import annotations

import pathlib
from typing import Optional, Union

import numpy as np

from ._base import DoseEngineBase
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.plan import Plan
from pyRadPlan.dij import Dij


class GateMonteCarloEngine(DoseEngineBase):
    """Dose engine wrapper around Gate Monte Carlo simulations."""

    short_name = "GATE_MC"
    name = "Gate Monte Carlo"
    possible_radiation_modes = ["electrons"]

    def __init__(
        self,
        pln: Union[Plan, dict] = None,
        gate_exec: Union[str, pathlib.Path, None] = None,
        workdir: Union[str, pathlib.Path, None] = None,
        threads: int = 1,
    ) -> None:
        super().__init__(pln)
        self.gate_exec = pathlib.Path(gate_exec) if gate_exec is not None else None
        self.workdir = pathlib.Path(workdir) if workdir is not None else None
        self.threads = threads

    def calc_dose_influence(
        self,
        ct: CT,
        cst: StructureSet,
        stf: SteeringInformation,
        pln: Plan,
    ) -> Dij:
        raise NotImplementedError("Gate MC dose influence not implemented yet.")

    def calc_dose_forward(
        self,
        ct: CT,
        cst: StructureSet,
        stf: SteeringInformation,
        pln: Plan,
        weights: Optional[np.ndarray] = None,
    ) -> Dij:
        raise NotImplementedError("Gate MC forward dose not implemented yet.")

    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> Dij:
        raise NotImplementedError("Gate MC core dose calculation not implemented yet.")
