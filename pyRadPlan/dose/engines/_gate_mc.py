from __future__ import annotations

import os
import pathlib
from typing import Optional, Sequence, Union

import numpy as np


from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.dij import Dij
from pyRadPlan.io import write_ct_to_mhd, write_mask_to_mhd
from pyRadPlan.plan import Plan
from pyRadPlan.stf import SteeringInformation

from ._base import DoseEngineBase


class GateMonteCarloEngine(DoseEngineBase):
    """
    Dose engine wrapper around a Gate (Geant4) Monte Carlo simulation.

    The engine establishes a canonical working layout:

    - ``{workdir}/input``  : exported CT volumes, masks, macro templates, etc.
    - ``{workdir}/output`` : Gate-produced artefacts (dose cubes, logs, ...).

    Actual Gate execution and result parsing are still left as TODOs; this class
    focuses on configuration plumbing and the geometry export scaffolding so the
    subsequent implementation steps can concentrate on macro generation and data
    ingestion.
    """

    short_name = "GATE_MC"
    name = "Gate Monte Carlo"
    possible_radiation_modes = ["electrons"]

    def __init__(
        self,
        pln: Union[Plan, dict] = None,
        gate_exec: Union[str, os.PathLike, None] = None,
        workdir: Union[str, os.PathLike, None] = None,
        threads: int = 1,
        seed: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
    ) -> None:
        # Defaults so assign_properties_from_pln (invoked by super().__init__)
        # can override them from plan.prop_dose_calc when present.
        self.gate_exec = gate_exec
        self.workdir = workdir
        self.threads = threads
        self.seed = seed
        self.extra_args = list(extra_args or ())

        self._input_dir: pathlib.Path | None = None
        self._output_dir: pathlib.Path | None = None

        # Let the base class assign plan properties / validation.
        super().__init__(pln)

        self._configure_gate_exec(gate_exec)
        self._configure_workdir(workdir)
        self._normalise_runtime_controls(seed)

    # ------------------------------------------------------------------ public

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
        """Prepare Gate inputs, run simulation (TODO), and build a Dij."""

        self._verify_workspace()
        exported = self._export_geometry(ct, cst)

        raise NotImplementedError(
            "Gate MC core dose calculation not implemented yet. "
            f"Exported artefacts: {exported}"
        )

    # --------------------------------------------------------------- configure

    def _configure_gate_exec(self, cli_override: Union[str, os.PathLike, None]) -> None:
        """Resolve the Gate executable or wrapper script path."""

        candidate = (
            cli_override
            or self.gate_exec
            or os.environ.get("GATE_EXECUTABLE")
        )
        if candidate is None:
            stub = pathlib.Path(__file__).parent / "gateexec" / "gate_stub.py"
            if not stub.exists():
                raise ValueError(
                    "Gate executable not provided (set gate_exec or GATE_EXECUTABLE)."
                )
            candidate = stub

        self.gate_exec = pathlib.Path(candidate)

    def _configure_workdir(self, cli_override: Union[str, os.PathLike, None]) -> None:
        """Resolve the working directory and make sure the layout exists."""

        candidate = (
            cli_override
            or self.workdir
            or os.environ.get("GATE_WORKDIR")
        )

        if candidate is None:
            base_path = (pathlib.Path(__file__).parent / "gateexec" / "workdir").resolve()
            base_path.mkdir(parents=True, exist_ok=True)
        else:
            base_path = pathlib.Path(candidate)
            base_path.mkdir(parents=True, exist_ok=True)

        self.workdir = base_path

        self._input_dir = self.workdir / "input"
        self._output_dir = self.workdir / "output"
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _normalise_runtime_controls(self, cli_seed: Optional[int]) -> None:
        """Normalise threads, seeds, and supplemental CLI arguments."""

        self.threads = int(self.threads)

        final_seed = cli_seed if cli_seed is not None else self.seed
        self.seed = int(final_seed) if final_seed is not None else None

        self.extra_args = list(self.extra_args or [])

    # ----------------------------------------------------------------- helpers

    def _verify_workspace(self) -> None:
        """Ensure workspace folders are initialised before exporting assets."""

        if self._input_dir is None or self._output_dir is None:
            raise RuntimeError("Gate workspace directories were not prepared.")

    def _export_geometry(self, ct: CT, cst: StructureSet) -> dict[str, pathlib.Path]:
        """
        Export CT data and VOI masks to MHD/RAW pairs consumable by Gate.

        Returns
        -------
        dict[str, Path]
            Mapping of artefact identifiers to the generated `.mhd` file paths.
        """

        exported: dict[str, pathlib.Path] = {}

        exported["ct"] = write_ct_to_mhd(ct, self._input_dir, basename="ct")

        target_mask = cst.target_union_mask()
        exported["target_mask"] = write_mask_to_mhd(
            target_mask,
            self._input_dir,
            basename="target_mask",
        )

        # Additional VOIs / material maps can be exported here as needed.

        return exported
