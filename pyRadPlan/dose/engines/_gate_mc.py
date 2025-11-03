from __future__ import annotations

import os
import pathlib
import subprocess
import logging
from typing import Optional, Sequence, Union
from glob import glob

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.dij import Dij
from pyRadPlan.io import write_ct_to_mhd, write_mask_to_mhd
from pyRadPlan.plan import Plan
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.core import Grid

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

        command = self._build_command(stf, exported)
        self._run_gate(command)

        return self._build_dij(ct)

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

    # ----------------------------------------------------------------- Gate run

    def _build_command(
        self,
        stf: SteeringInformation,
        exported: dict[str, pathlib.Path],
    ) -> list[str]:
        """
        Construct the command used to invoke the Gate executable.

        Currently this uses placeholder spot / gantry values until the steering
        information is fully mapped to Gate sources.
        """

        command: list[str] = [str(self.gate_exec)]

        # Allow caller to specify additional arguments (e.g. python script path)
        command.extend(map(str, self.extra_args))

        beam_definitions: list[tuple[float, float, float]] = []
        if stf and getattr(stf, "beams", None):
            for beam in stf.beams:
                gantry_angle = float(getattr(beam, "gantry_angle", 0.0))
                spot_x = 0.0
                spot_y = 0.0
                spot_map = getattr(beam, "spots", None)
                if spot_map is not None and len(spot_map) > 0:
                    first_spot = spot_map[0]
                    spot_x = float(getattr(first_spot, "position_x", spot_x))
                    spot_y = float(getattr(first_spot, "position_y", spot_y))
                beam_definitions.append((spot_x, spot_y, gantry_angle))

        if not beam_definitions:
            beam_definitions.append((0.0, 0.0, 0.0))

        # Keep backward compatibility for drivers expecting single-spot args
        first_spot_x, first_spot_y, first_gantry = beam_definitions[0]
        command.extend(
            [
                "--spot",
                str(first_spot_x),
                str(first_spot_y),
                "--gantry",
                str(first_gantry),
                "--id",
                "0",
                "--output",
                str(self._output_dir.resolve()),
                "--threads",
                str(self.threads),
            ]
        )

        for spot_x, spot_y, gantry_angle in beam_definitions:
            command.extend(["--beam", str(spot_x), str(spot_y), str(gantry_angle)])

        if self.seed is not None:
            command.extend(["--seed", str(self.seed)])

        return command

    def _run_gate(self, command: list[str]) -> None:
        """Execute the Gate process and surface errors."""

        logging.getLogger(__name__).debug("Running Gate command: %s", " ".join(command))

        proc = subprocess.run(
            command,
            cwd=self.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            logging.getLogger(__name__).error("Gate stdout:\n%s", proc.stdout)
            logging.getLogger(__name__).error("Gate stderr:\n%s", proc.stderr)
            raise RuntimeError(
                f"Gate execution failed with return code {proc.returncode}."
            )

        logging.getLogger(__name__).debug("Gate stdout:\n%s", proc.stdout)

    def _find_dose_file(self) -> pathlib.Path:
        """Locate the Gate-generated dose file."""

        pattern = str(self._output_dir / "dose3d*.mhd")
        candidates = sorted(glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No Gate dose file matching pattern {pattern} found in {self._output_dir}."
            )
        return pathlib.Path(candidates[-1])

    def _build_dij(self, ct: CT) -> Dij:
        """Construct a Dij object from the simulated dose cube."""

        dose_path = self._find_dose_file()
        dose_image = sitk.ReadImage(str(dose_path))
        dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float64)

        num_voxels = dose_array.size
        dose_matrix = dose_array.reshape(num_voxels, 1)

        physical_dose = np.empty((1,), dtype=object)
        physical_dose[0] = dose_matrix

        empty_quantity = np.empty((0,), dtype=object)

        return Dij(
            ct_grid=ct.grid,
            dose_grid=Grid.from_sitk_image(dose_image),
            physical_dose=physical_dose,
            let_dose=empty_quantity,
            alpha_dose=empty_quantity,
            sqrt_beta_dose=empty_quantity,
            num_of_beams=1,
            beam_num=np.array([0], dtype=int),
            ray_num=np.array([0], dtype=int),
            bixel_num=np.array([0], dtype=int),
        )
