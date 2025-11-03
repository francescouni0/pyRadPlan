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

MM_TO_CM = 0.1


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

        requests = self._extract_gate_requests(stf)
        if not requests:
            raise ValueError("No beam spots available for Gate Monte Carlo simulation.")

        for old_file in self._output_dir.glob("dose3d*.mhd"):
            old_file.unlink()
        for old_file in self._output_dir.glob("dose3d*.raw"):
            old_file.unlink()

        dose_columns: list[np.ndarray] = []
        beam_num: list[int] = []
        ray_num: list[int] = []
        dose_image_reference: Optional[sitk.Image] = None

        for column_index, request in enumerate(requests):
            self._clear_output_files(column_index)
            command = self._build_command(request, exported, column_index)
            self._run_gate(command)
            column, dose_image = self._load_dose_column(column_index)
            column *= request.get("weight", 1.0)
            if dose_image_reference is None:
                dose_image_reference = dose_image
            dose_columns.append(column)
            beam_num.append(request["beam_index"])
            ray_num.append(request["ray_index"])
        # Note: beamlet index information can be encoded into ray/beam mappings if needed.

        dose_matrix = np.column_stack(dose_columns)

        physical_dose = np.empty((1,), dtype=object)
        physical_dose[0] = dose_matrix

        empty_quantity = np.empty((0,), dtype=object)

        dose_grid = Grid.from_sitk_image(dose_image_reference) if dose_image_reference else ct.grid

        beam_num_arr = np.array(beam_num, dtype=int)
        ray_num_arr = np.array(ray_num, dtype=int)
        bixel_num_arr = np.arange(dose_matrix.shape[1], dtype=int)
        num_of_beams = int(np.unique(beam_num_arr).size)

        return Dij(
            ct_grid=ct.grid,
            dose_grid=dose_grid,
            physical_dose=physical_dose,
            let_dose=empty_quantity,
            alpha_dose=empty_quantity,
            sqrt_beta_dose=empty_quantity,
            num_of_beams=num_of_beams,
            beam_num=beam_num_arr,
            ray_num=ray_num_arr,
            bixel_num=bixel_num_arr,
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

    # ----------------------------------------------------------------- Gate run

    def _extract_gate_requests(self, stf: SteeringInformation) -> list[dict[str, float]]:
        """Collect beam/ray positions to be simulated with Gate."""

        requests: list[dict[str, float]] = []
        if not stf or not getattr(stf, "beams", None):
            return requests

        for beam_index, beam in enumerate(stf.beams):
            gantry_angle = float(getattr(beam, "gantry_angle", 0.0))
            rays = getattr(beam, "rays", [])
            if not rays:
                continue
            for ray_index, ray in enumerate(rays):
                ray_pos = getattr(ray, "ray_pos", None)
                if ray_pos is None:
                    continue
                spot_x_cm = float(ray_pos[0]) * MM_TO_CM
                spot_y_cm = float(ray_pos[2]) * MM_TO_CM
                beamlets = getattr(ray, "beamlets", [])
                if not beamlets:
                    requests.append(
                        {
                            "beam_index": beam_index,
                            "ray_index": ray_index,
                            "beamlet_index": 0,
                            "spot_x_cm": spot_x_cm,
                            "spot_y_cm": spot_y_cm,
                            "gantry_deg": gantry_angle,
                            "weight": 1.0,
                        }
                    )
                    continue

                # Use the first beamlet as representative for now (TODO: extend to all beamlets)
                for beamlet_index, beamlet in enumerate(beamlets):
                    weight = float(getattr(beamlet, "weight", 0.0))
                    if weight == 0.0:
                        continue
                    requests.append(
                        {
                            "beam_index": beam_index,
                            "ray_index": ray_index,
                            "beamlet_index": beamlet_index,
                            "spot_x_cm": spot_x_cm,
                            "spot_y_cm": spot_y_cm,
                            "gantry_deg": gantry_angle,
                            "weight": weight,
                        }
                    )
                    break

        return requests

    def _build_command(
        self,
        request: dict[str, float],
        exported: dict[str, pathlib.Path],
        identifier: int,
    ) -> list[str]:
        """
        Construct the command used to invoke the Gate executable.

        Currently this uses placeholder spot / gantry values until the steering
        information is fully mapped to Gate sources.
        """

        command: list[str] = [str(self.gate_exec)]

        # Allow caller to specify additional arguments (e.g. python script path)
        command.extend(map(str, self.extra_args))

        command.extend(
            [
                "--spot",
                str(request["spot_x_cm"]),
                str(request["spot_y_cm"]),
                "--gantry",
                str(request["gantry_deg"]),
                "--id",
                str(identifier),
                "--output",
                str(self._output_dir.resolve()),
                "--threads",
                str(self.threads),
            ]
        )

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

    def _clear_output_files(self, identifier: int) -> None:
        base = self._output_dir / f"dose3d{identifier}"
        for ext in (".mhd", ".raw", ".mha"):
            path = base.with_suffix(ext)
            if path.exists():
                path.unlink()

    def _find_dose_file(self, identifier: Optional[int] = None) -> pathlib.Path:
        """Locate the Gate-generated dose file."""

        if identifier is not None:
            candidate = self._output_dir / f"dose3d{identifier}.mhd"
            if candidate.exists():
                return candidate

        pattern = str(self._output_dir / "dose3d*.mhd")
        candidates = sorted(glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No Gate dose file matching pattern {pattern} found in {self._output_dir}."
            )
        return pathlib.Path(candidates[-1])

    def _load_dose_column(self, identifier: int) -> tuple[np.ndarray, sitk.Image]:
        dose_path = self._find_dose_file(identifier)
        dose_image = sitk.ReadImage(str(dose_path))
        dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float64)
        return dose_array.ravel(), dose_image
