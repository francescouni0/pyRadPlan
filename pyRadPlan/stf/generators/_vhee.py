"""Geometry generator for very high energy electron (VHEE) plans."""

from __future__ import annotations

import numpy as np

from .._beamlet import IonSpot
from .._ray import Ray
from ._ions import StfGeneratorIonRayBixel


class StfGeneratorVHEE(StfGeneratorIonRayBixel):
    """
    Steering geometry generator for VHEE beams.

    The implementation mirrors the MATLAB matRad VHEE steering generator by assigning a single,
    user-controlled energy to every ray on a regular lateral spot grid while leaving all other
    particle handling infrastructure untouched.
    """

    name = "VHEE Geometry Generator"
    short_name = "VHEE"
    possible_radiation_modes = ["VHEE"]

    def __init__(self, pln=None):
        self.radiation_mode = "VHEE"
        self.energy = 200.0  # Default energy in MeV if the plan does not set one
        super().__init__(pln)

    def _initialize(self):
        super()._initialize()

        available_energies = np.asarray(self.machine.energies, dtype=float)
        if available_energies.size == 0:
            raise ValueError("Machine does not contain any energies for VHEE irradiation.")

        if getattr(self, "energy", None) is None:
            self.energy = float(available_energies.max())

        self.energy = float(self.energy)

        if not np.any(np.isclose(available_energies, self.energy, rtol=1e-5, atol=1e-2)):
            raise ValueError(
                f"Requested VHEE energy {self.energy} MeV is not available in the machine data "
                f"({available_energies.tolist()})."
            )

    def _create_rays(self, beam: dict) -> list[Ray]:
        raw_rays = super()._create_rays(beam)

        rays: list[Ray] = []
        for raw_ray in raw_rays:
            raw_ray["beamlets"] = [IonSpot(energy=self.energy)]
            rays.append(Ray.model_validate(raw_ray))

        beam["num_of_bixels_per_ray"] = np.ones(len(rays), dtype=int)

        return rays
