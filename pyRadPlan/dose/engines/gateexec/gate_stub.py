#!/usr/bin/env python3
"""Gate/OpenGATE stub simulation.

This script mirrors the structure of a Gate 10 Python driver so the
`GateMonteCarloEngine` can invoke it during development.  It builds a minimal
scene (water phantom, electron source, dose actor) and emits an MHD dose file
using the OpenGATE Python API.
"""

from __future__ import annotations

import argparse
import numpy as np
import opengate as gate
from scipy.spatial.transform import Rotation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate/OpenGATE development stub")
    parser.add_argument("--source_n", type=int, default=10, help="Number of sources (unused)")
    parser.add_argument("--id", type=int, default=0, help="Identifier used in the dose filename")
    parser.add_argument("--spot", type=float, nargs="+", default=[0.0, 0.0], help="Spot coordinates (x y)")
    parser.add_argument(
        "--isocenter",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="Isocenter translation (unused)",
    )
    parser.add_argument("--gantry", type=float, default=0.0, help="Gantry angle in degrees")
    parser.add_argument("--output", type=str, default="pyRadPlan/pyRadPlan/dose/engines/gateexec/workdir/output", help="Output directory")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads")
    parser.add_argument("--seed", type=str, default="auto", help="Random engine seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.g4_verbose_level = 0
    sim.visu = False
    sim.visu_type = "vrml"
    sim.random_engine = "MersenneTwister"
    sim.random_seed = args.seed
    sim.output_dir = args.output
    sim.number_of_threads = args.threads
    sim.progress_bar = True

    m = gate.g4_units.m
    cm = gate.g4_units.cm
    cm3 = gate.g4_units.cm3
    keV = gate.g4_units.keV  # noqa: F841
    MeV = gate.g4_units.MeV
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq  # noqa: F841
    gcm3 = gate.g4_units.g / cm3

    world = sim.world
    world.size = [10 * m, 10 * m, 10 * m]
    world.material = "G4_AIR"

    mat_db = sim.volume_manager.material_database
    mat_db.add_material_nb_atoms("G4_PLEXIGLASS", ["C", "H", "O"], [5, 8, 2], 1.19 * gcm3)
    mat_db.add_material_nb_atoms("G4_WATER", ["H", "O"], [2, 1], 1.0 * gcm3)

    patient = sim.add_volume("Box", "patient")
    patient.size = [20 * cm, 20 * cm, 20 * cm]
    patient.translation = [0.0 * cm, 0.0 * cm, 0.0 * cm]
    patient.material = "G4_WATER"
    patient.color = [-5, 0, 1, 1]

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.set_production_cut("world", "gamma", 1 * mm)
    sim.physics_manager.set_production_cut("world", "electron", 1 * mm)
    sim.physics_manager.set_production_cut("world", "positron", 1 * mm)
    sim.physics_manager.set_production_cut("world", "proton", 1 * mm)
    sim.physics_manager.set_production_cut("patient", "gamma", 15 * mm)
    sim.physics_manager.set_production_cut("patient", "electron", 5 * mm)
    sim.physics_manager.set_production_cut("patient", "positron", 10 * mm)
    sim.physics_manager.set_production_cut("patient", "proton", 10 * mm)

    source = sim.add_source("GenericSource", "mysource")
    source.particle = "e-"
    source.energy.type = "spectrum_discrete"
    source.energy.spectrum_energies = [116, 117, 118, 119, 120, 121, 122, 123, 124]
    source.energy.spectrum_weights = [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]

    source.position.type = "disc"
    source.position.radius = 200 * mm

    radius = -50 * cm
    angle_rad = np.deg2rad(args.gantry)
    source_x = 0.0
    source_y = radius * np.sin(angle_rad)
    source_z = radius * np.cos(angle_rad)
    source.position.translation = [source_x, source_y, source_z]

    spot_x = args.spot[0] * cm
    spot_y = args.spot[1] * cm
    spot_z = 50 * cm

    theta = np.arctan2(spot_y, spot_x)
    phi = np.arctan2(np.sqrt(spot_x**2 + spot_y**2), spot_z)

    momentum = np.array([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ])

    rot = Rotation.from_euler("x", -args.gantry, degrees=True)
    rotated_momentum = rot.apply(momentum)
    source.direction.type = "momentum"
    source.direction.momentum = rotated_momentum.tolist()
    source.n = 10_000_000

    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True
    stats.output_filename = "stats.txt"

    depth_dose = sim.add_actor("DoseActor", "dose")
    depth_dose.attached_to = "patient"
    depth_dose.output_filename = f"dose3d{args.id}.mhd"
    depth_dose.spacing = np.array([0.586, 0.586, 1.0])
    depth_dose.size = np.array([512, 512, 376])
    depth_dose.translation = [2 * mm, 3 * mm, -2 * mm]
    depth_dose.hit_type = "random"
    depth_dose.dose.active = True

    sim.run()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
