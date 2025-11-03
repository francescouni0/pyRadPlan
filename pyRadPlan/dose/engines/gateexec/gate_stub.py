#!/usr/bin/env python3
"""Gate/OpenGATE stub simulation.

This script mirrors the structure of a Gate 10 Python driver so the
`GateMonteCarloEngine` can invoke it during development.  It builds a minimal
scene (water phantom, electron source, dose actor) and emits an MHD dose file
using the OpenGATE Python API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--beam",
        action="append",
        nargs=3,
        type=float,
        metavar=("SPOT_X_CM", "SPOT_Y_CM", "GANTRY_DEG"),
        help="Beam definition; supply multiple times for multiple beams.",
    )
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

    ct_path = Path.cwd() / "input" / "ct.mhd"

    patient = sim.add_volume("Image", name="patient")
    patient.image = str(ct_path)
    patient.mother = "world"
    patient.material = "G4_AIR"
    patient.rotation = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    patient.voxel_materials = [
        [-2000, -900, "G4_AIR"],
        [-900, -100, "G4_LUNG_ICRP"],
        [-100, 0, "G4_ADIPOSE_TISSUE_ICRP"],
        [0, 300, "G4_TISSUE_SOFT_ICRP"],
        [300, 800, "G4_B-100_BONE"],
        [800, 6000, "G4_BONE_COMPACT_ICRU"],
    ]
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

    if args.beam:
        beam_definitions = [(float(b[0]), float(b[1]), float(b[2])) for b in args.beam]
    else:
        spot_vals = args.spot if len(args.spot) >= 2 else [0.0, 0.0]
        beam_definitions = [(float(spot_vals[0]), float(spot_vals[1]), float(args.gantry))]

    for beam_idx, (spot_x_cm, spot_y_cm, gantry_deg) in enumerate(beam_definitions):
        source = sim.add_source("GenericSource", f"mysource_{beam_idx}")
        source.particle = "e-"
        source.energy.type = "spectrum_discrete"
        source.energy.spectrum_energies = [116, 117, 118, 119, 120, 121, 122, 123, 124]
        source.energy.spectrum_weights = [0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]

        source.position.type = "disc"
        source.position.radius = 2 * mm

        radius = 100 * cm
        base_source = np.array([0.0, 0.0, -radius])
        rot = Rotation.from_euler("y", gantry_deg, degrees=True)
        source_position = rot.apply(base_source)
        source.position.translation = source_position.tolist()

        spot_vec = np.array([spot_x_cm * cm, spot_y_cm * cm, 0.0])
        direction = spot_vec - source_position
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction /= norm

        source.direction.type = "momentum"
        source.direction.momentum = direction.tolist()
        source.n = 10000

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
