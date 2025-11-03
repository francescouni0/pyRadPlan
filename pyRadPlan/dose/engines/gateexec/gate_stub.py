#!/usr/bin/env python3
"""
Minimal Gate stub.

This module stands in for the external Gate executable during development and
testing. It mimics the command-line interface of a Gate binary but does not
perform any Monte Carlo simulation. Instead, it writes a short log describing
the invocation and exits successfully.
"""

from __future__ import annotations

import argparse
import json
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("gate_stub")
    parser.add_argument(
        "--macro",
        metavar="FILE",
        help="Path to the Gate macro file that would normally be executed.",
    )
    parser.add_argument(
        "--output-log",
        metavar="FILE",
        help="Optional path for a JSON log of the invocation.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Requested number of threads (ignored by the stub).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random-number seed (ignored by the stub).",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Any extra arguments Gate would receive.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.output_log:
        payload = {
            "macro": args.macro,
            "threads": args.threads,
            "seed": args.seed,
            "extra": args.extra,
        }
        with open(args.output_log, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
