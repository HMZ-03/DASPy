"""Console entry point for DASPy.

DASPy is primarily used as a Python library. The console script is kept for
packaging completeness and for lightweight environment checks.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version


def _package_version() -> str:
    """Return the installed package version, or a source-tree fallback."""
    try:
        return version("DASPy-toolbox")
    except PackageNotFoundError:
        return "1.2.3"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="daspy",
        description=(
            "DASPy is a Python package for Distributed Acoustic Sensing data "
            "processing. Use it from Python via 'import daspy'."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"DASPy {_package_version()}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DASPy console entry point."""
    build_parser().parse_args(argv)
    print("DASPy is intended to be used as a Python library. Try: from daspy import read")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
