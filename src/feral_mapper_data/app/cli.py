"""Command line entry point: ``feral-edges CONFIG.toml``."""

from __future__ import annotations

import argparse
import logging
import sys

from feral_mapper_data.app.config import ConfigError, PipelineConfig
from feral_mapper_data.app.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert geospatial layers onto graph edges.")
    parser.add_argument("config", help="Pipeline TOML config.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = PipelineConfig.from_toml(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
