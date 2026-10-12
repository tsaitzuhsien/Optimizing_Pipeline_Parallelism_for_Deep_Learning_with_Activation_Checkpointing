import yaml
import argparse
from typing import Any


def parse_positive_int_csv(value: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("expansion pattern must be text")

    elements = value.split(",")
    if not value.strip() or any(not element.strip() for element in elements):
        raise argparse.ArgumentTypeError(
            "expansion pattern must be a nonempty comma-separated list"
        )

    try:
        pattern = tuple(int(element) for element in elements)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expansion pattern values must be integers"
        ) from error

    if any(ratio <= 0 for ratio in pattern):
        raise argparse.ArgumentTypeError(
            "expansion pattern values must be positive"
        )
    return pattern


class LoadConfig(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # values is the path to the config file
        with open(values, "r") as f:
            cfg = yaml.safe_load(f)
        setattr(namespace, self.dest, cfg)
        for key, val in cfg.items():
            setattr(namespace, key, val)
