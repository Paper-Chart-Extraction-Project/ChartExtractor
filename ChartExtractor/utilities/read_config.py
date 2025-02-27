"""Reads the object detection model config yaml file."""

# Built-in Imports
from pathlib import Path
import sys
from typing import Dict

# External Imports
import yaml


def read_config() -> Dict:
    """Reads the object detection model config yaml file.

    Returns:
        A dictionary mapping constant names to file names of model weights
        and the tile size proportion that the model was trained on.
    """
    try:
        config_path: Path = Path(__file__) / ".." / ".." / ".." / "config.yaml"
        data: str = open(config_path.resolve(), "r").read()
        parsed_data: Dict = yaml.load(data, Loader=yaml.Loader)
        return parsed_data
    except Exception as e:
        err_msg = "An exception has occured, ensure that config.yaml is "
        err_msg += "correctly formatted and is at the root of the "
        err_msg += "ChartExtractor package."
        print(err_msg)
        print("Exact exception:")
        print(e)
        sys.exit()
