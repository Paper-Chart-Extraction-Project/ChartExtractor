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
    config_path: Path = Path(__file__) / ".." / ".." / ".." / "config.yaml"
    potential_err_msg = "An exception has occured, ensure that config.yaml is "
    potential_err_msg += "correctly formatted and is at the root of the "
    potential_err_msg += "ChartExtractor package."
    config_data: Dict = read_yaml_file(config_path, potential_err_msg)


def read_yaml_file(filepath: Path, err_msg) -> Dict:
    """Reads a yaml file. Raises slightly more helpful exceptions.
    
    Args:
        filepath (Path):
            The path to the yaml file.

    Raises:
        Any exception relating to loading and parsing a yaml file.

    Returns:
        The data within the yaml file as a dictionary.
    """
    try:
        data: str = open(filepath, "r").read()
        parsed_data: Dict = yaml.load(data, Loader=yaml.Loader)
        return parsed_data
    except Exception as e:
        print(err_msg)
        print("Exact exception:")
        print(e)
        sys.exit()
