import json
import typing
import os
import os.path as osp

import yaml
from datasets import load_from_disk


def read_yaml_config(config_file_path: str) -> dict:
    with open(config_file_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)