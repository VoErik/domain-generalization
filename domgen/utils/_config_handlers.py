import yaml
import json
from types import SimpleNamespace


def config_to_namespace(file_path):
    with open(file_path, 'r') as file:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            data = yaml.safe_load(file)
        elif file_path.endswith('.json'):
            data = json.load(file)
        else:
            raise ValueError("Unsupported file type. Please provide a .yaml, .yml, or .json file.")

    def dict_to_namespace(d):
        obj = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                value = dict_to_namespace(value)
            setattr(obj, key, value)
        return obj

    return dict_to_namespace(data)


def merge_namespace(base, override):
    for key, value in override.__dict__.items():
        if value is not None:
            setattr(base, key, value)
    return base
