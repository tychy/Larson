import json


def read_json():
    with open("configs.json", "r") as f:
        config = json.load(f)
    return config
