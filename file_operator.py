import json


def read_json():
    with open("configs.json", "r") as f:
        json_open = json.load(f)
    return json_open
