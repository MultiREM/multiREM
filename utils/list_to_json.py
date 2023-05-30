import json


def list_to_json(rows, file_path):
    with open(file_path, 'w') as file:
        json.dump(rows, file, indent=4)
    return True