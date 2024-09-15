import json
from datetime import date

import numpy as np


def save_dataset(filename, St, format_):
    if format_ == "diffusionts":
        np.savetxt(f"{filename}.csv", St, delimiter=",")
    elif format_ == "tsdiff":
        dataset = []
        for path in St.T:
            dataset.append({"start": str(date(2000, 1, 1)), "target": list(path)})
        with open(f"{filename}.jsonl", "w") as file:
            for path in dataset:
                file.write(json.dumps(path) + "\n")


def extract_targets_from_jsonl(file_path):
    targets = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            targets.append(data["target"])
    return np.array(targets).T
