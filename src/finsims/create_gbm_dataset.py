import json
import os
from datetime import date

import numpy as np
from gbm import simulate_gbm


def main(format_):
    simulation_parameters = [
        {"n": 1000, "M": 1000},
    ]

    gbm_parameters = [
        {"mu": 0.0, "sigma": 0.4},
        {"mu": 0.0, "sigma": 0.6},
        {"mu": 0.0, "sigma": 0.8},
        {"mu": 0.0, "sigma": 1.0},
        {"mu": 0.1, "sigma": 0.4},
        {"mu": 0.1, "sigma": 0.6},
        {"mu": 0.1, "sigma": 0.8},
        {"mu": 0.1, "sigma": 1.0},
        {"mu": 0.3, "sigma": 0.4},
        {"mu": 0.3, "sigma": 0.6},
        {"mu": 0.3, "sigma": 0.8},
        {"mu": 0.3, "sigma": 1.0},
    ]

    i = 1
    for sim_param in simulation_parameters:
        for gbm_param in gbm_parameters:
            n = sim_param["n"]
            St = simulate_gbm(cos_transform=True, dt=1 / n, **sim_param, **gbm_param)
            dir_path = f"{format_}_dataset"
            os.makedirs(dir_path, exist_ok=True)
            save_dataset(f"./{dir_path}/gbm-{i}", St, format_)

            parameters = {
                "simulation_parameters": sim_param,
                "gbm_parameters": gbm_param,
            }
            param_filename = f"./{dir_path}/gbm-{i}-params.json"
            with open(param_filename, "w") as f:
                json.dump(parameters, f, indent=4)

            i += 1


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


if __name__ == "__main__":
    main("tsdiff")
