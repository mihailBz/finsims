import json
import os
from datetime import date

import numpy as np
from gbm import simulate_gbm


def main(format_):
    simulation_parameters = [
        # {'n': 1000, 'M': 1},
        # {'n': 5000, 'M': 1},
        {"n": 10000, "M": 1},
    ]

    gbm_parameters = [
        {"mu": 0.0, "sigma": 0.2},
        {"mu": 0.2, "sigma": 0.0},
        {"mu": 0.05, "sigma": 0.1},
        {"mu": 0.05, "sigma": 0.2},
        {"mu": 0.05, "sigma": 0.3},
        {"mu": 0.05, "sigma": 0.4},
        {"mu": 0.1, "sigma": 0.1},
        {"mu": 0.1, "sigma": 0.2},
        {"mu": 0.1, "sigma": 0.3},
        {"mu": 0.1, "sigma": 0.4},
        {"mu": 0.2, "sigma": 0.1},
        {"mu": 0.2, "sigma": 0.2},
        {"mu": 0.2, "sigma": 0.3},
        {"mu": 0.2, "sigma": 0.4},
        {"mu": 0.5, "sigma": 0.1},
        {"mu": 0.5, "sigma": 0.2},
        {"mu": 0.5, "sigma": 0.3},
        {"mu": 0.5, "sigma": 0.4},
    ]

    i = 1
    for sim_param in simulation_parameters:
        for gbm_param in gbm_parameters:
            St = simulate_gbm(S0=100, T=1, **sim_param, **gbm_param)
            dir_path = f"{format_}_dataset"
            os.makedirs(dir_path)
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
    main("diffusionts")
