import argparse
import json
import os

import pywt
from gbm import simulate_gbm
from jsonl_utils import save_dataset
from transformations import cosine_transform, log_return
from wavelet_transformations import wavelet_transform


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def main(format_, config_path, data_dir="data", transformations=None, normalization=None):
    config = load_config(config_path)
    simulation_parameters = config["simulation_parameters"]
    gbm_parameters = config["gbm_parameters"]

    i = 0
    for sim_param in simulation_parameters:
        for gbm_param in gbm_parameters:
            n = sim_param["n"]
            St = simulate_gbm(dt=1 / n, **sim_param, **gbm_param)
            dir_path = f"{data_dir}"
            os.makedirs(dir_path, exist_ok=True)
            save_dataset(f"{dir_path}/gbm-{i}", St, format_)
            for transformation in transformations:
                if transformation == "cosine":
                    St_cos = cosine_transform(St)
                    if normalization == 'zscore':
                        mean = St_cos.mean(axis=0)
                        std = St_cos.std(axis=0)
                        St_cos = (St_cos - mean) / std
                        zscore_params = {
                            "mean": mean.mean(),
                            "std": std.mean(),
                        }
                        zscore_params_f = f"{dir_path}/zscore-gbm-{i}-params.json"
                        with open(zscore_params_f, "w") as f:
                            json.dump(zscore_params, f, indent=4)

                    save_dataset(f"{dir_path}/cos-gbm-{i}", St_cos, format_)
                elif transformation == "log-return":
                    log_returns = log_return(St)
                    save_dataset(f"{dir_path}/log-return-gbm-{i}", log_returns, format_)
                else:
                    if transformation in pywt.wavelist():
                        coeffs, coeffs_shapes = wavelet_transform(St, transformation)
                        save_dataset(
                            f"{dir_path}/{transformation}-gbm-{i}", coeffs, format_
                        )
                        wavelet_params_f = (
                            f"{dir_path}/{transformation}-gbm-{i}-params.json"
                        )
                        wavelet_params = {
                            "coeffs_shapes": coeffs_shapes[0].flatten().tolist(),
                            "sequence_length": coeffs.shape[0],
                            "wavelet_name": transformation,
                        }
                        with open(wavelet_params_f, "w") as f:
                            json.dump(wavelet_params, f, indent=4)

            parameters = {
                "simulation_parameters": sim_param,
                "gbm_parameters": gbm_param,
            }
            param_filename = f"{dir_path}/gbm-{i}-params.json"
            with open(param_filename, "w") as f:
                json.dump(parameters, f, indent=4)

            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate GBM and save datasets in specified format."
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        help="The format to save the dataset (e.g., 'diffusionts' or 'tsdiff').",
    )
    parser.add_argument(
        "-c", "--config_path", type=str, help="The path to the configuration file."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="data",
        help="The directory to save the datasets.",
    )
    parser.add_argument(
        "-t",
        "--transformations",
        nargs="+",
        default=[],
        help="The transformations to apply to the dataset.",
    )
    parser.add_argument(
        "-n",
        "--normalization",
        type=str,
        default=None,
        help="The normalization to apply to the dataset.",
    )
    args = parser.parse_args()

    main(args.format, args.config_path, args.data_dir, args.transformations, args.normalization)
