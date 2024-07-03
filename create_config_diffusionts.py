import yaml


def create_yaml_config(dataset_number):
    # Set the basic structure of the YAML content
    config = {
        "model": {
            "target": "Models.interpretable_diffusion.gaussian_diffusion.Diffusion_TS",
            "params": {
                "seq_length": 390,
                "feature_size": 1,
                "n_layer_enc": 2,
                "n_layer_dec": 2,
                "d_model": 64,
                "timesteps": 500,
                "sampling_timesteps": 500,
                "loss_type": "l1",
                "beta_schedule": "cosine",
                "n_heads": 4,
                "mlp_hidden_times": 4,
                "attn_pd": 0.0,
                "resid_pd": 0.0,
                "kernel_size": 1,
                "padding_size": 0
            }
        },
        "solver": {
            "base_lr": 1.0e-5,
            "max_epochs": 10000,
            "results_folder": f"./Checkpoints_gbm-{dataset_number}",
            "gradient_accumulate_every": 2,
            "save_cycle": 1000,
            "ema": {
                "decay": 0.995,
                "update_interval": 10
            },
            "scheduler": {
                "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params": {
                    "factor": 0.5,
                    "patience": 2000,
                    "min_lr": 1.0e-5,
                    "threshold": 1.0e-1,
                    "threshold_mode": "rel",
                    "warmup_lr": 8.0e-4,
                    "warmup": 500,
                    "verbose": False
                }
            }
        },
        "dataloader": {
            "train_dataset": {
                "target": "Utils.Data_utils.real_datasets.CustomDataset",
                "params": {
                    "name": "stock",
                    "proportion": 1.0,
                    "data_root": f"Data/datasets/diffusionts_dataset/gbm-{dataset_number}.csv",
                    "window": 390,
                    "save2npy": True,
                    "neg_one_to_one": True,
                    "seed": 123,
                    "period": "train"
                }
            },
            "test_dataset": {
                "target": "Utils.Data_utils.real_datasets.CustomDataset",
                "params": {
                    "name": "stock",
                    "proportion": 0.9,
                    "data_root": f"Data/datasets/diffusionts_dataset/gbm-{dataset_number}.csv",
                    "window": 390,
                    "save2npy": True,
                    "neg_one_to_one": True,
                    "seed": 123,
                    "period": "test",
                    "style": "separate",
                    "distribution": "geometric",
                    "coefficient": 1.0e-2,
                    "step_size": 5.0e-2,
                    "sampling_steps": 200
                }
            },
            "batch_size": 64,
            "sample_size": 256,
            "shuffle": True
        }
    }

    # Write YAML file
    file_name = f'config_gbm_{dataset_number}.yaml'
    with open(file_name, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def main():
    # Generate YAML files for datasets gbm-1 to gbm-18
    for i in range(1, 19):
        create_yaml_config(i)


if __name__ == '__main__':
    main()
