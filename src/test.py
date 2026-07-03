import os
import pandas as pd
import subprocess
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Produces test set predictions with the defined checkpoints and sends them
    to an evaluation server.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.checkpoints

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.module)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    for rel_ckpt_path in cfg.checkpoints:
        # Load checkpoint
        ckpt_path = os.path.join(cfg.paths.log_dir, rel_ckpt_path)
        ckpt = torch.load(ckpt_path, weights_only=False)

        # If the model was compiled, we need to strip the "_orig_mod" prefix from the dict keys
        renamed_ckpt = {}
        for k, v in ckpt["state_dict"].items():
            new_k = k.replace("_orig_mod.", "")
            renamed_ckpt[new_k] = v
        model.load_state_dict(renamed_ckpt)

        # Run on test set
        trainer.test(model=model, datamodule=datamodule)
        metric_dict = trainer.callback_metrics

        # Save metrics
        rel_ckpt_dir = rel_ckpt_path.split(os.sep)[:-2]  # Strip '/checkpoints/<checkpoint name>'
        #prediction_subdir = os.path.join(cfg.paths.prediction_dir, "valid", *rel_ckpt_dir)
        #os.makedirs(prediction_subdir, exist_ok=True)
        #prediction_path = os.path.join(prediction_subdir, "metrics.txt")
        #with open(prediction_path, 'w') as f:
        #    for k, v in metric_dict.items():
        #        f.write(f"{k}: {v}\n")

        # Print time, energy and emissions
        experiment_name = rel_ckpt_path.split(os.sep)[0]
        emissions_path = os.path.join(cfg.paths.codecarbon_dir, *rel_ckpt_dir, "emissions.csv")
        
        if os.path.isfile(emissions_path):
            with open(emissions_path, 'r') as f:
                df = pd.read_csv(f)
            energy_kwh = df.iloc[-1]['energy_consumed']
            duration_s = df.iloc[-1]['duration']
            emissions_kgco2e = df.iloc[-1]['emissions']
            print(f"Energy: {energy_kwh:.2f} kWh")
            print(f"Time: {duration_s:.2f} s")
            print(f"Emissions: {emissions_kgco2e * 1e3:.2f} gCO2eq")
        else:
            print(f"Couldn't find {emissions_path}!")
        
        
        # Send to evaluation server
        #dest_dir = f"172.22.11.44::eval_server/valid/{cfg.team_name}/{experiment_name}/"
        #subprocess.call(["rsync", "-avz", "--mkpath", prediction_path, f"{dest_dir}metrics.txt"])
        #subprocess.call(["rsync", "-avz", "--mkpath", emissions_path, f"{dest_dir}emissions.csv"])

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()