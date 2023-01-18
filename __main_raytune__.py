"""hyperparameterize one model using ray tune; configured in cocpit/config.py"""
import cocpit

import cocpit.config as config  # isort: split
import os
from typing import Any, Dict

from ray import tune
from ray.tune.schedulers import ASHAScheduler


def model_setup(
    f: cocpit.fold_setup.FoldSetup, config: Dict[str, Any]
) -> None:
    """
    Create instances for model configurations and training/validation.
    Runs model.

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        config (Dict[str, str]): raytune configurations
    """
    m = cocpit.models.Model()
    # call method based on str model name
    method = getattr(cocpit.models.Model, config["MODEL_NAMES"])
    method(m)

    c = cocpit.model_config.ModelConfig(m.model)
    c.set_optimizer(lr=config["LR"])
    c.set_criterion()
    c.set_dropout(drop_rate=config["DROP_RATE"])
    c.to_device()
    cocpit.runner.main(
        f,
        c,
        config["MODEL_NAMES"],
        config["MAX_EPOCHS"],
        kfold=0,
    )


def train_models(config) -> None:
    """
    Train ML models by looping through all batch sizes, models, epochs, and/or folds
    """
    f = cocpit.fold_setup.FoldSetup(config["BATCH_SIZE"], 0, [], [])
    f.nofold_indices()
    f.split_data()
    f.create_dataloaders()
    model_setup(f, config)


def ray_tune_hp_search():

    CONFIG_RAY = {
        "BATCH_SIZE": tune.choice(config.BATCH_SIZE_TUNE),
        "MODEL_NAMES": tune.choice(config.MODEL_NAMES_TUNE),
        "LR": tune.choice(config.LR_TUNE),
        "WEIGHT_DECAY": tune.choice(config.WEIGHT_DECAY_TUNE),
        "DROP_RATE": tune.choice(config.DROP_RATE_TUNE),
        "MAX_EPOCHS": tune.choice(config.MAX_EPOCHS_TUNE),
    }

    scheduler = ASHAScheduler(max_t=50, grace_period=1, reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_models),
        resources_per_trial={"cpu": config.NUM_WORKERS, "gpu": 2},
        config=CONFIG_RAY,
        metric="loss",
        mode="min",
        num_samples=10,
        scheduler=scheduler,
    )
    result.results_df.to_csv("ray_tune_dfs.csv")

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(
        f'Best trial final validation loss: {best_trial.last_result["loss"]}'
    )
    print(
        "Best trial final validation accuracy:"
        f' {best_trial.last_result["accuracy"]}'
    )


if __name__ == "__main__":
    ray_tune_hp_search()
