import cocpit

import cocpit.config as config  # isort: split
import os
from typing import Any, Dict

from ray import tune
from ray.tune.schedulers import ASHAScheduler


def nofold_training(model_name, batch_size, epochs):
    f = cocpit.fold_setup.FoldSetup(model_name, batch_size, epochs)
    f.nofold_indices()
    f.split_data()
    f.create_dataloaders()
    optimizer, model = cocpit.model_config.main(model_name)
    cocpit.runner.main(f.dataloaders, optimizer, model, epochs, model_name, batch_size)


def model_setup(f: cocpit.fold_setup.FoldSetup, config: Dict[str, Any]) -> None:
    """
    Create instances for model configurations and training/validation.
    Runs model.

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
    """
    m = cocpit.models.Model()
    # call method based on str model name
    method = getattr(cocpit.models.Model, config["MODEL_NAMES"])
    method(m)

    c = cocpit.model_config.ModelConfig(m.model)
    c.set_optimizer(config)
    c.set_criterion()
    c.set_dropout(config)
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

    scheduler = ASHAScheduler(max_t=50, grace_period=1, reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_models),
        resources_per_trial={"cpu": config.NUM_WORKERS, "gpu": 2},
        config=config.config_ray,
        metric="loss",
        mode="min",
        num_samples=10,
        scheduler=scheduler,
    )
    result.results_df.to_csv("ray_tune_dfs.csv")

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')


if __name__ == "__main__":

    print(
        f"num workers in loader = {config.NUM_WORKERS}"
    ) if config.CLASSIFICATION or config.BUILD_MODEL else print(
        f"num cpus for parallelization = {config.NUM_WORKERS}"
    )

    # only run one arbitrary year in loop if building model
    years = [2018] if config.BUILD_MODEL else [2018, 2019, 2020, 2021]
    for year in years:
        print("years: ", year)

        # create dir for final databases
        outname = f"{year}.parquet"

        df_path = os.path.join(config.FINAL_DIR, outname)

        ray_tune_hp_search()
