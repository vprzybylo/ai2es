import cocpit

import cocpit.config as config  # isort: split

from sklearn.model_selection import StratifiedKFold


def nofold_training(model_name, batch_size, epochs):
    f = cocpit.fold_setup.FoldSetup(model_name, batch_size, epochs)
    f.nofold_indices()
    f.split_data()
    f.create_dataloaders()
    optimizer, model = cocpit.model_config.main(model_name)
    cocpit.runner.main(
        f.dataloaders,
        optimizer,
        model,
        epochs,
        model_name,
        batch_size,
    )


def kfold_training(
    batch_size: int,
    model_name: str,
    epochs: int,
) -> None:
    """
    - Split dataset into folds
    - Preserve the percentage of samples for each class with stratified
    - Create dataloaders for each fold

    Args:
        batch_size (int): number of images read into memory at a time
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
        kfold (int): number of folds
    """
    skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=42)
    # datasets based on phase get called again in split_data
    # needed here to initialize for skf.split
    data = cocpit.data_loaders.get_data("val")
    for kfold, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", kfold)

        # apply appropriate transformations for training and validation sets
        f = cocpit.fold_setup.FoldSetup(
            batch_size, kfold, train_indices, val_indices
        )
        f.split_data()
        f.update_save_names()
        f.create_dataloaders()
        model_setup(f, model_name, epochs, kfold)


def model_setup(
    f: cocpit.fold_setup.FoldSetup, model_name: str, epochs: int, kfold: int
) -> None:
    """
    Create instances for model configurations and training/validation.
    Runs model.

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
        kfold (int): number of folds
    """
    m = cocpit.models.Model()
    # call method based on str model name
    method = getattr(cocpit.models.Model, model_name)
    method(m)

    c = cocpit.model_config.ModelConfig(m.model)
    c.set_optimizer()
    c.set_criterion()
    c.to_device()
    cocpit.runner.main(
        f,
        c,
        model_name,
        epochs,
        kfold=kfold,
    )


def train_models() -> None:
    """
    Train ML models by looping through all batch sizes, models,
    epochs, and/or folds
    """
    for batch_size in config.BATCH_SIZE:
        print("BATCH SIZE: ", batch_size)
        for model_name in config.MODEL_NAMES:
            print("MODEL: ", model_name)
            for epochs in config.MAX_EPOCHS:
                print("MAX EPOCH: ", epochs)

                if config.KFOLD != 0:
                    # Setup k-fold cross validation on labeled dataset
                    kfold_training(batch_size, model_name, epochs)
                else:
                    f = cocpit.fold_setup.FoldSetup(batch_size, 0, [], [])
                    f.nofold_indices()
                    f.split_data()

                    f.create_dataloaders()
                    model_setup(f, model_name, epochs, kfold=0)


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

        if config.BUILD_MODEL:
            train_models()
