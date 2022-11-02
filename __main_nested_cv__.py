"""main file to run for all training"""
import cocpit

import cocpit.config as config  # isort: split
from sklearn.model_selection import StratifiedKFold


def kfold_training(
    model_name: str,
) -> None:
    """
    - Split dataset into folds
    - Preserve the percentage of samples for each class with stratified
    - Create dataloaders for each fold

    Args:
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
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
            model_name, kfold, train_indices, val_indices
        )
        f.split_data()
        f.update_save_names()

        c = model_setup(model_name)
        cocpit.runner.main(
            f,
            c,
            model_name,
            kfold=kfold,
        )


def model_setup(model_name: str) -> cocpit.model_config.ModelConfig:
    """
    Create instances for model configurations and training/validation.
    Runs model.

    Args:
        model_name (str): name of model architecture
    """
    m = cocpit.models.Model()
    # call method based on str model name
    method = getattr(cocpit.models.Model, model_name)
    method(m)

    c = cocpit.model_config.ModelConfig(m.model)
    c.set_optimizer()
    c.to_device()
    return c


if __name__ == "__main__":
    for model_name in config.MODEL_NAMES:
        kfold_training()
