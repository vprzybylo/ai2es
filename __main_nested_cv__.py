"""main file to run for all training"""
import cocpit

import cocpit.config as config  # isort: split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from cocpit.plotting_scripts import report as report


def outer_kfold_training(
    model_name: str,
) -> None:
    """
    - Split dataset into folds and loop through hold out test sets
    - Preserve the percentage of samples for each class with stratified
    - Create dataloaders for each fold

    Args:
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
    """
    test_accs = []
    skf = StratifiedKFold(
        n_splits=config.KFOLD_OUTER, shuffle=True, random_state=42
    )
    # datasets based on phase get called again in split_data
    # needed here to initialize for skf.split
    data = cocpit.data_loaders.get_data("val")
    for kfold, (train_indices, test_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", kfold)

        # apply appropriate transformations for training and validation sets
        f = cocpit.fold_setup.FoldSetup(
            model_name, kfold, train_indices, test_indices
        )
        f.split_data()
        f.update_save_names()

        c = cocpit.tune.model_setup(model_name)
        best_trial = cocpit.runner.inner_kfold_tune(
            best_trial, f, c, model_name, kfold
        )
        (
            test_acc,
            test_uncertainties,
            test_probs,
            test_labels,
            test_preds,
        ) = train_outer(best_trial)
        test_accs.append(test_acc)
        record_performance(
            model_name,
            kfold,
            test_uncertainties,
            test_probs,
            test_labels,
            test_preds,
        )
    return np.mean(test_accs)


def train_outer(best_trial, f, c, model_name, kfold):
    """
    outer kfold loop for train test/train split
    applies best hyperoptimization of inner loop (best_trial)
    """
    (
        test_acc,
        test_uncertainties,
        test_probs,
        test_labels,
        test_preds,
    ) = cocpit.tune.train_val(
        f,
        c,
        best_trial.params["epochs"],
        best_trial.params["batch_size"],
        model_name,
        kfold,  # outer k-fold cross validation index
    )
    return test_acc, test_uncertainties, test_probs, test_labels, test_preds


def record_performance(model_name, kfold, uncertainties, probs, labels, preds):
    """record performance plots and uncertainties"""
    r = report.Report(uncertainties, probs, labels, preds)
    r.conf_matrix(labels, preds)
    r.class_report(model_name, labels, preds, kfold)
    r.uncertainty_prob_scatter(probs, uncertainties)
    r.hist(probs, f"{config.PLOT_DIR}/histogram_probs.png")
    r.hist(uncertainties, f"{config.PLOT_DIR}/histogram_uncertainties.png")


if __name__ == "__main__":
    overall_model_accs = []
    for model_name in config.MODEL_NAMES:
        avg_test_acc = outer_kfold_training(model_name)
        overall_model_accs.append(avg_test_acc)
    print(
        "The best overall model is"
        f" {config.MODEL_NAMES_TUNE[np.argmax(overall_model_accs)]}"
    )
