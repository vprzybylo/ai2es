"""main file to run for all training"""
import cocpit
import cocpit.config as config  # isort: split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os


def make_output_dirs():
    for directory in [
        config.FINAL_DIR,
        config.MODEL_SAVE_DIR,
        config.VAL_LOADER_SAVE_DIR,
        config.ACC_SAVE_DIR,
        config.PLOT_DIR,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def nested_kfold_runner(
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
    for k_outer, (train_indices, test_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", k_outer)

        # apply appropriate transformations for training and validation sets
        f = cocpit.fold_setup.FoldSetup(
            model_name, k_outer, train_indices, test_indices
        )
        f.split_data()
        f.update_save_names()

        c = cocpit.tune.model_setup(model_name)
        best_trial = train_inner(f, c, model_name, k_outer)

        (
            test_acc,
            test_uncertainties,
            test_probs,
            test_labels,
            test_preds,
        ) = train_outer(best_trial, f, c, model_name, k_outer)
        test_accs.append(test_acc)
        record_performance(
            model_name,
            k_outer,
            test_uncertainties,
            test_probs,
            test_labels,
            test_preds,
        )
    return np.mean(test_accs)


def train_inner(f, c, model_name, k_outer):
    # Wrap the objective inside a lambda and call objective inside it
    func = lambda trial: cocpit.tune.train_val_kfold_split(
        trial,
        f,
        c,
        model_name,
        k_outer,
    )
    best_trial = cocpit.tune.inner_kfold_tune(model_name, func)
    return best_trial


def train_outer(best_trial, f, c, model_name, k_outer):
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
    ) = cocpit.tune.run_after_split(
        f,
        c,
        model_name,
        k_outer,
        best_trial.params["epochs"],
        best_trial.params["batch_size"],
    )
    return test_acc, test_uncertainties, test_probs, test_labels, test_preds


def record_performance(
    model_name, k_outer, uncertainties, probs, labels, preds
):
    """record performance plots and uncertainties"""
    r = cocpit.plotting_scripts.report.Report(
        uncertainties, probs, labels, preds
    )
    r.conf_matrix(labels, preds)
    r.class_report(model_name, labels, preds, k_outer)
    r.uncertainty_prob_scatter(probs, uncertainties)
    r.hist(probs, f"{config.PLOT_DIR}/histogram_probs.png")
    r.hist(uncertainties, f"{config.PLOT_DIR}/histogram_uncertainties.png")


if __name__ == "__main__":
    overall_model_accs = []
    make_output_dirs()
    for model_name in config.MODEL_NAMES_TUNE:
        avg_test_acc = nested_kfold_runner(model_name)
        overall_model_accs.append(avg_test_acc)
    print(
        "The best overall model is"
        f" {config.MODEL_NAMES_TUNE[np.argmax(overall_model_accs)]}"
    )
