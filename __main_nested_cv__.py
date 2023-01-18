"""main file to run for all training"""
import cocpit

import cocpit.config as config  # isort: split
import csv
import os
from typing import Any, List

import numpy as np


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


def main():
    overall_model_accs = []
    make_output_dirs()
    filename = "overall_model_accs.txt"
    for model_name in config.MODEL_NAMES_TUNE:

        outer_k = cocpit.tune.KfoldOuter()
        avg_test_acc = outer_k.kfold_outer_runner(model_name)
        overall_model_accs.append(avg_test_acc)
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, avg_test_acc])
        file.close()
    print(
        "The best overall model is"
        f" {config.MODEL_NAMES_TUNE[np.argmax(overall_model_accs)]}"
    )


if __name__ == "__main__":
    main()
