"""Read df with:
    two class human labels (precip and no precip, removed obstructed),
    three class human labels (precip obstructed and no precip),
    two class rain gauge (obstructed mixed into precip and no precip),
    and three class joint human/rain gauge (human labeled obs but rain gauge labeled precip and no precip)"""
import os

import pandas as pd


def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        filename,
        names=[
            "Labeling Strategy",
            "Precision",
            "Recall",
            "F1-score",
            "kfold",
            "Model Architecture",
        ],
    )


def main(root_dir: str) -> pd.DataFrame:
    two_gauge = read_csv(
        os.path.join(
            root_dir,
            "val_metrics_e50_bs64_k3_9model(s)_2class_rain_gauge_copy.csv",
        )
    )
    three_gauge = read_csv(
        os.path.join(
            root_dir,
            "val_metrics_e50_bs64_k3_9model(s)_3class_rain_gauge_copy.csv",
        )
    )
    two_human = read_csv(
        os.path.join(
            root_dir,
            "val_metrics_e50_bs64_k3_9model(s)_2class_human_copy.csv",
        )
    )
    three_human = read_csv(
        os.path.join(
            root_dir,
            "val_metrics_e50_bs64_k3_9model(s)_3class_human_copy.csv",
        )
    )
    return pd.concat([two_human, three_human, two_gauge, three_gauge])
