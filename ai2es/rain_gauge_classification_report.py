import pandas as pd
import os


def read_csv(filename):
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


def main(root_dir):
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
