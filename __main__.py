import cocpit

from ai2es import config as config  # isort: split
import os
import time

import pandas as pd
import torch


def build_model():
    """
    train ML models
    """
    # loop through batch sizes, models, epochs, and/or folds
    for batch_size in config.BATCH_SIZE:
        print("BATCH SIZE: ", batch_size)
        for model_name in config.MODEL_NAMES:
            print("MODEL: ", model_name)
            for epochs in config.MAX_EPOCHS:
                print("MAX EPOCH: ", epochs)

                cocpit.setup_training.main(
                    batch_size,
                    model_name,
                    epochs,
                )


def classification():
    """
    classify images using the ML model
    """
    print("running ML model to classify ice...")

    start_time = time.time()

    # load ML model for predictions
    model = torch.load(config.MODEL_PATH)

    # load df of quality ice particles to make predictions on
    df = pd.read_csv(df_path)
    df = cocpit.run_model.main(df, model)  # remove open_dir from run_model
    # df.to_csv(df_path, index=False)

    print("done classifying all images!")
    print("time to classify ice = %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":

    print(
        "num workers in loader = {}".format(config.NUM_WORKERS)
    ) if config.CLASSIFICATION or config.BUILD_MODEL else print(
        "num cpus for parallelization = {}".format(config.NUM_WORKERS)
    )

    # only run one arbitrary year in loop if building model
    years = [2018] if config.BUILD_MODEL else [2018, 2019, 2020, 2021]
    for year in years:
        print("years: ", year)

        # create dir for final databases
        outname = f"{year}.parquet"

        df_path = os.path.join(config.FINAL_DIR, outname)

        if config.BUILD_MODEL:
            build_model()

        if config.CLASSIFICATION:
            classification()
