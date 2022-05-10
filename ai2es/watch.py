from cocpit import config as config
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cocpit
import numpy as np
import torch
from datetime import datetime
import csv


class MonitorFolder(FileSystemEventHandler):
    def __init__(self, w):
        self.w = w

    def on_created(self, event):

        test_data = cocpit.data_loaders.TestDataSet(
            open_dir="", file_list=[event.src_path]
        )
        test_loader = cocpit.data_loaders.create_loader(
            test_data, batch_size=100, sampler=None
        )
        for imgs, _ in test_loader:
            b = cocpit.predictions.BatchPredictions(imgs, torch.load(config.MODEL_PATH))
            with torch.no_grad():
                b.find_max_preds()
                b.top_k_preds(top_k_preds=3)
                self.w.writerow(
                    [
                        event.src_path,
                        config.CLASS_NAMES[np.argmax(b.probs)],
                        b.probs[0],
                        b.probs[1],
                        b.probs[2],
                    ]
                )


def write_csv_header():
    with open("precip_predictions.csv", "a", newline="") as csvfile:
        w = csv.writer(csvfile, delimiter=",")
        w.writerow(
            [
                "filename",
                "top class",
                "no precipitation",
                "obstructed",
                "precipitation",
            ]
        )
    return w


def path_to_check():
    current_date = datetime.now().strftime("%Y/%m/%d")
    print(type(current_date))
    print(type(f"/ai2es/cam_photos/{current_date}/"))
    return f"/ai2es/cam_photos/{current_date}/"
    # check_path = f"/ai2es/test_set/"


if __name__ == "__main__":

    w = write_csv_header()
    observer = Observer()
    observer.schedule(MonitorFolder(w), path=path_to_check(), recursive=True)
    print("Monitoring started")
    observer.start()
    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()
        observer.join()
