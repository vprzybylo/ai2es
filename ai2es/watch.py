from cocpit import config as config
import time
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver
import cocpit
import numpy as np
import torch
from datetime import datetime
import csv
from PIL import Image


class MonitorFolder(FileSystemEventHandler):
    """
    Run CNN when a new image comes into directory

    Args:
        w (_csv._writer): output file to write to
        b (cocpit.predictions.BatchPredictions): predictions for an image
    """

    def __init__(self, w):
        self.w = w
        self.b = None

    def check_night_image(self, filename):
        """
        Only make a prediction on image at night

        Args:
            filename (str): path to file to open
        Returns
            (bool): True if night else False
        """
        image = np.asarray(Image.open(filename))
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return bool((b == g).all() and (b == r).all())

    def write(self, filename):
        """
        Write probability for each class out to a csv

        Args:
            filename (str): path to image
        """
        self.w.writerow(
            [
                filename,
                config.CLASS_NAMES[np.argmax(self.b.probs)],
                self.b.probs[0],
                self.b.probs[1],
                self.b.probs[2],
            ]
        )

    def on_created(self, event):
        """
        Overrides FileSystemEventHandler and what to do when file created
        Creates a dataloader and makes pred

        Args:
            event (FileCreatedEvent): Event representing file/directory creation.
        """
        test_data = cocpit.data_loaders.TestDataSet(
            open_dir="", file_list=[event.src_path]
        )

        if self.check_night_image(event.src_path):
            test_loader = cocpit.data_loaders.create_loader(
                test_data, batch_size=100, sampler=None
            )
            for imgs, _ in test_loader:
                self.b = cocpit.predictions.BatchPredictions(
                    imgs, torch.load(config.MODEL_PATH)
                )
                with torch.no_grad():
                    self.b.find_max_preds()
                    self.b.top_k_preds(top_k_preds=3)

                self.write(event)


def path_to_check():
    """Directory to monitor"""
    current_date = datetime.now().strftime("%Y/%m/%d")
    print(f"/ai2es/cam_photos/{current_date}/")
    return f"/ai2es/cam_photos/{current_date}/"
    # check_path = f"/ai2es/test_set/"


if __name__ == "__main__":

    # write file header first
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

        print("Monitoring started")
        observer = PollingObserver()
        observer.schedule(MonitorFolder(w), path=path_to_check(), recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            observer.stop()
            observer.join()
