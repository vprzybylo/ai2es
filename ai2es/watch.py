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
import os
from typing import List, Tuple
import io


class MonitorFolder(FileSystemEventHandler):
    """
    Run CNN when a new image comes into directory

    Args:
        w (csv.writer): output file to write to
        csvfile (io.TextIOWrapper): open file to write to
        b (cocpit.predictions.BatchPredictions): predictions for an image
    """

    def __init__(self, csvfile: io.TextIOWrapper):
        self.csvfile = csvfile
        self.w = csv.writer(self.csvfile, delimiter=",")
        self.b: cocpit.predictions.BatchPredictions = None

    def check_night_image(self, filename: str) -> bool:
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

    def write_csv(self, event: FileSystemEventHandler):
        """
        Write probability for each class out to a csv

        Args:
            event (FileSystemEventHandler): event.src_path = path to new image
        """
        self.w.writerow(
            [
                event.src_path.split("/")[-1],
                config.CLASS_NAMES[self.b.classes[0]],
                f"{self.b.probs[0]* 100:.2f}",
                f"{self.b.probs[1]* 100:.2f}",
                f"{self.b.probs[2]* 100:.2f}",
            ]
        )
        self.csvfile.flush()

    def on_created(self, event: FileSystemEventHandler):
        """
        Overrides FileSystemEventHandler and what to do when file created
        Creates a dataloader and makes prediction

        Args:
            event (FileSystemEventHandler): Event representing file/directory creation.
        """
        # print(event.src_path)
        test_data = cocpit.data_loaders.TestDataSet(
            open_dir="",
            file_list=[event.src_path],
        )

        if self.check_night_image(event.src_path):
            test_loader = cocpit.data_loaders.create_loader(
                test_data, batch_size=1, sampler=None
            )
            for imgs, _ in test_loader:

                self.b = cocpit.predictions.BatchPredictions(
                    imgs, torch.load(config.MODEL_PATH)
                )
                with torch.no_grad():
                    self.b.find_max_preds()
                    self.b.top_k_preds(top_k_preds=3)
                # print(
                #     path,
                #     config.CLASS_NAMES[self.b.classes[0]],
                #     f"{self.b.probs[0]* 100:.2f}",
                #     f"{self.b.probs[1]* 100:.2f}",
                #     f"{self.b.probs[2]* 100:.2f}",
                # )

                self.write_csv(event)
                torch.cuda.empty_cache()


def current_date() -> str:
    """
    Current year/month/day for outfiles

    Returns:
        (str): current date down to day
    """
    return datetime.now().strftime("%Y/%m/%d")


def path_to_check(stn: str) -> str:
    """
    Directory to monitor

    Args:
        stn (str): station id
    Returns:
        (str): where images are getting fed into
    """
    # print(f"/ai2es/cam_photos/{current_date()}/{stn}")
    return f"/ai2es/cam_photos/{current_date()}/{stn}"


def csv_output_path() -> str:
    """
    Where to save csv output file

    Returns:
        (str): where predictions should be saved. Once daily.
    """
    if not os.path.exists(f"/ai2es/realtime_predictions/csv/{current_date()}/"):
        os.makedirs(f"/ai2es/realtime_predictions/csv/{current_date()}")
    return f"/ai2es/realtime_predictions/csv/{current_date()}/{current_date().replace('/', '_')}.csv"


def write_header(w) -> csv.writer:
    """
    open csv file and write header for columns

    Returns:
        w (_csv._writer): a writer object responsible for converting data to CSV format
    """

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


def observer_setup() -> Tuple[List[PollingObserver], io.TextIOWrapper, PollingObserver]:
    """
    Create observers to watch directories across all stations

    Returns:
        observers (List[PollingObserver]): list of observers across all stations
        csvfile (TextIOWrapper):  csv file to write preds to
        observer (PollingObserver): a PollingObserver instance
    """
    observer = PollingObserver()
    observers = []

    csvfile = open(csv_output_path(), "a", newline="")
    # w = write_header(w)

    for stn in config.stnid:
        observer.schedule(
            MonitorFolder(csvfile), path=path_to_check(stn), recursive=True
        )
        observers.append(observer)
    return (observers, csvfile, observer)


if __name__ == "__main__":

    observers, csvfile, observer = observer_setup()
    observer.start()
    print("Monitoring started: ", datetime.now().strftime("%Y/%m/%d/%H:%M:%S"))

    try:
        while True:
            # Poll every x seconds
            time.sleep(10)

    except KeyboardInterrupt:
        for o in observers:
            o.unschedule_all()

            # Stop observer if interrupted
            o.stop()
    finally:
        csvfile.close()
        for o in observers:
            # Wait until the thread terminates before exit
            o.join()
