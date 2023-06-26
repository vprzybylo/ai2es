"""Monitor as images are populating the Mesonet folders and make predictions
Runs indefinitely

TODO: put in cron job/k8s job"""
import csv
import io
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import cocpit
import numpy as np
import torch
from cocpit import config as config
from PIL import Image, UnidentifiedImageError
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver


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

    def check_night_image(self, filename: str) -> Optional[bool]:
        """
        Only make a prediction on image at night

        Args:
            filename (str): path to file to open
        Returns
            (bool): True if night else False
        """
        try:
            image = np.asarray(Image.open(filename))
            b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            return bool((b == g).all() and (b == r).all())
        except UnidentifiedImageError:
            print(f"couldn't find file: {filename}")

    def write_csv(self, event: FileSystemEventHandler) -> None:
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

    def on_created(self, event: FileSystemEventHandler) -> None:
        """
        Overrides FileSystemEventHandler and what to do when file created
        Creates a dataloader and makes prediction

        Args:
            event (FileSystemEventHandler): Event representing file/directory creation.
        """

        # if self.check_night_image(event.src_path):
        # print(f"found image: {event.src_path}")
        test_data = cocpit.data_loaders.TestDataSet(
            open_dir="",
            file_list=[event.src_path],
        )
        test_loader = cocpit.data_loaders.create_loader(
            test_data, batch_size=1, sampler=None
        )
        with torch.no_grad():
            for imgs, _ in test_loader:

                self.b = cocpit.predictions.BatchPredictions(
                    imgs,
                    torch.load(
                        "/home/vanessa/hulk/ai2es/saved_models/v0.0.0/e[30]_bs[64]_k0_1model(s).pt"
                    ),
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
                    del self.b
                    torch.cuda.empty_cache()


def current_date() -> str:
    """
    Current year/month/day for outfiles

    Returns:
        (str): current date down to day
    """
    return datetime.now().strftime("%Y/%m/%d/%H/%M/%S")


def cam_photo_date() -> str:
    """
    Monitored dirs

    Returns:
        (str): current date down to day
    """
    return datetime.now().strftime("%Y/%m/%d/")


def csv_output_path(
    output_dir: str = "/home/vanessa/hulk/ai2es/realtime_predictions",
) -> str:
    """
    Where to save csv output file

    Returns:
        (str): where predictions should be saved. Once daily.
    """
    if not os.path.exists(f"{output_dir}/csv/{current_date()}/"):
        os.makedirs(f"{output_dir}/csv/{current_date()}/", exist_ok=True)
    return f"{output_dir}/csv/{current_date()}/{current_date().replace('/', '_')}.csv"


def write_header(w: csv.writer) -> csv.writer:
    """
    open csv file and write header for columns

    Returns:
        w (csv._writer): a writer object responsible for converting data to CSV format
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


def observer_setup(
    photo_dir: str = "/rdma/dgx-a100/NYSM/archive/nysm/cam_photos",
) -> Tuple[List[PollingObserver], io.TextIOWrapper, PollingObserver]:
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
        path = f"{photo_dir}/{cam_photo_date()}/{stn}"
        if os.path.exists(path):
            observer.schedule(MonitorFolder(csvfile), path=path, recursive=True)
            observers.append(observer)
    return (observers, csvfile, observer)


if __name__ == "__main__":

    observers, csvfile, observer = observer_setup()
    observer.start()

    print("Monitoring started: ", datetime.now().strftime("%Y/%m/%d/%H:%M:%S"))

    try:
        # kill script if midnight to write to next days file
        while datetime.now().minute % 5 != 0:
            # Poll every x seconds
            time.sleep(10)

    except KeyboardInterrupt:
        for o in observers:

            o.unschedule_all()

            # Stop observer if interrupted
            o.stop()
    finally:
        print("Stopping monitoring to write to next days file")
        csvfile.close()
        for o in observers:
            # Wait until the thread terminates before exit
            o.join()
