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
import netCDF4 as nc
import os


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

    def write_netcdf(self, filename):
        fn = "test.nc"
        ds = nc.Dataset(fn, "w", format="NETCDF4")

        time = ds.createDimension("time", None)
        lat = ds.createDimension("lat", 10)
        lon = ds.createDimension("lon", 10)

    def write_csv(self, filename):
        """
        Write probability for each class out to a csv

        Args:
            filename (str): path to image
        """
        self.w.writerow(
            [
                filename.src_path,
                config.CLASS_NAMES[np.argmax(self.b.probs)],
                np.round(self.b.probs[0], 3) * 100,
                np.round(self.b.probs[1], 3) * 100,
                np.round(self.b.probs[2], 3) * 100,
            ]
        )
        self.w.flush()

    def on_created(self, event):
        """
        Overrides FileSystemEventHandler and what to do when file created
        Creates a dataloader and makes pred

        Args:
            event (FileCreatedEvent): Event representing file/directory creation.
        """
        print(event.src_path)
        test_data = cocpit.data_loaders.TestDataSet(
            open_dir="", file_list=[event.src_path]
        )

        # if self.check_night_image(event.src_path):
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

            self.write_csv(event)
            # self.write_netcdf(event)


def current_date():
    """Current year/month/day for outfiles

    Returns:
        (datetime): current date down to day
    """
    return datetime.now().strftime("%Y/%m/%d")


def path_to_check(stn):
    """
    Directory to monitor

    Args:
        stn (str): station id
    Returns:
        (str): where images are getting fed into
    """
    print(f"/ai2es/cam_photos/{current_date()}/{stn}")
    return f"/ai2es/cam_photos/{current_date()}/{stn}"


def csv_output_path(stn: str):
    """
    Where to save csv output files

    Args:
        stn (str): station id

    Returns:
        (str): where predictions should be saved. Once daily.
    """
    if not os.path.exists(f"/ai2es/realtime_predictions/csv/{current_date()}/{stn}"):
        os.makedirs(f"/ai2es/realtime_predictions/csv/{current_date()}/{stn}")
    return f"/ai2es/realtime_predictions/csv/{current_date()}/{stn}/{current_date().replace('/', '_')}.csv"


def nc_output_path():
    return f"/ai2es/realtime_predictions/nc/{current_date()}.nc"


def observer_setup():
    """
    Create observers to watch directories across all stations

    Returns:
        observers (List[PollingObserver]): list of observers across all stations
        file_handles (List[TextIOWrapper]): list of open csv files to write preds to
        observer (PollingObserver): a PollingObserver instance
    """
    observers = []
    file_handles = []
    observer = PollingObserver()
    for stn in config.stnid:
        # write file header first
        csvfile = open(csv_output_path(stn), "a", newline="")
        file_handles.append(csvfile)

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
        observer.schedule(MonitorFolder(w), path=path_to_check(stn), recursive=True)
        observers.append(observer)
    return observers, file_handles, observer


if __name__ == "__main__":

    observers, file_handles, observer = observer_setup()
    observer.start()
    print("Monitoring started")
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
        for fh in file_handles:
            fh.close()
        for o in observers:
            # Wait until the thread terminates before exit
            o.join()
