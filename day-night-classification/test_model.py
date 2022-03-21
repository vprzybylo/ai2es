"""make prediction on folder for day or night
run using:
python test_model.py --model model/day_night.hd5 --folder test/ --save True
"""


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from os import listdir
from os.path import isfile, join


class Image:
    """make prediction for day/night on image"""

    def __init__(self, args, file):
        self.args = args
        self.filename = file
        self.image = cv2.imread(f"test/{self.filename}")
        self.orig = self.image.copy()
        self.label = None
        self.output = None

    def preprocess(self):
        """pre-process the image for classification"""
        image = cv2.resize(self.image, (28, 28))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        self.image = np.expand_dims(image, axis=0)

    def predict(self):
        """load the trained convolutional neural network"""
        model = load_model(self.args["model"])

        # classify the input image
        (night, day) = model.predict(self.image)[0]

        # build the label
        label = "day" if day > night else "night"
        proba = day if day > night else night
        self.label = "{}: {:.2f}%".format(label, proba * 100)

    def draw_label(self):
        """draw the label on the image"""
        self.output = imutils.resize(self.orig, width=800)
        cv2.putText(
            self.output,
            self.label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def show_output(self, save=False):
        """show the output image"""
        label = self.label.split(":")[0]
        if save and (".jpg" in self.filename):
            cv2.imwrite(f"test/{label}/{self.filename}", self.output)
            # cv2.imshow("output", self.output)
            # cv2.waitKey(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to trained model model")
    ap.add_argument("-f", "--folder", required=True, help="path to folder with images")
    ap.add_argument(
        "-s", "--save", required=True, help="save the images based on label"
    )
    args = vars(ap.parse_args())

    # list files in dir
    files = [f for f in listdir(args["folder"]) if isfile(join(args["folder"], f))]

    for file in files:
        img = Image(args, file)
        img.preprocess()
        img.predict()
        img.draw_label()
        img.show_output(args["save"])


if __name__ == "__main__":
    main()
