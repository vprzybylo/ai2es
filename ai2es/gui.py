"""
Holds the class for ipywidget buttons to
label from a folder of images.
Run in notebooks/label.ipynb
"""
import ipywidgets
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ipywidgets import Button
import PIL
from shutil import copyfile
import os


class GUI:
    """
    view and label images in notebooks/label.ipynb
    """

    def __init__(self, all_paths, labels, folder_dest):
        self.all_paths = all_paths
        self.n_paths = len(self.all_paths)
        self.folder_dest = folder_dest
        self.labels = labels
        self.index = 0
        self.center = ipywidgets.Output()
        self.undo_btn = Button(description="Undo")
        self.buttons = []

    def open_image(self) -> PIL.JpegImagePlugin.JpegImageFile:
        try:
            image = PIL.Image.open(self.all_paths[self.index])

        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")
        return image

    def make_buttons(self):
        self.undo_btn.on_click(self.undo)

        for idx, label in enumerate(self.labels):
            self.buttons.append(Button(description=label))
            self.buttons[idx].on_click(self.cp_to_dir)

    def cp_to_dir(self, b):
        filename = self.all_paths[self.index].split("/")[-1]
        output_path = os.path.join(self.folder_dest, b.description, filename)
        copyfile(self.all_paths[self.index], output_path)
        self.index = self.index + 1
        self.display_image()

    def undo(self, b):
        self.index = self.index - 1
        self.display_image()

    def display_image(self):
        """
        show image
        """
        with self.center:
            clear_output()  # so that the next fig doesnt display below
            image = self.open_image()
            fig, ax = plt.subplots(
                constrained_layout=True, figsize=(10, 10), ncols=1, nrows=1
            )
            ax.set_title(f"{self.index}/{self.n_paths}")
            ax.imshow(image)
            ax.axis("off")
            plt.show()
