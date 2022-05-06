import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import Button
import PIL
import torch
import cocpit
import cocpit.config as config
from cocpit.auto_str import auto_str
from typing import Optional

plt_params = {
    "axes.labelsize": "xx-large",
    "axes.titlesize": "xx-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
    "legend.title_fontsize": 12,
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


@auto_str
class GUI:
    """create widgets"""

    def __init__(self, all_paths, all_topk_probs, all_topk_classes, precip=None):

        self.index = 0
        self.all_paths = all_paths
        self.all_topk_probs = all_topk_probs
        self.all_topk_classes = all_topk_classes
        self.next_btn = Button(description="Next")
        self.next_btn.on_click(self.on_button_next)
        self.count = 0  # number of moved images
        self.center = ipywidgets.Output()  # center image with predictions
        self.precip = precip

    def open_image(self) -> Optional[PIL.Image.Image]:
        try:
            return PIL.Image.open(self.all_paths[self.index])
        except FileNotFoundError:
            print("The file cannot be found.")
            return

    def on_button_next(self, b) -> None:
        """
        when the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1

        Args:
            b: button instance
        """
        self.index = self.index + 1
        self.visualizations()

    def init_fig(self, image: PIL.Image.Image, ax1: plt.Axes) -> None:
        """
        display the raw image

        Args:
            image (PIL.Image.Image): opened image
            ax1 (plt.Axes): subplot axis
        """
        clear_output()  # so that the next fig doesnt display below
        ax1.imshow(image, aspect="auto")
        station = self.all_paths[self.index].split("/")[-1].split("_")[-1].split(".")[0]
        if self.precip:
            ax1.set_title(
                f"Model Labeled as: {[config.CLASS_NAMES[e] for e in self.all_topk_classes[self.index]][0]}\n"
                f"Station: {station}\n"
                f"1 min precip accumulation: {self.precip[self.index].values[0]}"
            )
        else:
            ax1.set_title(
                f"Model Labeled as: {[config.CLASS_NAMES[e] for e in self.all_topk_classes[self.index]][0]}\n"
            )
        ax1.axis("off")

    def bar_chart(self, ax3: plt.Axes) -> None:
        """create barchart that outputs top k predictions for a given image

        Args:
            ax3 (plt.Axes): subplot axis
        """
        y_pos = np.arange(len(self.all_topk_probs[self.index]))
        ax3.barh(y_pos, self.all_topk_probs[self.index], align="center")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(
            [config.CLASS_NAMES[e] for e in self.all_topk_classes[self.index]]
        )
        ax3.tick_params(axis="y", rotation=45)
        ax3.invert_yaxis()  # labels read top-to-bottom
        ax3.set_title("Class Probability")

    def plot_saliency(
        self, image: PIL.Image.Image, ax2: plt.Axes, size: int = 224
    ) -> None:
        """create saliency map for image in test dataset

        Args:
            image (PIL.Image.Image): opened image
            ax2 (plt.Axes): subplot axis
            size (int): image size for transformation
        """
        image = cocpit.plotting_scripts.saliency.preprocess(image.convert("RGB"), size)
        saliency, _, _ = cocpit.plotting_scripts.saliency.get_saliency(image)
        ax2.imshow(saliency[0], cmap=plt.cm.hot, aspect="auto")
        ax2.axes.xaxis.set_ticks([])
        ax2.axes.yaxis.set_ticks([])

    def visualizations(self) -> None:
        """
        use the human and model labels and classes to
        show the image prediction probability per class and
        output a saliency map for the current image
        """

        # add chart to ipywidgets.Output()
        with self.center:
            if self.index == len(self.all_topk_probs):
                print("You have completed looking at all predictions!")
                return
            else:
                image = self.open_image()
                _, (ax1, ax2, ax3) = plt.subplots(
                    constrained_layout=True, figsize=(7, 11), ncols=1, nrows=3
                )
                self.init_fig(image, ax1)
                self.plot_saliency(image, ax2)
                self.bar_chart(ax3)
                plt.show()
