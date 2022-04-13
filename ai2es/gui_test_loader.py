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

    def __init__(self, all_paths, all_topk_probs, all_topk_classes, precip):

        self.index = 0
        self.all_paths = all_paths
        self.all_topk_probs = all_topk_probs
        self.all_topk_classes = all_topk_classes
        self.next_btn = Button(description="Next")
        self.next_btn.on_click(self.on_button_next)
        self.count = 0  # number of moved images
        self.center = ipywidgets.Output()  # center image with predictions
        self.precip = precip

    def open_image(self) -> PIL.Image.Image:
        return PIL.Image.open(self.all_paths[self.index])

    def on_button_next(self, b) -> None:
        """
        when the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1

        Args:
            b: button instance
        """

        self.index = self.index + 1
        self.bar_chart()

    def bar_chart(self) -> None:
        """
        use the human and model labels and classes to
        create a bar chart with the top k predictions
        from the image at the current index
        """

        # # add chart to ipywidgets.Output()
        with self.center:
            if len(self.all_topk_probs) > self.index:

                self.topk_probs = self.all_topk_probs[self.index]
                self.topk_classes = self.all_topk_classes[self.index]

                # puts class names in order based on probabilty of prediction
                crystal_names = [config.CLASS_NAMES[e] for e in self.topk_classes]
                self.view_classifications(self.topk_probs, crystal_names)

            else:
                print("You have completed looking at all predictions!")
                return

    def view_classifications(self, probs, crystal_names) -> None:
        """
        create barchart that outputs top k predictions for a given image
        """
        clear_output()  # so that the next fig doesnt display below
        fig, (ax1, ax2, ax3) = plt.subplots(
            constrained_layout=True, figsize=(7, 11), ncols=1, nrows=3
        )
        try:
            image = self.open_image()
            ax1.imshow(image, aspect="auto")
            station = (
                self.all_paths[self.index].split("/")[-1].split("_")[-1].split(".")[0]
            )
            ax1.set_title(
                f"Model Labeled as: {crystal_names[0]}\n"
                f"Station: {station}\n"
                f"1 min precip accumulation: {self.precip[self.index].values[0]}"
            )
            ax1.axis("off")

            model = torch.load(config.MODEL_PATH).cuda()
            file = self.all_paths[self.index]
            cocpit.plotting_scripts.saliency.saliency_test_set(model, file, ax2)

            y_pos = np.arange(len(self.topk_probs))
            ax3.barh(y_pos, probs, align="center")
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(crystal_names)
            ax3.tick_params(axis="y", rotation=45)
            ax3.invert_yaxis()  # labels read top-to-bottom
            ax3.set_title("Class Probability")
            # fig.savefig(f"/ai2es/plots/wrong_preds{21+self.index}.pdf")
            plt.show()

        except FileNotFoundError:
            print("This file was already moved and cannot be found.")
            return
