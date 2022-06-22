import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import Button
import PIL
import cocpit
from cocpit.interpretability.misc_funcs import (normalize,
                                preprocess_image,
                                apply_colormap_on_image,
                                )
import cocpit.config as config
from cocpit.auto_str import auto_str
from typing import Optional, Tuple
import cv2
import os
from cocpit.interpretability import (gradcam, vanilla_backprop, guided_backprop)

plt_params = {
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.title_fontsize": 12,
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


class Interp():
    """
    Interpretability plot methods for GUI

    Args:
        cam (np.ndarray): Class activation map.  Size of original image
        vanilla_grads (np.ndarray): basic implementation of gradient descent. No momentum or stochastic version.
        gradients (np.ndarray):  a vector which gives us the direction in which the loss function has the steepest ascent.
        pos_saliency (np.ndarray): Positive values in the gradients in which a small change to that pixel will increase the output value
        neg_saliency (np.ndarray): Negative values in the gradients in which a small change to that pixel will decrease the output value
    """

    def __init__(self, vanilla_grads=None, gradients=None, pos_saliency=None, neg_saliency=None, cam=None):
        self.vanilla_grads = vanilla_grads
        self.gradients = gradients
        self.pos_saliency = pos_saliency
        self.neg_saliency = neg_saliency
        self.cam = cam
    
    def plot_saliency(
        self, ax2: plt.Axes, size: int = 224
    ) -> None:
        """create saliency map for image in test dataset

        Args:
            ax2 (plt.Axes): subplot axis
            size (int): image size for transformation
        """
        image = cocpit.plotting_scripts.saliency.preprocess(self.image, size)
        saliency, _, _ = cocpit.plotting_scripts.saliency.get_saliency(image)
        saliency = cv2.resize(np.array(np.transpose(saliency, (1,2,0))), self.target_size)
        ax2.imshow(saliency, cmap=plt.cm.hot)
        ax2.axes.xaxis.set_ticks([])
        ax2.axes.yaxis.set_ticks([])
        ax2.set_title("Saliency Map")

    def get_vanilla_grads(self) -> None:
        """gradients for vanilla backpropagation"""
        VBP = vanilla_backprop.VanillaBackprop()
        vanilla_grads = VBP.generate_gradients(self.prep_img, self.target_size)
        self.vanilla_grads = normalize(vanilla_grads)
        print(np.shape(self.vanilla_grads))

    def plot_vanilla_bp(self, ax: plt.Axes) -> None:
        """plot vanilla backpropagation gradients"""
        ax.imshow(self.vanilla_grads)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.axes.set_title('Vanilla Backpropagation')

    def generate_cam(self):
        """generate gradient class activation map"""
        grad_cam = gradcam.GradCam(target_layer=42)
        self.cam = grad_cam.generate_cam(self.prep_img)

    def plot_gradcam(self, ax: plt.Axes) -> None:
        """plot gradient class activation map"""
        heatmap = apply_colormap_on_image(self.cam, self.image, alpha=0.5)
        ax.imshow(heatmap)
        ax.axes.set_title('GRAD-CAM')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def get_guided_grads(self):
        """
        Guided backpropagation and saliency maps.
        Positive and negative gradients indicate the direction in which we
        would have to change this feature to increase the conditional
        probability of the attended class given this input example and
        their magnitude shows the size of this effect.
        """
        GBP = guided_backprop.GuidedBackprop()
        self.gradients = GBP.generate_gradients(self.prep_img, self.target_size)
        self.pos_saliency = (np.maximum(0, self.gradients[:, :, 0]) / self.gradients[:, :, 0].max())
        self.neg_saliency = (np.maximum(0, -self.gradients[:, :, 0]) / -self.gradients[:, :, 0].min())
        

    def plot_saliency_pos(self, ax: plt.Axes):
        """
        plot positive saliency - where gradients are positive after RELU
        """
        ax.imshow(self.pos_saliency)
        ax.axes.set_title('Positive Saliency')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def plot_saliency_neg(self, ax: plt.Axes):
        """
        plot negative saliency - where gradients are positive after RELU
        """
        ax.imshow(self.neg_saliency)
        ax.axes.set_title('Negative Saliency')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def plot_guided_gradcam(self, ax: plt.Axes) -> None:
        """
        Guided Grad CAM combines the best of Grad CAM,
        which is class-discriminative and localizes relevant image regions,
        and Guided Backpropagation, which visualizes gradients with respect
        to the image where negative gradients set to zero to highlight
        import pixel in the image when backpropagating through ReLU layers.
        """
        cam_gb = np.multiply(self.cam, self.gradients[:,:,0])
        ax.imshow(cam_gb)
        ax.axes.set_title('Guided GRAD-CAM')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])


@auto_str
class GUI(Interp):
    """
    - ipywidget to view model predictions from a test loader (no labels)
    - The dataloader, model, and all class variables are initialized in notebooks/check_classifications.ipynb

    Args:
        index (int): index of the image in the list of paths
        paths (np.ndarray[str]): image paths
        topk_probs (np.ndarray[float]): top predicted probabilites
        topk_classes (np.ndarray[int]): classes related to the top predicted probabilites
        precip (Optional[List[float]]): list of precip obs for display in title and verification
        output (ipywidget.Output): main display
        next_btn (widgets.Button): next button to move index by one
        count (int): number of moved images
        prep_image (torch.Tensor): preprocessed image. Default None, defined in interp()
        target_size (Tuple[int, int]): original image size for resizing interpretability plots
        """
    
    def __init__(self, paths, topk_probs, topk_classes, precip=None):
        self.index = 0
        self.paths = paths
        self.topk_probs = topk_probs
        self.topk_classes = topk_classes
        self.precip = precip
        self.output = ipywidgets.Output()
        self.next_btn = Button(description="Next")
        self.next_btn.on_click(self.on_button_next)
        self.count = 0
        self.prep_img = None
        self.target_size = None

    def open_image(self) -> Optional[PIL.Image.Image]:
        try:
            self.image = PIL.Image.open(self.paths[self.index])
            self.target_size = (np.shape(self.image)[1], np.shape(self.image)[0])
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
        self.interp()

    def show_original(self, ax1: plt.Axes) -> None:
        """
        display the raw image

        Args:
            ax1 (plt.Axes): subplot axis
        """
        clear_output()  # so that the next fig doesnt display below
        ax1.imshow(self.image)
        station = self.paths[self.index].split("/")[-1].split("_")[-1].split(".")[0]
        if self.precip:
            ax1.set_title(
                f"Prediction: {[config.CLASS_NAMES[e] for e in self.topk_classes[self.index]][0]}\n"
                f"Station: {station}\n"
                f"1 min precip accumulation: {self.precip[self.index].values[0]}"
            )
        else:
            pred_list = [config.CLASS_NAMES[e] for e in self.topk_classes[self.index]]
            pred_mag = [np.round(i*100,2) for i in self.topk_probs[self.index]]
            ax1.set_title(
                f"Prediction [%]: \n"
                f"{', '.join(repr(e) for e in pred_list)}\n"
                f"{', '.join(repr(e) for e in pred_mag)}"
            )
        ax1.axis("off")

    def bar_chart(self, ax3: plt.Axes) -> None:
        """create barchart that outputs top k predictions for a given image

        Args:
            ax3 (plt.Axes): subplot axis
        """
        y_pos = np.arange(len(self.topk_probs[self.index]))
        ax3.barh(y_pos, self.topk_probs[self.index], align="center")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(
            [config.CLASS_NAMES[e] for e in self.topk_classes[self.index]]
        )
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax3.invert_yaxis()  # labels read top-to-bottom
        ax3.set_title("Class Probability")

    def save(self, fig: plt.Axes, directory: str='/ai2es/codebook_dataset/carly/interpretability', class_='unsure_dark'):
        if not os.path.exists(os.path.join(directory, class_)):
            os.makedirs(os.path.join(directory, class_))
        fig.savefig(os.path.join(directory, class_, self.paths[self.index].split('/')[-1]))

    def call_plots(self, figsize: Tuple[int,int]=(12, 6), ncols=3, nrows=2):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            constrained_layout=True, figsize=figsize, ncols=ncols, nrows=nrows
        )
        self.show_original(ax1)
        #self.plot_saliency(ax2)
        #self.bar_chart(ax3)
        self.plot_vanilla_bp(ax4)
        self.plot_gradcam(ax5)
        self.plot_guided_gradcam(ax6)
        self.plot_saliency_pos(ax2)
        self.plot_saliency_neg(ax3)
        self.save(fig)
        
    def interp(self) -> None:
        """
        Calculate gradients used in interpretability
        """
        with self.output:
            # add chart to ipywidgets.Output()
            if self.index == len(self.topk_probs):
                print("You have completed looking at all predictions!")
                return
            else:
                self.open_image()
                self.prep_img = preprocess_image(self.image).cuda()        
                self.generate_cam()
                self.get_guided_grads()
                self.get_vanilla_grads()
                self.call_plots()
                plt.show()
                
