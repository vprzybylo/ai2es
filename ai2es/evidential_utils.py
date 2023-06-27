""" functions for evidential uncertainty calculations

used in notebooks/evidential_uncertainty.ipynb
"""
from typing import List, Tuple

import cocpit.config as config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


def model_logit_output(model, img: PIL.Image) -> torch.Tensor:
    """run model on image and output logit"""
    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    return model(img_variable)


def calc_uncertainty(
    model: torch.nn.parallel.DataParallel, output: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """calculate max probability prediction, class probability, and uncertainty"""
    softmax = F.softmax(output, dim=1)
    evidence = F.relu(output)
    alpha = evidence + 1
    # uncertainty calculated as # of classes/total evidence summed across classes + 1
    uncertainty = len(config.CLASS_NAMES) / torch.sum(alpha, dim=1, keepdim=True)
    _, pred = torch.max(output, 1)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    output = output.flatten()
    prob = softmax.flatten()
    pred = pred.flatten()
    return (pred, prob, uncertainty)


def uncertainty_examples(model, img_paths: List[str]) -> None:
    """loop through a sample of images, make predictions, and output uncertainty"""
    for img_path in img_paths:
        img = PIL.Image.open(img_path)
        output = model_logit_output(model, img)
        pred, prob, uncertainty = calc_uncertainty(model, output)
        plot_uncertainty(img, pred, prob, uncertainty)


def plot_uncertainty(
    img: PIL.Image,
    preds: torch.Tensor,
    prob: torch.Tensor,
    uncertainty: torch.Tensor,
    precip: float = None,
    fontsize: int = 16,
) -> None:
    """plot image above with bar chart below of probability and uncertainty value in title"""
    labels = np.arange(len(config.CLASS_NAMES))
    fig, axs = plt.subplots(
        2, 1, figsize=(6, 12), gridspec_kw={"height_ratios": [3, 1]}
    )

    plt.title(
        f"Classified as: {config.CLASS_NAMES[preds[0]]}, Uncertainty: {np.round(uncertainty.item(), 3)}",
        fontsize=fontsize,
    )

    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")

    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    plt.xticks(
        np.arange(len(config.CLASS_NAMES)),
        config.CLASS_NAMES,
        rotation="vertical",
        fontsize=fontsize,
    )

    axs[1].set_ylim([0, 1])
    axs[1].set_xticklabels(config.CLASS_NAMES, fontsize=fontsize)

    axs[1].set_xlabel("Classes", fontsize=fontsize + 2)
    axs[1].set_ylabel("Softmax Probability", fontsize=fontsize + 2)
    if precip:
        axs[1].set_title(f"Accumulated Precip: {precip}")

    fig.tight_layout()

    # plt.savefig("{/home/vanessa/hulk/ai2es}/plots/{}".format(os.path.basename(img_path)))


def uncertainty_acc_hist(grouped_df: pd.DataFrame) -> None:
    """histogram of accuracy binned by uncertainty"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    grouped_df["acc"].plot(kind="bar", ax=ax, color="k")
    plt.xlabel("Uncertainty Range", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    xlocs, xlabs = plt.xticks()
    plt.ylim(0.0, 1.10)
    plt.ylabel("Mean Accuracy [%]", fontsize=18)
    for index, (c, acc) in enumerate(zip(grouped_df["count"], grouped_df["acc"])):
        shift = 0.2 if c < 100 else 0.3
        plt.text(xlocs[index] - shift, acc + 0.02, f"{c}", fontsize=16)
    plt.savefig(f"{config.BASE_DIR}/plots/uncertainty_vs_acc_hist.png")


def grouped_acc_df(group: pd.DataFrame) -> pd.Series:
    """bin df by uncertainty"""
    return pd.Series(
        {
            "preds": group["preds"],
            "labels": group["labels"],
            "probs": group["probs"],
            "uncertainties": group["uncertainties"],
            "paths": group["paths"],
            "acc": (group["preds"] == group["labels"]).mean(),
            "count": len(group),
        }
    )
