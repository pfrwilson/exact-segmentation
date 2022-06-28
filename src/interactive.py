from typing import Optional
from ipywidgets import Output, Text, Button, Label, HBox, interact, FloatSlider
from IPython.display import display
import numpy as np
from exactvu.data import Core
import matplotlib
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from exactvu.client import (
    check_prostate_mask_exists,
    load_prostate_mask,
    add_prostate_mask,
    load_prostate_mask_probs,
)


def run(cores):

    idx: int = 0
    bmode: Optional[np.ndarray]
    probs: np.ndarray
    core: Core
    output: Output
    threshold_prob: float = 0.5

    title = Text(
        value=None, placeholder="Type something", description="Core:", disabled=True
    )
    message = Text(value=None, disabled=True)
    count = Text(value=str(len(cores)), description="Cores left")
    title_bar = HBox([title, message, count])

    back_button = Button(description="Go Back", button_style="info")
    save_button = Button(description="Save Mask", button_style="success")
    overwrite_button = Button(description="Overwrite Mask", button_style="danger")
    next_button = Button(description="Next", button_style="info")

    buttons = HBox(
        [
            back_button,
            save_button,
            overwrite_button,
            next_button,
        ]
    )

    output = Output()
    output.clear_output()
    output = Output()

    with output:
        plt.clf()
        plt.ioff()
        fig, ax = plt.subplots()
        plt.ion()
        bmode_imshow = ax.imshow(np.zeros((256, 256)), vmin=0, vmax=255, cmap="gray")
        mask_imshow = ax.imshow(
            np.zeros((256, 256)), alpha=0.5, cmap=ListedColormap("green")
        )
        plt.axis("off")
        plt.ioff()
        fig.show()
        plt.ion()

    def refresh_pred_and_bmode():

        if core is None:
            return

        nonlocal bmode
        bmode = core.bmode
        bmode = cv2.normalize(bmode, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        nonlocal probs
        probs = load_prostate_mask_probs(core.specifier)

    def update_threshold(change):
        with output:
            t = change["new"]
            nonlocal threshold_prob
            threshold_prob = t

            with output:
                mask_imshow.set_data(np.where(probs >= threshold_prob, 1, np.nan))
                fig.canvas.draw_idle()

    threshold_slider = FloatSlider(value=0.5, min=0, max=1, step=0.02)
    threshold_slider.observe(update_threshold, "value")

    def render():
        nonlocal core
        core = cores[idx]
        nonlocal title
        title.value = core.specifier

        if idx + 1 == len(cores):
            next_button.disabled = True
        else:
            next_button.disabled = False

        if idx == 0:
            back_button.disabled = True
        else:
            back_button.disabled = False

        if check_prostate_mask_exists(core.specifier):
            overwrite_button.disabled = False
            save_button.disabled = True
        else:
            overwrite_button.disabled = True
            save_button.disabled = False

        message.value = ""

        refresh_pred_and_bmode()
        with output:
            bmode_imshow.set_data(bmode)
            mask_imshow.set_data(np.where(probs >= threshold_prob, 1, np.nan))
            fig.canvas.draw_idle()

        message.value = "image loaded"

    def save(b):
        try:
            add_prostate_mask(core.specifier, probs >= threshold_prob)
            nonlocal count
            count.value = str(int(count.value) - 1)
            next(b)
        except:
            message.value = "Error occurred when saving"

    def overwrite(b):
        try:
            add_prostate_mask(core.specifier, probs >= threshold_prob, overwrite=True)
            message.value = "Overwrite successful."
        except:
            message.value = "Error occurred while overwriting"

    save_button.on_click(save)
    overwrite_button.on_click(overwrite)

    def next(b):
        nonlocal idx
        idx += 1
        render()

    next_button.on_click(next)

    def back(b):
        nonlocal idx
        idx -= 1
        render()

    back_button.on_click(back)

    display(
        title_bar,
        buttons,
        threshold_slider,
        output,
    )
    render()
