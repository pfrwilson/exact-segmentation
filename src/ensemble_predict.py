import os
from typing import List, Optional
from omegaconf import OmegaConf
from sklearn import ensemble

import torch
from tqdm import tqdm

from src.segmentation_module import ExactSegmentationModule
from exactvu.data import Core

from pkg_resources import resource_filename
import numpy as np


def download_checkpoints_from_server():
    from exactvu.client import sftp
    from omegaconf import OmegaConf

    fname = resource_filename(
        __name__, os.path.join("resources", "server_checkpoints.yaml")
    )
    checkpoints = OmegaConf.load(fname)["checkpoints"]  # type:ignore
    local_checkpoints = []

    for checkpoint in checkpoints:
        name = checkpoint.split("/")[-1]
        local_fname = resource_filename(__name__, f"resources/{name}")
        sftp.get(checkpoint, local_fname)
        local_checkpoints.append(local_fname)

    with open(
        resource_filename(
            __name__, os.path.join("resources", "local_checkpoints.yaml")
        ),
        "w",
    ) as f:
        f.write(OmegaConf.to_yaml({"checkpoints": local_checkpoints}))


def get_checkpoints(local_checkpoints_file=None):
    if local_checkpoints_file is None:

        fname = resource_filename(
            __name__, os.path.join("resources", "local_checkpoints.yaml")
        )
        if not os.path.isfile(fname):
            download_checkpoints_from_server()

        return OmegaConf.load(fname)["checkpoints"]  # type:ignore

    else:
        return OmegaConf.load(local_checkpoints_file)["checkpoints"]  # type: ignore


def collect_models(checkpoints: List[str]):

    from .segmentation_module import ExactSegmentationModule

    return [
        ExactSegmentationModule.load_from_checkpoint(checkpoint, map_location="cpu")
        for checkpoint in checkpoints
    ]


def ensemble_predict(
    models: List[ExactSegmentationModule],
    bmode: torch.Tensor,
    anatomical_location: Optional[torch.Tensor] = None,
    compute_device: Optional[str] = None,
):

    if not compute_device:
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    if bmode.ndim == 3:
        bmode = torch.unsqueeze(bmode, dim=0)
    if anatomical_location is not None and anatomical_location.ndim == 0:
        anatomical_location = torch.unsqueeze(anatomical_location, dim=0)

    bmode = bmode.to(compute_device)
    if anatomical_location is not None:
        anatomical_location = anatomical_location.to(compute_device)

    logits = []
    for model in models:
        model.eval()
        model.to(compute_device)
        logits.append(model(bmode, info=anatomical_location))
        model.to("cpu")

    preds = [logits_.softmax(1) for logits_ in logits]
    preds = sum(preds) / len(preds)

    preds = preds.detach().cpu()  # type:ignore

    return preds


class CoreDataExtractor:
    @staticmethod
    def from_seg_module(module: ExactSegmentationModule):
        module.setup()
        return CoreDataExtractor(module.eval_transform, module.info_dict_transform)

    def __init__(self, transform, info_dict_transform):
        self.transform = transform
        self.info_dict_transform = info_dict_transform

    def __call__(self, core: Core):

        if core.bmode is None:
            core.download_bmode()

        bmode = core.bmode
        info_dict = {"anatomical_location": core.metadata["loc"]}

        # use zeros as dummy mask since transform expects bmode, mask as args
        bmode, _ = self.transform(bmode, np.zeros_like(bmode).astype("uint8"))
        info = self.info_dict_transform(info_dict)

        return bmode, info


def predict(core_specifiers, cores_root, checkpoints_path=None, overwrite=False):
    """
    Predicts the pixel-wise probablilities for each of the specified cores and pushes
    these predictions to the server.

    Args:
        core_specifiers(List[str]): the list of core specifiers to predict
        cores_root(str): the root directory containing the data for the core objects
        checkpoints_path: the .yaml file listing the checkpoints to load into the ensemble
            for prediction. If None, it will download the checkpoints from the server.
        overwrite (bool): If true, will overwrite the mask probabilities already
            existing on the server.

    Returns:
        A ProbsDownloader object which can be used to access the newly created predictions
        using index notation.
    """

    from exactvu.client import (
        add_prostate_mask_probs,
        check_prostate_mask_probs_exists,
        load_prostate_mask_probs,
    )

    checkpoints = get_checkpoints(checkpoints_path)
    ensemble = collect_models(checkpoints)
    transform = CoreDataExtractor.from_seg_module(ensemble[0])

    def predict_on_core(core):
        bmode, info = transform(core)
        probs = ensemble_predict(ensemble, bmode, info)

        return bmode.numpy()[0], probs.numpy()[0][1]

    for specifier in tqdm(core_specifiers):

        if check_prostate_mask_probs_exists(specifier) and not overwrite:
            continue

        core = Core(specifier, cores_root)

        _, probs = predict_on_core(core)

        add_prostate_mask_probs(specifier, probs)

    class ProbsDownloader:
        def __init__(self, specifiers):
            self.specifiers = specifiers

        def __getitem__(self, idx):
            specifier = self.specifiers[idx]
            return load_prostate_mask_probs(specifier)

    return ProbsDownloader(core_specifiers)
