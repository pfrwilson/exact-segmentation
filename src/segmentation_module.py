import pytorch_lightning as pl
from .unet import UNetWithDiscreteFeatureEmbedding, UNet
from .loss import SegmentationCriterion
from .metrics import Metrics
from .dataset import BModeSegmentationDataset
from .preprocessing import Preprocessor
from torch.utils.data import DataLoader
import torch

ANATOMICAL_LOCATIONS = [
    "LML",
    "RBL",
    "LMM",
    "RMM",
    "LBL",
    "LAM",
    "RAM",
    "RML",
    "LBM",
    "RAL",
    "RBM",
    "LAL",
]

ANATOMICAL_LOCATIONS_INV = {name: idx for idx, name in enumerate(ANATOMICAL_LOCATIONS)}


class ExactSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        root=None,
        use_anatomical_location_embeddings=False,
        dice_loss_weight=0.5,
        use_augmentations=True,
        equalize_hist=False,
        random_rotate=True,
        grid_distortion=True,
        horizontal_flip=True,
        gaussian_blur=True,
        random_brightness=True,
        to_tensor=True,
        out_size=(512, 512),
        batch_size=32,
        num_workers=8,
        optimizer_name="sgd",
        lr: float = 0.001,
        num_epochs=100,
        scheduler="cosine",
    ):

        super().__init__()
        self.save_hyperparameters()

        self.root = root

        self.use_anatomical_location_embeddings = use_anatomical_location_embeddings

        self.use_augmentations = use_augmentations
        self.equalize_hist = equalize_hist
        self.random_rotate = random_rotate
        self.grid_distortion = grid_distortion
        self.horizontal_flip = horizontal_flip
        self.gaussian_blur = gaussian_blur
        self.random_brightness = random_brightness
        self.to_tensor = to_tensor
        self.out_size = out_size

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.num_epochs = num_epochs
        self.scheduler = scheduler

        if use_anatomical_location_embeddings:
            self.model = UNetWithDiscreteFeatureEmbedding(
                1, len(ANATOMICAL_LOCATIONS), 2
            )
        else:
            self.model = UNet(1, 2)

        self.criterion = SegmentationCriterion(dice_loss_weight=dice_loss_weight)
        self.train_metrics = Metrics()
        self.val_metrics = Metrics()
        self.test_metrics = Metrics()

        self.train_ds = None
        self.val_ds = None

    def setup(self, *args, **kwargs):

        if self.train_ds is None or self.val_ds is None:

            self.train_transform = Preprocessor(
                use_augmentations=self.use_augmentations,
                equalize_hist=self.equalize_hist,
                random_rotate=self.random_rotate,
                grid_distortion=self.grid_distortion,
                horizontal_flip=self.horizontal_flip,
                gaussian_blur=self.gaussian_blur,
                random_brightness=self.random_brightness,
                to_tensor=self.to_tensor,
                out_size=self.out_size,
            )

            self.info_dict_transform = lambda d: torch.tensor(
                ANATOMICAL_LOCATIONS_INV[d["anatomical_location"]]
            )

            self.eval_transform = Preprocessor(
                use_augmentations=False, out_size=self.out_size
            )

            self.train_ds = BModeSegmentationDataset(
                self.root,
                split="train",
                transform=self.train_transform,
                info_dict_transform=self.info_dict_transform,
            )

            # self.val_ds = BModeSegmentationDataset(
            #    self.root,
            #    split="val",
            #    transform=self.eval_transform,
            #    info_dict_transform=self.info_dict_transform,
            # )

            self.val_ds = BModeSegmentationDataset(
                self.root,
                split="test",
                transform=self.eval_transform,
                info_dict_transform=self.info_dict_transform,
            )

    def train_dataloader(self):

        assert self.train_ds is not None

        return DataLoader(
            self.train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):

        assert self.val_ds is not None

        return DataLoader(
            self.val_ds, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def configure_optimizers(self):
        from torch.optim import Adam, SGD

        if self.optimizer_name == "adam":
            opt = Adam(self.parameters(), lr=self.lr)
        else:
            opt = SGD(self.parameters(), self.lr, momentum=0.9)

        if self.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            sched = CosineAnnealingLR(opt, T_max=self.num_epochs)
        else:
            sched = None

        if sched is not None:
            return [opt,], [
                sched,
            ]
        else:
            return opt

    def configure_callbacks(self):
        from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
        from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar

        return LearningRateMonitor(), RichModelSummary()

    def forward(self, bmode, info=None):
        if self.use_anatomical_location_embeddings:
            return self.model(bmode, features=info)
        else:
            return self.model(bmode)

    def training_step(self, batch, batch_idx=None):

        bmode, seg, info = batch
        logits = self(bmode, info=info)

        loss = self.criterion(logits, seg)
        metrics = self.train_metrics(logits, seg)

        self._log_dict(metrics, "train")

        return {
            "loss": loss,
            "bmode": bmode.cpu(),
            "seg": seg.cpu(),
            "logits": logits.detach().cpu(),
            "info": info,
        }

    def training_epoch_end(self, outputs) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        bmode, seg, info = batch
        logits = self(bmode, info=info)

        self._log_dict(self.val_metrics(logits, seg), "val")

        return {
            "bmode": bmode.cpu(),
            "seg": seg.cpu(),
            "logits": logits.detach().cpu(),
            "info": info,
        }

    def validation_epoch_end(self, outputs) -> None:
        self.val_metrics.reset()
        self.test_metrics.reset()

    def _log_dict(self, d, prefix=""):
        self.log_dict({f"{prefix}_{k}": v for k, v in d.items()})
