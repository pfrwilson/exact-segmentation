from exactvu.data import Core
import os
from torch.utils.data import Dataset
from tqdm import tqdm


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


class BModeSegmentationDataset(Dataset):
    def __init__(
        self, root=None, split="train", transform=None, info_dict_transform=None
    ):

        self.root = root if root is not None else os.getenv("DATA_ROOT")
        if self.root is None:
            raise ValueError(
                "Data root not specifier or found as environment variable. Cannot proceed"
            )

        self.directory = os.path.join(self.root, "cores_dataset")
        self.transform = transform
        self.info_dict_transform = info_dict_transform

        from exactvu.data.splits import get_splits, filter_splits, HasProstateMaskFilter

        train, val, test = filter_splits(
            get_splits(
                "UVA", split_seed=26, train_val_ratio=0.15, balance_classes=False
            ),
            HasProstateMaskFilter(),
        )

        # merge the train and validation sets to keep as much data available as possible
        train = train + val
        self.core_specifiers = {"train": train, "test": test}[split]

        self.cores = [
            Core(core_specifier, self.directory)
            for core_specifier in self.core_specifiers
        ]
        for core in tqdm(
            self.cores, desc="Downloading b_modes and prostate masks if necessary"
        ):
            core: Core
            if core.bmode is None:
                core.download_bmode()
            if core.prostate_mask is None:
                if not core.download_prostate_mask():
                    import warnings

                    warnings.warn(
                        f"Prostate mask not available for core {core.specifier}"
                    )

        from exactvu.resources import metadata

        self.metadata = metadata().query("core_specifier in @self.core_specifiers")

    def __len__(self):
        return len(self.cores)

    def __getitem__(self, idx):

        core_specifier = self.core_specifiers[idx]
        core = self.cores[idx]
        bmode = core.bmode
        seg = core.prostate_mask

        if self.transform:
            bmode, seg = self.transform(bmode, seg)

        metadata = {
            "anatomical_location": self.metadata.query(
                "core_specifier == @core_specifier"
            ).iloc[0]["loc"]
        }
        if self.info_dict_transform:
            metadata = self.info_dict_transform(metadata)

        return bmode, seg, metadata
