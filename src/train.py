from omegaconf import OmegaConf
from logging import info
from hydra.utils import instantiate
import pytorch_lightning
import wandb


def train(config):

    if seed := config.get("seed"):
        info(f"Using seed {seed}")
        from pytorch_lightning import seed_everything

        seed_everything(seed)

    info("Instantiating module <src.segmentation_module.ExactSegmentationModule>")
    from src.segmentation_module import ExactSegmentationModule

    module = ExactSegmentationModule(**config.module)

    info(f"Instantiating logger <{config.logger['_target_']}>")
    logger = instantiate(config.logger)

    info("Configuring callbacks")
    callbacks = []
    for name, callback in config.callbacks.items():
        info(f'Adding callback {name}, <{callback["_target_"]}>')
        callbacks.append(instantiate(callback))

    info(f"Instantiating Trainer")
    trainer: pytorch_lightning.Trainer
    trainer = instantiate(config.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(module)
    assert trainer.checkpoint_callback is not None

    best_score = trainer.checkpoint_callback.best_model_score
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    wandb.log({"best_validation_dice": best_score, "best_model_path": best_checkpoint})

    info(f"Best score observed during training: {best_score}")
    info(f"Best model checkpoint saved at {best_checkpoint}!")

    if fname := config.get("checkpoint_file"):
        info(f"Found file {fname} to list checkpoint path.")
        if (
            threshold := config.get("score_threshold_for_checkpoint")
        ) and best_score >= threshold:
            info(
                f"Performance of this model ({best_score}) passes threshold {threshold}. Saving!"
            )

            from omegaconf import OmegaConf

            checkpoints = OmegaConf.load(fname)
            checkpoints["checkpoints"].append(best_checkpoint)

            with open(fname, "w") as f:
                f.write(OmegaConf.to_yaml(checkpoints))
