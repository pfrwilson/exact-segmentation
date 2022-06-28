import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv


@hydra.main(
    config_path="configs",
    config_name="config",
)
def main(config):

    load_dotenv()

    import wandb

    wandb.init(
        name=config.general.name,
        project=config.general.project,
        entity=config.general.entity,
        config=OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        ),  # type:ignore
    )

    config = wandb.config
    from src.utils import print_config

    print_config(config)

    from src.train import train

    train(config)

    wandb.finish()


if __name__ == "__main__":
    main()
