import rich
import rich.tree
import rich.syntax
from omegaconf import OmegaConf


def print_config(config):
    tree = rich.tree.Tree("CONFIG")

    for field in config.keys():
        branch = tree.add(field)
        branch_content = config[field]
        if isinstance(branch_content, dict):
            branch.add(
                rich.syntax.Syntax(
                    OmegaConf.to_yaml(branch_content), "yaml", theme="solarized-light"
                )
            )
        else:
            branch.add(str(branch_content))

    rich.print(tree)
