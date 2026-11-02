import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import hydra
from omegaconf import OmegaConf
import pathlib

from petal.workspace.base_workspace import BaseWorkspace

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_ROOT, 'petal', 'config'),
    config_name="petal_workspace"
)
def train(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    if cfg.task.dataset_path is None:
            raise ValueError(
                "For training, dataset_path cannot be null. "
                "Please specify a valid dataset path."
            )

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    train()
