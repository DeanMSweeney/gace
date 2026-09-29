import hydra
import time

from gace.helm.config import Config
from gace.helm.cli import main as trainer


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: Config) -> None:
    # Run training
    trainer(cfg)

if __name__ == "__main__":
    main()

