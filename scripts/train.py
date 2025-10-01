import hydra 
from adsflow.helm.configure import Config
from adsflow.helm.cli import main as train

@hydra.main(config_path=None, config_name=None, version_base=None,)
def main(cfg: Config) -> None: 
    train(cfg)

if __name__ == "__main__":
    main()
    
   
