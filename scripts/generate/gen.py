import hydra 
from adsflow.helm.configure import Config
from adsflow.helm.cli import main as generate

@hydra.main(config_path=None, config_name=None, version_base=None,)
def main(cfg: Config) -> None: 
    generate(cfg)

if __name__ == "__main__":
    main()
    
   
