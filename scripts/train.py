import hydra
import time
from adsflow.helm.configure import Config
from adsflow.helm.cli import main as train

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: Config) -> None:
    start_time = time.time()

    # Run training
    train(cfg)

    end_time = time.time()
    elapsed = end_time - start_time

    # Format nicely: hours, minutes, seconds
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n✅ Execution completed in {int(hrs)}h {int(mins)}m {secs:.2f}s\n")


if __name__ == "__main__":
    main()

