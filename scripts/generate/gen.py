import hydra
import time
from adsflow.helm.configure import Config
from adsflow.helm.cli import main as generate

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: Config) -> None:
    start_time = time.time()

    # Run generation
    generate(cfg)

    end_time = time.time()
    elapsed = end_time - start_time

    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n✅ Generation completed in {int(hrs)}h {int(mins)}m {secs:.2f}s\n")


if __name__ == "__main__":
    main()
