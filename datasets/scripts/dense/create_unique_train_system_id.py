import argparse
import pickle

from adsflow.utils.lmdb_loader import LazyLmdbDataset as LD
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a list of unique system IDs from the OC20 dense train split."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/datasets/oc20/dense/train_is2re",
        help="Path to OC20 IS2RE training set",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="/datasets/oc20/dense/pkls/oc20dense_mapping.pkl",
        help="Path to pickle file containing OC20 Dense mappings",
    )
    args = parser.parse_args()

    lmdb = LD({"src": args.src})

    with open(args.mapping, "rb") as f:
        mapping = pickle.load(f)

    system_ids = {}
    for i in tqdm(range(len(lmdb))):
        data = lmdb[i]
        config = data.sid
        system_id = mapping[config]["system_id"]
        try:
            system_ids[system_id] += 1
        except KeyError:
            system_ids[system_id] = 1

    with open("unique_train_system_id.txt", "w") as f:
        for system_id in system_ids.keys():
            f.write(system_id)
            f.write("\n")

    print(f"Done. {len(system_ids)} unique system IDs written to unique_train_system_id.txt")
