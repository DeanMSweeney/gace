from fairchem.core.datasets import LmdbDataset


# load your zero-shot test split
ood_ads = LmdbDataset({"src": "/ocean/projects/chm250007p/dsweeney/oc20/tutorial/data/is2re/train_100"})
print(len(ood_ads), ood_ads[0].keys())


