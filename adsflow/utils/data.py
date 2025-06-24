import torch 
from torch_geometric.data import Data
from fairchem.core.datasets import LmdbDataset


class OCP_DataLoader():
    """
    This class loads OC20 dataset from the specified path and partitions it into troch_geometric Data objects 

    Arguments 
    ---------
    path: str
        Path to lmbd file
    eta: int
        Fraction of data for training 
    """

    def __init__(
            self, 
            path: str = None, 
            eta: int = 1.00, 
        ):
        super().__init__()

        self.ds = LmdbDataset({"src": path})
        self.M = int(len(self.ds) * eta)
    
    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        samp = self.ds[idx]

        data = Data(
            atomic_numbers = samp['atomic_numbers'].clone().detach().long(),
            natoms = torch.tensor(samp['natoms'], dtype=torch.long),
            cell = samp['cell'].clone().detach().long(),
            pbc = torch.tensor([[True, True, True]]),
            tags = samp['tags'].clone().detach().long,
            fixed = samp['fixed'].clone().detach().long,
            pos = samp['pos'].clone().detach().long(),
            pos_relaxed = samp['pos_relaxed'].clone().detach().long(),
        )
        
        return data

### Testing ###
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    dataset = OCP_DataLoader("/Users/deanmsweeney/Documents/CFM/AdsFlow/adsflow/dataset/mini/data.lmdb")
    loader = DataLoader(dataset, batch_size=25)

    for batch in loader: 
        print(batch)