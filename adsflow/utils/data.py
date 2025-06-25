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
    pbc: bool
        Wheter to use periodic boundary conditions
    """

    def __init__(
            self, 
            path: str = None, 
            eta: int = 1.00, 
            pbc: bool = True,
        ):
        super().__init__()

        self.ds = LmdbDataset({"src": path})
        self.M = int(len(self.ds) * eta)
        self.pbc = pbc
    
    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        samp = self.ds[idx]

        data = Data(
            atomic_numbers = samp['atomic_numbers'].clone().detach().long(),
            natoms = torch.tensor(samp['natoms'], dtype=torch.long),
            cell = samp['cell'].clone().detach().long(),
            pbc = torch.tensor([[self.pbc, self.pbc, self.pbc]]),
            tags = samp['tags'].clone().detach().long,
            fixed = samp['fixed'].clone().detach().long,
            pos = samp['pos'].clone().detach().long(),
            pos_relaxed = samp['pos_relaxed'].clone().detach().long(),
        )
        
        return data

### Testing ###
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    dataset = OCP_DataLoader("/workspace/adsflow/dataset/mini/data.lmdb", pbc=True)
    loader = DataLoader(dataset, batch_size=25)

    with open("testing.txt", "w") as f:
        for batch in loader:
            f.write(str(batch) + "\n\n")
