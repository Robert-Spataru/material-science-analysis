import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class MoleculeDataset(Dataset):
    def __init__(self, flattened_hydrogen_nitrogen, energy_array):
        self.flattened_hydrogen_nitrogen = flattened_hydrogen_nitrogen
        self.energy_array = energy_array
    
    def __len__(self):
        return len(self.flattened_hydrogen_nitrogen)
    
    def __getitem__(self, idx):
        hydrogen_nitrogen_vector = self.flattened_hydrogen_nitrogen[idx]
        energy = self.energy_array[idx]
        return hydrogen_nitrogen_vector, energy

class MoleculeDatasetGNN(Dataset):
    pass