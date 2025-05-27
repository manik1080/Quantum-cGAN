import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


class ProteinMoleculeDataset(Dataset):
    def __init__(self, folder_path, output=['adj', 'features', 'sequence'], sample=None):
        self.folder_path = folder_path
        self.file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        self.output = output
        self.sample = sample
        if sample:
            self.file_list = self.file_list[:sample]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)

        smiles = str(data['smiles'])
        uniprot = str(data['uniprot'])
        sequence = torch.tensor(data['sequence'])
        adj = torch.tensor(data['adj'])
        features = torch.tensor(data['features'])
        d = {
            'smiles': smiles,
            'uniprot': uniprot,
            'sequence': sequence,
            'adj': adj.permute(1, 2, 0),
            'features': features
        }

        out = [d[i] for i in self.output]

        return out
        '''{
            'smiles': smiles,
            'uniprot': uniprot,
            'sequence': sequence,
            'adj': adj.permute(1, 2, 0),
            'features': features
        }'''

if __name__ == '__main__':
    from config import folder_path
    dataset = ProteinMoleculeDataset(folder_path, output=['smiles', 'uniprot', 'sequence', 'adj', 'features'])
    print("Size of dataset: ", len(dataset))
    for key, val in zip(['smiles', 'uniprot', 'sequence', 'adj', 'features'], dataset[0]):
        try:
            print(f"{key.capitalize()} shape: ", val.shape)
        except AttributeError:
            print(f"{key.capitalize()} Length: ", len(val))
