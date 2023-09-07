import torch
import torch_geometric as pyg

from data.gden import GraphDynamicEarthNet

class ObjectDynamicEarthNet(torch.utils.data.Dataset):
    def __init__(self, den_root, gden_root, mode, n_segments=1000, compactness=0.1):
        """
        Dataset built from Graph Dynamic Earth Net by group all region features independently of their neighborhood
        Args:
            den_root: the root of the folder which contains planet imagery and labels
            gden_root: the root of the folder containing the PyG graph files
            mode: train/val/test -- selects the splits
            k_slic and smoothness: SLIC parameters to create STG (cf STG file)
        If the gden_root already contains the preprocessed spatio-temporal graph, the dataset will automatically load it.
        """
        self.gden = GraphDynamicEarthNet(den_root=den_root, gden_root=gden_root, mode=mode,
                                    n_segments=n_segments, compactness=compactness)
        self.x = torch.zeros((0,16))
        self.y = torch.zeros((0,7))
        for stg in self.gden:
            self.x = torch.cat((self.x, stg["region"].x))
            self.y = torch.cat((self.y, stg["region"].y))
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = pyg.data.HeteroData()
        data['region'].x = self.x[idx].unsqueeze(0)
        data['region'].y = self.y[idx].unsqueeze(0)
        data['region','spatial','region'].edge_index = torch.tensor([],dtype=torch.long)
        data['region','temporal','region'].edge_index = torch.tensor([],dtype=torch.long)        
        return data
    
