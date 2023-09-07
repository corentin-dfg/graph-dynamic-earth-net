import os
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric as pyg
from data.stg import STG

class ToTensorScaled(object):
    '''Convert a Image to a CHW ordered Tensor, scale the range to [0, 1]'''
    def __call__(self, im):
        im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        return torch.from_numpy(im)

class GraphDynamicEarthNet(pyg.data.Dataset):
    def __init__(self, den_root, gden_root, mode, n_segments=1000, compactness=0.1):
        """
        A dataset that built spatio-temporal graph from the SITS of Dynamic Earth Net
        Args:
            den_root: the root of the folder which contains planet imagery and labels
            gden_root: the root of the folder containing the PyG graph files
            mode: train/val/test -- selects the splits
            k_slic and smoothness: SLIC parameters to create STG (cf STG file)
        If the gden_root already contains the preprocessed spatio-temporal graph, the dataset will automatically load it.
        """
        
        self.mode = mode
        self.den_root = den_root
        self.gden_root = gden_root

        self.n_segments = n_segments
        self.compactness = compactness
        self.scalete = ToTensorScaled()

        # Statistics computed from the Dynamic Earth Net dataset to allow normalization
        self.mean = torch.as_tensor([689.2493277051232, 944.2717592896837, 1085.2940649863447, 2626.1396464362288])
        self.std = torch.as_tensor([271.8228109336546, 308.3809174781065, 421.25083012374427, 602.1772764866997])

        self.set_files()
        super().__init__(gden_root)
        

    @property
    def raw_file_names(self):
        return self.raws

    @property
    def label_file_names(self):
        return self.labels

    @property
    def processed_file_names(self):
        return self.processeds

    def set_files(self):
        self.raws, self.labels, self.processeds, self.train_processed_file_names, self.val_processed_file_names, self.test_processed_file_names = [], [], [], [], [], []
        for mode in ["train", "val", "test"]:
            file_list = os.path.join(self.gden_root, "splits", f"{mode}" + ".txt")
            file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
            raws, labels, processeds = list(zip(*file_list))
            self.raws += raws
            self.labels += labels
            self.processeds += processeds
            if mode=="train":
                self.train_processed_file_names += processeds
            elif mode=="val":
                self.val_processed_file_names += processeds
            elif mode=="test":
                self.test_processed_file_names += processeds

    def len(self):
        return len(list(open(os.path.join(self.gden_root, "splits", f"{self.mode}" + ".txt"), "r")))

    def process(self):
        print("Generation of spatio-temporal graphs from DynamicEarthNet's datacubes...")
        for i in tqdm(range(len(self.processed_file_names))):
            data = STG(os.path.join(self.den_root,self.raw_file_names[i]), os.path.join(self.den_root,self.label_file_names[i]), n_segments=self.n_segments, compactness=self.compactness)
            torch.save(data, os.path.join(self.gden_root, "processed/", self.processed_file_names[i]))

    def get(self, i):
        if self.mode=="train":
            file_name = self.train_processed_file_names[i]
        elif self.mode=="val":
            file_name = self.val_processed_file_names[i]
        elif self.mode=="test":
            file_name = self.test_processed_file_names[i]
        else:
            file_name = self.processed_file_names[i]
        stg = torch.load(os.path.join(self.gden_root, "processed/", file_name))

        # Normalise features
        stg["region"].x[:,0:4] = (stg["region"].x[:,0:4]-self.mean)/self.std #Mean features
        stg["region"].x[:,4:8] = stg["region"].x[:,4:8]/self.std #Std features
        stg["region"].x[:,8:12] = (stg["region"].x[:,8:12]-self.mean)/self.std #Min value features
        stg["region"].x[:,12:16] = (stg["region"].x[:,12:16]-self.mean)/self.std #Max value features

        # Fix dtype issue
        stg["region", "spatial", "region"].edge_index = stg["region", "spatial", "region"].edge_index.long()
        stg["region", "temporal", "region"].edge_index = stg["region", "temporal", "region"].edge_index.long()

        return stg

