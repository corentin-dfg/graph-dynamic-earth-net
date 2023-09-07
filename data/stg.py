import glob
import copy

import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric as pyg

from skimage import segmentation, morphology
from scipy.ndimage import center_of_mass

import rasterio

class STG(pyg.data.HeteroData):
    r"""A data object describing a spatio-temporal graph as a heterogeneous graph,
    with one node types ('region') and two edge types ('spatial' and 'temporal').
    It takes as input the directories' path containing the sequence of TIF files
    with multispectral images and annotated labels. The `k_slic` and `smoothness`
    parameters are passed to the SLIC algorithm managed by OpenCV (the number of pixels of a SLIC region will be approximately equal to k_slic²). The `threshold_iou`
    parameter filters the temporal edges by their IoU attributes (minimum value to exist).
    `label_mode` can be 'multi', 'majority' or 'center' depending if multilabel or monolabel is enable.
    We can save or not the superpixel map of the SITS to allow a faster reconstruction (without rerun the superpixel algorithm) of the datacube from the STG but it needs more memory to store it."""

    def __init__(self, img_dir=None, label_dir=None, n_segments=1000, compactness=0.1, save_spxl=True):
        super().__init__()

        if not img_dir is None and not label_dir is None: 
            self.img_dir = img_dir
            self.label_dir = label_dir
            ts_imgs = sorted(glob.glob(self.img_dir+"*.tif"))
            ts_labels = sorted(glob.glob(self.label_dir+"*.tif"))

            num_nodes = 0

            # Compute the first date graph
            x1, edge_index1, edge_attr1, y1, sp_order1, superpixels1, pos1 = STG.__img_to_graph__(ts_imgs[0],ts_labels[0], n_segments, compactness)
            pos = torch.cat((pos1,torch.zeros((x1.shape[0],1))),dim=1)
            spxl = superpixels1[:,:,None]
            edge_temp_index1 = torch.zeros((2,0))
            edge_temp_attr1 = torch.zeros((0,1))
            num_nodes1 = len(sp_order1)

            # Concatenate the other date graphs            
            for j in range(1,len(ts_imgs)):
                num_nodes += num_nodes1

                # Get the spatial graph at time j
                x2, edge_index2, edge_attr2, y2, sp_order2, superpixels2, pos2 = STG.__img_to_graph__(ts_imgs[j],ts_labels[j], n_segments, compactness)

                # Concatenate new spatial graph to the STG
                edge_index2 += num_nodes
                sp_order2 += num_nodes
                x1 = torch.cat((x1,x2),dim=0)
                edge_index1 = torch.cat((edge_index1,edge_index2),dim=1)
                edge_attr1 = torch.cat((edge_attr1,edge_attr2),dim=0)
                y1 = torch.cat((y1,y2),dim=0)
                sp_order1 = torch.cat((sp_order1,sp_order2),dim=0)
                spxl = torch.cat((spxl,superpixels2[:,:,None]),dim=2)
                pos = torch.cat((pos,torch.cat((pos2,j*torch.ones((x2.shape[0],1))),dim=1)),dim=0)

                # Add temporal edges with attributes between time j-1 and j
                edge_temp_index2, inter = torch.unique(
                    torch.cat(
                        (torch.as_tensor(superpixels1.reshape((1,-1))+num_nodes-num_nodes1,dtype=int),
                            torch.as_tensor(superpixels2.reshape((1,-1))+num_nodes,dtype=int)
                        ),dim=0)
                    ,dim=1, return_counts=True)
                
                edge_temp_index2, inter = pyg.utils.to_undirected(edge_temp_index2, inter)

                _, count1 = torch.unique(torch.as_tensor(superpixels1.reshape((1,-1))), return_counts=True)
                _, count2 = torch.unique(torch.as_tensor(superpixels2.reshape((1,-1))), return_counts=True)
                edge_temp_attr2 = (inter/torch.cat((count1,count2))[edge_temp_index2[0]-(num_nodes-num_nodes1)])[:,None]

                edge_temp_index1 = torch.cat((edge_temp_index1,edge_temp_index2),dim=1)
                edge_temp_attr1 = torch.cat((edge_temp_attr1, edge_temp_attr2),dim=0)
                
                superpixels1 = superpixels2
                num_nodes1 = len(sp_order2)

            # Set the number of nodes
            self.num_nodes = num_nodes+num_nodes1

            # Set the attributes of the data structure
            self['region'].x = x1
            self['region','spatial','region'].edge_index=edge_index1.long()
            self['region','spatial','region'].edge_attr=edge_attr1
            self['region','temporal','region'].edge_index=edge_temp_index1.long()
            self['region','temporal','region'].edge_attr=edge_temp_attr1
            self['region'].y = y1
            self['region'].pos = pos
            self['region'].node_index = sp_order1
            if save_spxl:
                self.superpixels = spxl # Save each superpixels map

    def __slic__(arr, numsuperpixels, compactness):
        superpixels = segmentation.slic(arr, n_segments=numsuperpixels, compactness=compactness, min_size_factor=0.25, max_size_factor=1.0,  convert2lab=False, enforce_connectivity=True, channel_axis=2)
        return superpixels-1

    def __img_to_graph__(raw_path, gt_path, n_segments, compactness):
        # Opening data image and its corresponding labels
        with rasterio.open(raw_path, 'r') as tif:
            img = tif.read().transpose((1,2,0))-1
        with rasterio.open(gt_path, 'r') as tif:
            gt = tif.read().transpose((1,2,0))/255
            
        # Processing superpixel algorithm on multispectral input
        superpixels = STG.__slic__(img.astype(np.float32), n_segments, compactness)
        superpixels = torch.tensor(superpixels, dtype=torch.int)
        sp_order = torch.unique(superpixels)

        n_ch = img.shape[-1]

        # Processing the node features and label
        sp_intensity, sp_coord, sp_label, perim = [], [], [], []
        for seg in sp_order:
            mask = (superpixels == seg).squeeze()
            avg_value = torch.zeros(n_ch)
            std_value = torch.zeros(n_ch)
            max_value = torch.zeros(n_ch)
            min_value = torch.zeros(n_ch)
            for c in range(n_ch):
                avg_value[c] = np.mean(img[:, :, c][mask])
                std_value[c] = np.std(img[:, :, c][mask])
                max_value[c] = np.max(img[:, :, c][mask])
                min_value[c] = np.min(img[:, :, c][mask])
            cntr = center_of_mass(mask.numpy())  # row, col
            sp_coord.append(torch.tensor(cntr))
            
            sp_intensity.append(torch.cat((avg_value,
                                            std_value,
                                            max_value,
                                            min_value), -1))

            # Setting the superpixel label. Take the importance fraction of each class in the superpixel
            label_count = np.sum(gt[mask],axis=0)
            sp_label.append(label_count / np.sum(label_count))

        x = torch.stack(sp_intensity)
        y = torch.as_tensor(np.array(sp_label))

        # Processing the edge indices with a RAG and compute the edge weights (shared perimeter/target's perimeter)
        superpixels_bordered = np.full((superpixels.shape[0]+2,superpixels.shape[1]+2),-1)
        superpixels_bordered[1:-1,1:-1] = superpixels
        eroded = morphology.erosion(superpixels_bordered)
        dilated = morphology.dilation(superpixels_bordered)
        boundaries0 = (eroded != superpixels_bordered)
        boundaries1 = (dilated != superpixels_bordered)
        labels_small = torch.as_tensor(np.concatenate((eroded[boundaries0], superpixels_bordered[boundaries1])))
        labels_large = torch.as_tensor(np.concatenate((superpixels_bordered[boundaries0], dilated[boundaries1])))

        edge_index, inter = torch.unique(torch.stack((labels_small,labels_large)),dim=1, return_counts=True)
        inter = inter[edge_index[0,:]!=-1]
        edge_index = edge_index[:,edge_index[0,:]!=-1]

        _, perim = torch.unique(torch.stack((labels_small,labels_large)), return_counts=True)
        perim = perim[1:]

        edge_index, inter = pyg.utils.to_undirected(edge_index,inter[:,None])
        edge_attr = inter/perim[edge_index[1,:]][:,None]
                
        return x, edge_index, edge_attr, y, sp_order, superpixels, torch.stack(sp_coord)

    def datacube_from_node_classification(self,node_pred):
        assert hasattr(self,'superpixels')
        _, idx_sorted = torch.sort(self["region"].node_index, stable=True)
        return node_pred[idx_sorted[self.superpixels.long()]]

    def reconstruct_to_datacube(self):
        assert hasattr(self,'superpixels')
        return self.datacube_from_node_classification(self["region"].y)

    def from_heterogeneous(self, data):
        self.img_dir = data.img_dir
        self.label_dir = data.label_dir
        self['region'].x = data['region'].x
        self['region','spatial','region'].edge_index = data['region','spatial','region'].edge_index
        self['region','spatial','region'].edge_attr = data['region','spatial','region'].edge_attr
        self['region','temporal','region'].edge_index = data['region','temporal','region'].edge_index
        self['region','temporal','region'].edge_attr = data['region','temporal','region'].edge_attr
        self['region'].y = data['region'].y
        self['region'].pos = data['region'].pos
        self['region'].node_index = data['region'].node_index
        if hasattr(data,'superpixels'):
            self.superpixels = data.superpixels
        self.num_nodes = len(self['region'].node_index)
        return self

    def to_homogeneous(self, spatial=True, temporal=True):
        data = pyg.data.Data()
        data.x = self['region'].x
        data.edge_index = torch.zeros((2,0))
        data.edge_attr = torch.zeros((0,self['region','spatial','region'].edge_attr.shape[-1]+self['region','temporal','region'].edge_attr.shape[-1]))
        if spatial:
            data.edge_index = torch.cat((data.edge_index, self['region','spatial','region'].edge_index),dim=1)
            data.edge_attr = torch.cat((data.edge_attr, F.pad(self['region','spatial','region'].edge_attr,(0,self['region','temporal','region'].edge_attr.shape[-1]))),dim=0)
        if temporal:
            data.edge_index = torch.cat((data.edge_index, self['region','temporal','region'].edge_index),dim=1)
            data.edge_attr = torch.cat((data.edge_attr, F.pad(self['region','temporal','region'].edge_attr,(self['region','spatial','region'].edge_attr.shape[-1],0))),dim=0)
        data.y = self['region'].y
        data.pos = self['region'].pos
        data.node_index = self['region'].node_index
        if hasattr(self,'superpixels'):
            data.superpixels = self.superpixels
        data.num_nodes = self.num_nodes
        return data

    def to_spatial(self, tmin=None, tmax=None):
        data = self.clone(tmin=tmin, tmax=tmax)
        data['region','temporal','region'].edge_index = torch.zeros((2,0))
        data['region','temporal','region'].edge_attr = torch.zeros((0,2))
        return data

    def to_temporal(self, tmin=None, tmax=None):
        data = self.clone(tmin=tmin, tmax=tmax)
        data['region','spatial','region'].edge_index = torch.zeros((2,0))
        data['region','spatial','region'].edge_attr = torch.zeros((0,2))
        return data

    def clone(self, tmin=None, tmax=None):
        if tmin is None:
            tmin = 0
        if tmax is None:
            tmax = self['region'].pos[-1,-1].int()

        node_mask = torch.logical_and(tmin <= self['region'].pos[:,2],self['region'].pos[:,2] <= tmax)
        edge_spatial_mask = torch.isin(self['region', 'spatial', 'region'].edge_index,self['region'].node_index[node_mask])
        edge_spatial_mask = torch.logical_and(edge_spatial_mask[0], edge_spatial_mask[1])
        edge_temporal_mask = torch.isin(self['region', 'temporal', 'region'].edge_index,self['region'].node_index[node_mask])
        edge_temporal_mask = torch.logical_and(edge_temporal_mask[0], edge_temporal_mask[1])

        data = copy.deepcopy(self)
        data['region'].x = data['region'].x[node_mask]
        data['region','spatial','region'].edge_index = data['region','spatial','region'].edge_index[:,edge_spatial_mask]
        data['region','spatial','region'].edge_attr = data['region','spatial','region'].edge_attr[edge_spatial_mask]
        data['region','temporal','region'].edge_index = data['region','temporal','region'].edge_index[:,edge_temporal_mask]
        data['region','temporal','region'].edge_attr = data['region','temporal','region'].edge_attr[edge_temporal_mask]
        data['region'].y = data['region'].y[node_mask]
        data['region'].pos = data['region'].pos[node_mask]
        data['region'].node_index = data['region'].node_index[node_mask]
        if hasattr(data,'superpixels'):
            data.superpixels = data.superpixels[:,:,tmin:tmax+1]
        data.num_nodes = len(data['region'].node_index)
        return data

    def view(self, tmin=None, tmax=None):
        if tmin is None:
            tmin = 0
        if tmax is None:
            tmax = self['region'].pos[-1,-1].int()

        node_mask = torch.logical_and(tmin <= self['region'].pos[:,2],self['region'].pos[:,2] <= tmax)
        edge_spatial_mask = torch.isin(self['region', 'spatial', 'region'].edge_index,self['region'].node_index[node_mask])
        edge_spatial_mask = torch.logical_and(edge_spatial_mask[0], edge_spatial_mask[1])
        edge_temporal_mask = torch.isin(self['region', 'temporal', 'region'].edge_index,self['region'].node_index[node_mask])
        edge_temporal_mask = torch.logical_and(edge_temporal_mask[0], edge_temporal_mask[1])

        self['region'].x = self['region'].x[node_mask]
        self['region','spatial','region'].edge_index = self['region','spatial','region'].edge_index[:,edge_spatial_mask].long()
        self['region','spatial','region'].edge_attr = self['region','spatial','region'].edge_attr[edge_spatial_mask]
        self['region','temporal','region'].edge_index = self['region','temporal','region'].edge_index[:,edge_temporal_mask].long()
        self['region','temporal','region'].edge_attr = self['region','temporal','region'].edge_attr[edge_temporal_mask]
        self['region'].y = self['region'].y[node_mask]
        self['region'].pos = self['region'].pos[node_mask]
        self['region'].node_index = self['region'].node_index[node_mask]
        if hasattr(self,'superpixels'):
            self.superpixels = self.superpixels[:,:,tmin:tmax+1]
        self.num_nodes = len(self['region'].node_index)
        return self